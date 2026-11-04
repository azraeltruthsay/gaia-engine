"""What the activation stream writes for each token (GAIA_Project-6k9i)."""
import json

import torch

import gaia_engine.config as config
from gaia_engine.core import ActivationMonitor, _read_with_atlas, _write_activation


def _atlas(size=8, features=8):
    weight = torch.zeros(features, size)
    weight[2, 3] = 1.0  # feature 2 reads neuron 3
    return {"encoder_weight": weight, "encoder_bias": torch.zeros(features),
            "norm_mean": torch.zeros(size), "norm_std": torch.ones(size)}


def test_a_feature_comes_from_encoding_the_token_not_from_a_neuron_number():
    token = torch.zeros(8)
    token[3] = 5.0
    read = _read_with_atlas(token, _atlas())
    assert read["f"][0] == [2, 5.0] and all(v > 0 for _, v in read["f"])
    assert read["z"][0] == [3, 5.0]


def test_z_shows_the_unusual_neuron_rather_than_the_always_loud_one():
    atlas = _atlas()
    atlas["norm_mean"][0] = 100.0
    atlas["norm_std"][0] = 10.0
    token = torch.zeros(8)
    token[0] = 100.0  # loud, and as usual
    token[5] = 4.0  # quiet, and unusual
    assert _read_with_atlas(token, atlas)["z"][0][0] == 5


def _row(tmp_path, monkeypatch, **kw):
    path = tmp_path / "stream.jsonl"
    monkeypatch.setattr(config, "ACTIVATION_STREAM_PATH", str(path))
    hidden = {0: torch.ones(1, 1, 8) * 0.1, 35: torch.ones(1, 1, 8) * 500}
    hidden[35][0, 0, 3] = 900.0
    snap = ActivationMonitor().capture(hidden, atlas={35: _atlas()})
    _write_activation("prime", "x", 1, "sess", snap, {35: {2: "identity_self"}}, "turn", "/models/prime", **kw)
    return json.loads(path.read_text())


def test_every_sampled_layer_keeps_its_own_values(tmp_path, monkeypatch):
    row = _row(tmp_path, monkeypatch)
    assert set(row["layers"]) == {"0", "35"} and row["layers"]["0"]["n"]
    assert (row["session_id"], row["turn_id"], row["model"]) == ("sess", "turn", "/models/prime")


def test_a_feature_is_named_only_where_the_atlas_named_it(tmp_path, monkeypatch):
    row = _row(tmp_path, monkeypatch)
    assert [f["label"] for f in row["features"] if f["layer"] == 35][0] == "identity_self"
    assert any(f["label"].startswith("neuron_") and f["layer"] == 0 for f in row["features"])


def test_a_high_index_survives_a_half_precision_atlas_exactly():
    size, features = 4100, 8192
    atlas = {"encoder_weight": torch.zeros(features, size, dtype=torch.bfloat16),
             "encoder_bias": torch.zeros(features, dtype=torch.bfloat16),
             "norm_mean": torch.zeros(size, dtype=torch.bfloat16),
             "norm_std": torch.ones(size, dtype=torch.bfloat16)}
    atlas["encoder_weight"][8191, 4097] = 1.0
    token = torch.zeros(size)
    token[4097] = 3.0
    read = _read_with_atlas(token, atlas)
    assert read["z"][0][0] == 4097 and read["f"][0][0] == 8191
