"""The activation stream rotates by size instead of growing without bound."""

from gaia_engine.core import _rotate_activation_stream


def test_rotates_at_cap(tmp_path):
    p = tmp_path / "activation_stream.jsonl"
    p.write_text("x" * 100)
    (tmp_path / "activation_stream.jsonl.1").write_text("old")
    assert _rotate_activation_stream(str(p), max_bytes=100) is True
    assert not p.exists()
    assert (tmp_path / "activation_stream.jsonl.1").read_text() == "x" * 100


def test_below_cap_untouched(tmp_path):
    p = tmp_path / "activation_stream.jsonl"
    p.write_text("x" * 99)
    assert _rotate_activation_stream(str(p), max_bytes=100) is False
    assert p.read_text() == "x" * 99


def test_missing_file_is_fine(tmp_path):
    assert _rotate_activation_stream(str(tmp_path / "nope.jsonl"), max_bytes=1) is False


def test_zero_cap_disables_rotation(tmp_path):
    p = tmp_path / "activation_stream.jsonl"
    p.write_text("x" * 10)
    assert _rotate_activation_stream(str(p), max_bytes=0) is False
