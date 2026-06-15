"""Tests for SAETrainer.record_activations_gguf — the GGUF→SAE recorder bridge.

Validates (with a mock gaia_cpp backend, no model / no GPU) that the GGUF
recorder populates self.activations in the SAME {layer: [tensor[rows, n_embd]]}
shape that train_sae() consumes — the contract that lets us build SAE atlases on
quantized CPU activations (SAE-atlas plan phase A0; GAIA_Project-h6f).
"""
import numpy as np
import torch

from gaia_engine.sae_trainer import SAETrainer


class _MockResult:
    def __init__(self, hidden_states, prompt_tokens):
        self.hidden_states = hidden_states
        self.prompt_tokens = prompt_tokens


class _MockBackend:
    """Stands in for gaia_cpp.LlamaCppBackend: last-token capture, 1 vec/layer."""

    def __init__(self, n_embd=8, layers=(6, 12), prompt_tokens=5, all_token=False):
        self._n_embd = n_embd
        self._layers = layers
        self._prompt_tokens = prompt_tokens
        self._all_token = all_token

    def generate(self, prompt, max_tokens=0, capture_hidden=False):
        hs = {}
        for L in self._layers:
            if self._all_token:
                hs[L] = np.arange(self._n_embd * self._prompt_tokens, dtype=np.float32)
            else:
                hs[L] = np.arange(self._n_embd, dtype=np.float32)
        return _MockResult(hs, self._prompt_tokens)


def _trainer():
    # tokenizer=None → chat formatter skipped; model unused (backend passed in).
    return SAETrainer(model=None, tokenizer=None, device="cpu")


def test_gguf_recorder_populates_train_sae_shape():
    t = _trainer()
    be = _MockBackend(n_embd=8, layers=(6, 12))
    t.record_activations_gguf(["p1", "p2", "p3"], layers=[6, 12], backend=be)
    for L in (6, 12):
        acts = t.activations[L]
        assert len(acts) == 3                          # one sample per prompt
        for a in acts:
            assert isinstance(a, torch.Tensor)
            assert a.ndim == 2 and a.shape == (1, 8)   # [rows, n_embd]


def test_gguf_recorder_stats():
    t = _trainer()
    be = _MockBackend(n_embd=8, layers=(6, 12), prompt_tokens=4)
    stats = t.record_activations_gguf(["a", "b"], layers=[6, 12], backend=be)
    assert stats["prompts"] == 2
    assert stats["backend"] == "gguf"
    assert stats["tokens"] == 8                         # 2 prompts × 4 tokens
    assert stats["activations_per_layer"][6] == 2       # rows accumulated


def test_gguf_recorder_all_token_reshape():
    # When n_embd is known and the flat capture == n_embd * n_tok, it reshapes
    # to [n_tok, n_embd] (the all-token build → ~seq_len× more samples).
    t = _trainer()
    be = _MockBackend(n_embd=8, layers=(6,), prompt_tokens=3, all_token=True)
    t.record_activations_gguf(["p"], layers=[6], backend=be, n_embd=8)
    a = t.activations[6][0]
    assert a.shape == (3, 8)                             # n_tok × n_embd


def test_gguf_recorder_skips_missing_layer():
    t = _trainer()
    be = _MockBackend(n_embd=8, layers=(6,))             # backend only has layer 6
    t.record_activations_gguf(["p"], layers=[6, 99], backend=be)
    assert len(t.activations[6]) == 1
    assert t.activations[99] == []                       # missing → empty, no crash


def test_gguf_recorder_rejects_non_backend():
    t = _trainer()
    try:
        t.record_activations_gguf(["p"], layers=[6], backend=object())
        assert False, "expected TypeError for backend without .generate()"
    except TypeError:
        pass


# ── top-k SAE: direct L0 sparsity (GAIA_Project-bup) ────────────────────────

def test_topk_sae_enforces_l0():
    from gaia_engine.sae_trainer import SparseAutoencoder
    sae = SparseAutoencoder(hidden_size=8, num_features=16, k=3)
    x = torch.randn(5, 8)
    recon, enc = sae(x)
    # Each sample keeps at most k=3 nonzero features (L0 ≤ k), not all 16.
    nnz = (enc.abs() > 1e-8).sum(dim=-1)
    assert int(nnz_max := nnz.max().item()) <= 3
    assert recon.shape == x.shape


def test_l1_sae_unchanged_when_k_none():
    from gaia_engine.sae_trainer import SparseAutoencoder
    sae = SparseAutoencoder(hidden_size=8, num_features=16, k=None)
    assert sae.k is None
    _, enc = sae(torch.randn(4, 8))
    assert enc.shape == (4, 16)   # ReLU mode: no top-k masking


def test_topk_ignored_if_k_ge_num_features():
    from gaia_engine.sae_trainer import SparseAutoencoder
    sae = SparseAutoencoder(hidden_size=8, num_features=16, k=99)
    assert sae.k is None          # k >= num_features → falls back to L1/ReLU
