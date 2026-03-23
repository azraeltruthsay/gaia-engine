"""
Thought Composer — KV cache composability for the GAIA Engine.

Merges KV cache states from different cognitive contexts, enabling
GAIA to combine separate trains of thought into unified understanding.

Three merge strategies based on layer type:

1. Full Attention Layers (KV cache):
   - Identity dedup: skip duplicate prefix tokens from source B
   - Concatenate unique KV states along sequence dimension
   - Position indices are implicit in attention — no re-indexing needed
     for causal attention (each token attends to all before it)

2. DeltaNet Recurrent Layers:
   - Weighted average of recurrent states (both captured the identity,
     but diverged on the content — averaging preserves both signals)
   - Conv states averaged similarly

3. Identity-aware merging:
   - Caller specifies how many prefix tokens are shared (identity segment)
   - Shared prefix KV kept from the "primary" thought
   - Unique content from "secondary" thought appended after
"""

import logging
import copy
from typing import Optional, Tuple

import torch

logger = logging.getLogger("GAIA.ThoughtComposer")


def compose_thoughts(
    primary_kv,          # KV state from the primary thought (the one we're resuming)
    secondary_kv,        # KV state from the secondary thought (the one with new knowledge)
    shared_prefix_len: int = 0,  # Number of tokens shared (identity prefix)
    primary_weight: float = 0.6, # Weight for primary in DeltaNet averaging
    secondary_weight: float = 0.4,
) -> Tuple:
    """Compose two KV cache states into a unified cognitive state.

    Args:
        primary_kv: The thought being resumed (e.g., architecture discussion)
        secondary_kv: The thought with new knowledge (e.g., document review)
        shared_prefix_len: Tokens shared between both (identity prefix)
        primary_weight: Weight for primary in recurrent state averaging
        secondary_weight: Weight for secondary in recurrent state averaging

    Returns:
        Merged KV cache state that can be passed to model.forward()
    """
    # Detect cache type
    cache_type = type(primary_kv).__name__

    if cache_type == "DynamicCache":
        # Standard transformer (Qwen3/Prime) — simple KV concatenation
        return _compose_standard(primary_kv, secondary_kv, shared_prefix_len)

    elif cache_type == "Qwen3_5DynamicCache":
        # Hybrid attention (Qwen3.5/Core/Nano) — mixed strategy
        return _compose_hybrid(primary_kv, secondary_kv, shared_prefix_len,
                                primary_weight, secondary_weight)

    else:
        logger.warning("Unknown cache type %s — returning primary unchanged", cache_type)
        return primary_kv


def _compose_standard(primary_kv, secondary_kv, shared_prefix_len: int):
    """Compose standard DynamicCache (Qwen3/Prime).

    Strategy: concatenate KV states along sequence dimension,
    deduplicating the shared identity prefix.
    """
    composed = copy.deepcopy(primary_kv)

    for layer_idx in range(len(composed.key_cache)):
        pk = composed.key_cache[layer_idx]
        pv = composed.value_cache[layer_idx]
        sk = secondary_kv.key_cache[layer_idx]
        sv = secondary_kv.value_cache[layer_idx]

        if pk is None or sk is None:
            continue

        # Skip the shared prefix from secondary (identity dedup)
        unique_k = sk[:, :, shared_prefix_len:, :]
        unique_v = sv[:, :, shared_prefix_len:, :]

        # Concatenate: primary full + secondary unique
        composed.key_cache[layer_idx] = torch.cat([pk, unique_k], dim=2)
        composed.value_cache[layer_idx] = torch.cat([pv, unique_v], dim=2)

    new_len = composed.key_cache[0].shape[2] if composed.key_cache[0] is not None else 0
    logger.info("COMPOSE (standard): %d + %d (-%d shared) = %d tokens",
                primary_kv.get_seq_length(), secondary_kv.get_seq_length(),
                shared_prefix_len, new_len)
    return composed


def _compose_hybrid(primary_kv, secondary_kv, shared_prefix_len: int,
                     pw: float = 0.6, sw: float = 0.4):
    """Compose Qwen3_5DynamicCache (hybrid DeltaNet + attention).

    Full attention layers: concatenate KV (with identity dedup)
    DeltaNet layers: weighted average of recurrent + conv states
    """
    composed = copy.deepcopy(primary_kv)

    num_layers = len(composed.key_cache)

    for layer_idx in range(num_layers):
        pk = composed.key_cache[layer_idx]
        sk = secondary_kv.key_cache[layer_idx]

        if pk is not None and sk is not None:
            # Full attention layer — concatenate with dedup
            pv = composed.value_cache[layer_idx]
            sv = secondary_kv.value_cache[layer_idx]

            unique_k = sk[:, :, shared_prefix_len:, :]
            unique_v = sv[:, :, shared_prefix_len:, :]

            composed.key_cache[layer_idx] = torch.cat([pk, unique_k], dim=2)
            composed.value_cache[layer_idx] = torch.cat([pv, unique_v], dim=2)

        # DeltaNet layers — weighted average of recurrent states
        pr = composed.recurrent_states[layer_idx]
        sr = secondary_kv.recurrent_states[layer_idx]

        if pr is not None and sr is not None:
            if isinstance(pr, torch.Tensor) and isinstance(sr, torch.Tensor):
                composed.recurrent_states[layer_idx] = pw * pr + sw * sr
            elif isinstance(pr, tuple) and isinstance(sr, tuple):
                composed.recurrent_states[layer_idx] = tuple(
                    pw * p + sw * s for p, s in zip(pr, sr)
                )

        # Conv states — weighted average
        pc = composed.conv_states[layer_idx]
        sc = secondary_kv.conv_states[layer_idx]

        if pc is not None and sc is not None:
            composed.conv_states[layer_idx] = pw * pc + sw * sc

    # Log stats
    full_attn_layers = [i for i in range(num_layers) if composed.key_cache[i] is not None]
    if full_attn_layers:
        new_len = composed.key_cache[full_attn_layers[0]].shape[2]
        logger.info("COMPOSE (hybrid): %d full-attn layers concatenated (%d tokens), "
                     "%d DeltaNet layers averaged (weights: %.1f/%.1f)",
                     len(full_attn_layers), new_len,
                     num_layers - len(full_attn_layers), pw, sw)

    return composed


def estimate_composed_size(primary_kv, secondary_kv, shared_prefix_len: int) -> dict:
    """Estimate memory requirements for a composed thought."""
    # Count tokens in full attention layers
    p_len = 0
    s_len = 0

    for i in range(len(primary_kv.key_cache)):
        if primary_kv.key_cache[i] is not None:
            p_len = primary_kv.key_cache[i].shape[2]
            break

    for i in range(len(secondary_kv.key_cache)):
        if secondary_kv.key_cache[i] is not None:
            s_len = secondary_kv.key_cache[i].shape[2]
            break

    unique_from_secondary = max(0, s_len - shared_prefix_len)
    total_tokens = p_len + unique_from_secondary

    # Rough memory estimate (per KV layer: 2 * num_heads * seq_len * head_dim * 2 bytes)
    num_kv_layers = sum(1 for k in primary_kv.key_cache if k is not None)
    if num_kv_layers > 0 and primary_kv.key_cache[0] is not None:
        num_heads = primary_kv.key_cache[0].shape[1]
        head_dim = primary_kv.key_cache[0].shape[3]
        bytes_per_token = 2 * num_heads * head_dim * 2 * num_kv_layers  # K+V, bf16
        total_bytes = bytes_per_token * total_tokens
    else:
        total_bytes = 0

    return {
        "primary_tokens": p_len,
        "secondary_tokens": s_len,
        "shared_prefix": shared_prefix_len,
        "unique_from_secondary": unique_from_secondary,
        "composed_tokens": total_tokens,
        "estimated_mb": round(total_bytes / (1024 * 1024), 2),
    }
