"""
ROME — Rank-One Model Editing for GAIA.

Implements the core ROME algorithm: compute a rank-one update to a
specific MLP layer's weight matrix that changes one factual association
while minimally disturbing everything else.

Based on: "Locating and Editing Factual Associations in GPT"
(Meng et al., 2022). Simplified implementation for GAIA's use case.

Usage:
    from gaia_engine.rome import rome_edit

    edits = [
        {"prompt": "gaia-core runs on port", "target": " 6415",
         "subject": "gaia-core"},
        {"prompt": "The Nano/Reflex tier handles", "target": " simple queries",
         "subject": "Nano"},
    ]
    model = rome_edit(model, tokenizer, edits, layer=20)

The edit modifies the model weights in-place. The model can then be
saved as a new checkpoint (the edit is permanent in the weights).
"""

import logging
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("GAIA.ROME")


def find_mlp_module(model, layer_idx: int, proj: str = "up_proj"):
    """Find an MLP projection weight matrix at a specific layer.

    For ROME, we edit up_proj (or gate_proj) since its input dimension
    matches the hidden_size. down_proj takes expanded intermediate_size input.

    Works with both Qwen3 and Qwen3.5 architectures.
    """
    for path_template in [
        f"model.layers.{{}}.mlp.{proj}",           # Qwen3, LLaMA-style
        f"model.model.layers.{{}}.mlp.{proj}",     # Some wrapped models
    ]:
        path = path_template.format(layer_idx)
        parts = path.split(".")
        module = model
        try:
            for part in parts:
                module = getattr(module, part)
            if hasattr(module, 'weight'):
                return module
        except AttributeError:
            continue

    raise ValueError(f"Could not find MLP {proj} at layer {layer_idx}")


def compute_key_vector(model, tokenizer, prompt: str, subject: str,
                        layer_idx: int) -> torch.Tensor:
    """Compute the key vector k* for a subject at a specific layer.

    The key vector represents how the subject is encoded in the MLP
    at the target layer. This is what ROME will use to "locate" the
    factual association.
    """
    # Tokenize and find subject token positions
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    subject_ids = tokenizer.encode(subject, add_special_tokens=False)

    # Find subject position in the prompt
    input_list = input_ids[0].tolist()
    subject_start = None
    for i in range(len(input_list) - len(subject_ids) + 1):
        if input_list[i:i+len(subject_ids)] == subject_ids:
            subject_start = i
            break

    if subject_start is None:
        # Subject not found as exact tokens — use last token of prompt
        subject_start = len(input_list) - 1
        logger.warning("Subject '%s' not found in prompt tokens, using last position", subject)

    subject_end = subject_start + len(subject_ids) if subject_start is not None else len(input_list)

    # Hook to capture MLP input at the target layer
    key_vector = None
    hook_handle = None

    def hook_fn(module, input, output):
        nonlocal key_vector
        # input[0] is the hidden state entering the MLP
        # Take the mean across subject token positions
        key_vector = input[0][0, subject_start:subject_end].mean(dim=0).detach()

    # Find and hook the MLP module
    mlp = find_mlp_module(model, layer_idx)
    # Hook the parent MLP module, not just down_proj
    # We need the input to the MLP block
    for path in ["model.layers.{}.mlp", "model.model.layers.{}.mlp"]:
        try:
            parts = path.format(layer_idx).split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
            hook_handle = module.register_forward_hook(hook_fn)
            break
        except AttributeError:
            continue

    if hook_handle is None:
        raise ValueError(f"Could not hook MLP at layer {layer_idx}")

    # Forward pass to capture key vector
    with torch.no_grad():
        model(input_ids)

    hook_handle.remove()

    if key_vector is None:
        raise ValueError("Hook did not capture key vector")

    return key_vector


def compute_value_vector(model, tokenizer, prompt: str, target: str,
                          layer_idx: int) -> torch.Tensor:
    """Compute the target value vector v* that we want the MLP to output.

    This represents what the model SHOULD output at this layer for the
    given prompt+target combination.
    """
    # Concatenate prompt and target
    full_text = prompt + target
    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_len = prompt_ids.shape[1]

    # Hook to capture MLP output at the target layer for the target tokens
    value_vector = None
    hook_handle = None

    def hook_fn(module, input, output):
        nonlocal value_vector
        # output is the MLP output
        if isinstance(output, tuple):
            output = output[0]
        # Take the mean across target token positions
        value_vector = output[0, prompt_len-1:].mean(dim=0).detach()

    for path in ["model.layers.{}.mlp", "model.model.layers.{}.mlp"]:
        try:
            parts = path.format(layer_idx).split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
            hook_handle = module.register_forward_hook(hook_fn)
            break
        except AttributeError:
            continue

    if hook_handle is None:
        raise ValueError(f"Could not hook MLP at layer {layer_idx}")

    with torch.no_grad():
        model(input_ids)

    hook_handle.remove()

    if value_vector is None:
        raise ValueError("Hook did not capture value vector")

    return value_vector


def rome_edit(model, tokenizer, edits: List[Dict], layer_idx: int = 20,
              clamp_factor: float = 0.1) -> Dict:
    """Apply ROME edits to the model's MLP weights.

    Each edit is: {"prompt": str, "target": str, "subject": str}
    - prompt: the context leading to the factual statement
    - target: what the model SHOULD say (the correction)
    - subject: the entity being edited

    The edit modifies model weights in-place.

    Returns summary of edits applied.
    """
    # Edit up_proj: its input is hidden_size (4096), matching our key vectors
    # up_proj.weight is [intermediate_size, hidden_size] e.g. [12288, 4096]
    mlp_module = find_mlp_module(model, layer_idx, proj="up_proj")
    W = mlp_module.weight.data  # [intermediate_size, hidden_size]

    # Handle quantized weights (NF4/int8) — dequantize for editing, re-quantize after
    if W.dtype in (torch.uint8, torch.int8):
        logger.info("ROME: NF4/int8 weights detected — editing via dequantized proxy")
        # For quantized models, we can't directly edit weights.
        # Instead, we'll edit the compute_dtype buffer if available,
        # or apply the edit as a bias offset.
        if hasattr(mlp_module, 'weight') and hasattr(mlp_module.weight, 'quant_state'):
            # bitsandbytes NF4 — dequantize, edit, requantize
            import bitsandbytes as bnb
            W_float = bnb.functional.dequantize_4bit(W, mlp_module.weight.quant_state).float()
            original_norm = W_float.norm().item()
            logger.info("ROME target: layer %d up_proj (dequantized), shape %s", layer_idx, list(W_float.shape))
        else:
            logger.error("ROME: Cannot dequantize weights — unsupported format")
            return {"edits_attempted": len(edits), "edits_applied": 0, "error": "quantized weights unsupported"}
    else:
        W_float = W.float()
        original_norm = W_float.norm().item()
        logger.info("ROME target: layer %d up_proj, shape %s", layer_idx, list(W.shape))

    results = []

    for edit in edits:
        prompt = edit["prompt"]
        target = edit["target"]
        subject = edit.get("subject", "")

        try:
            # Step 1: Compute key vector k* (how subject is encoded)
            k = compute_key_vector(model, tokenizer, prompt, subject, layer_idx)

            # Step 2: Compute current value (what up_proj outputs for this key)
            v_current = W @ k  # [intermediate_size]

            # Step 3: Compute target value
            # For up_proj, we want to steer the output to encode the target
            # We use the delta between current and desired hidden states
            v_target = compute_value_vector(model, tokenizer, prompt, target, layer_idx)

            # v_target is hidden_size but we need intermediate_size
            # Project v_target through up_proj to get the target intermediate
            v_target_proj = W @ v_target[:W.shape[1]]  # Use hidden_size portion

            # Step 4: Compute the rank-one update
            # ΔW = (v_target_proj - v_current) ⊗ k / (k · k)
            delta_v = v_target_proj - v_current
            k_norm_sq = k @ k
            if k_norm_sq < 1e-10:
                logger.warning("ROME: key vector near-zero for '%s', skipping", prompt[:40])
                results.append({"prompt": prompt[:40], "status": "skipped", "reason": "zero key"})
                continue

            # Apply with clamping to prevent catastrophic updates
            update = clamp_factor * torch.outer(delta_v, k) / k_norm_sq

            # Step 5: Apply the rank-one update
            W += update.to(W.dtype)

            # Verify the edit magnitude
            edit_norm = update.norm().item()
            relative_change = edit_norm / original_norm

            logger.info("ROME edit applied: '%s' → '%s' (layer %d, Δ=%.4f, relative=%.6f)",
                        prompt[:40], target[:20], layer_idx, edit_norm, relative_change)

            results.append({
                "prompt": prompt[:60],
                "target": target[:40],
                "subject": subject,
                "layer": layer_idx,
                "edit_norm": round(edit_norm, 4),
                "relative_change": round(relative_change, 6),
                "status": "applied",
            })

        except Exception as e:
            logger.warning("ROME edit failed for '%s': %s", prompt[:40], e)
            results.append({"prompt": prompt[:40], "status": "failed", "error": str(e)})

    final_norm = mlp_module.weight.data.norm().item()
    total_change = abs(final_norm - original_norm) / original_norm

    return {
        "edits_attempted": len(edits),
        "edits_applied": sum(1 for r in results if r["status"] == "applied"),
        "total_weight_change": round(total_change, 6),
        "original_norm": round(original_norm, 2),
        "final_norm": round(final_norm, 2),
        "results": results,
    }
