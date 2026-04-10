"""MoE Expert Offloading — Active Expert Buffering for Gemma 4.

Keeps foundation layers (shared MLP, attention, router, embeddings, norms)
on GPU while offloading 128 private experts to CPU. JIT-transfers only the
top-8 routed experts per forward pass via an LRU cache.

This reduces GPU footprint from ~13GB (full NF4) to ~2.5GB, making the
Gemma 4 26B-A4B MoE viable on 16GB GPUs.
"""

import logging
import time
from collections import OrderedDict
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("gaia_engine.moe_offload")


def is_moe_model(config: dict) -> bool:
    """Check if model config indicates an MoE architecture."""
    # Gemma 4 MoE: text_config.enable_moe_block = True
    text_cfg = config.get("text_config", config)
    if text_cfg.get("enable_moe_block", False):
        return True
    # Also check for num_experts > 1
    if text_cfg.get("num_experts", 0) > 1:
        return True
    return False


def build_moe_device_map(model_path: str) -> Dict[str, str]:
    """Build a module-level device map for accelerate's dispatch_model.

    Places foundation modules (shared MLP, attention, router, norms) on GPU
    and expert modules on CPU. Uses module paths, not weight names.

    For Gemma 4 26B-A4B the module hierarchy is:
      model.language_model.layers.{N}.self_attn → GPU
      model.language_model.layers.{N}.mlp → GPU (shared expert)
      model.language_model.layers.{N}.router → GPU
      model.language_model.layers.{N}.experts → CPU
      model.language_model.layers.{N}.*_layernorm* → GPU
      model.language_model.embed_tokens → GPU
      model.language_model.norm → GPU
      model.vision_tower → CPU
    """
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    text_cfg = config.get("text_config", config)
    num_layers = text_cfg.get("num_hidden_layers", 30)

    # Use "auto"-style mapping: place everything on GPU by default,
    # then override only the experts to CPU. accelerate handles leaf
    # parameter assignment when the parent module is assigned.
    device_map = {
        "": "cuda:0",  # default: everything on GPU
    }

    # Override: experts → CPU (these are the large 3D packed tensors)
    for layer in range(num_layers):
        device_map[f"model.language_model.layers.{layer}.experts"] = "cpu"

    # Override: vision tower → CPU (loaded on demand)
    device_map["model.vision_tower"] = "cpu"
    device_map["model.embed_vision"] = "cpu"

    logger.info("MoE device map: foundation→GPU, %d expert modules→CPU, vision→CPU",
                num_layers)
    return device_map


class ExpertCache:
    """LRU cache for expert weight slices on GPU.

    Experts in Gemma 4 are packed as 3D tensors:
    - gate_up_proj: [num_experts, 2*intermediate, hidden]
    - down_proj: [num_experts, hidden, intermediate]

    This cache stores individual 2D slices (one per expert) on GPU,
    evicting least-recently-used entries when the budget is exceeded.
    """

    def __init__(self, max_cached: int = 48):
        self.max_cached = max_cached
        self._cache: OrderedDict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, layer_idx: int, expert_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached expert slices (gate_up, down) if present."""
        key = (layer_idx, expert_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, layer_idx: int, expert_idx: int,
            gate_up: torch.Tensor, down: torch.Tensor):
        """Cache expert slices on GPU, evicting LRU if needed."""
        key = (layer_idx, expert_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = (gate_up, down)
            return

        while len(self._cache) >= self.max_cached:
            evicted_key, evicted_tensors = self._cache.popitem(last=False)
            # Explicitly delete GPU tensors
            del evicted_tensors
            self._evictions += 1

        self._cache[key] = (gate_up, down)

    def clear(self):
        """Clear all cached expert slices."""
        self._cache.clear()
        torch.cuda.empty_cache()

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "cached": len(self._cache),
            "max_cached": self.max_cached,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / max(1, total), 3),
        }


def patch_experts_forward(model: nn.Module, expert_cache: ExpertCache):
    """Monkey-patch Gemma4TextExperts.forward() to use GPU-cached slices.

    The original forward indexes packed 3D CPU tensors:
        F.linear(x, self.gate_up_proj[expert_idx])

    The patched version:
    1. Checks the ExpertCache for a GPU-cached 2D slice
    2. If miss: slices from CPU tensor, transfers to GPU, caches
    3. Runs F.linear with the GPU slice
    """
    # Find all expert modules in the model
    expert_modules = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == "Gemma4TextExperts":
            # Extract layer index from module name
            # e.g., "language_model.model.layers.5.experts" or "model.layers.5.experts"
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
            if layer_idx is not None:
                expert_modules.append((layer_idx, module))

    if not expert_modules:
        logger.warning("No Gemma4TextExperts modules found — skipping MoE patch")
        return

    logger.info("Patching %d Gemma4TextExperts modules for JIT expert offloading", len(expert_modules))

    for layer_idx, expert_module in expert_modules:
        _patch_single_experts_module(layer_idx, expert_module, expert_cache)


def _patch_single_experts_module(layer_idx: int, module: nn.Module, cache: ExpertCache):
    """Replace the forward method of a single Gemma4TextExperts module."""
    original_forward = module.forward
    # Store references to the CPU tensors
    cpu_gate_up = module.gate_up_proj  # [128, 2*intermediate, hidden]
    cpu_down = module.down_proj        # [128, hidden, intermediate]
    act_fn = module.act_fn
    num_experts = module.num_experts

    def patched_forward(hidden_states: torch.Tensor,
                        top_k_index: torch.Tensor,
                        top_k_weights: torch.Tensor) -> torch.Tensor:
        """Forward with JIT expert transfer from CPU to GPU via cache."""
        final_hidden_states = torch.zeros_like(hidden_states)
        device = hidden_states.device

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Phase 1: batch-prefetch all needed experts to GPU
        expert_weights = {}  # {expert_idx: (gate_up_gpu, down_gpu)}
        needs_sync = False
        for expert_idx_tensor in expert_hit:
            eidx = expert_idx_tensor[0].item()
            if eidx == num_experts:
                continue
            cached = cache.get(layer_idx, eidx)
            if cached is not None:
                expert_weights[eidx] = cached
            else:
                # Non-blocking transfer — batch all transfers before sync
                gu = cpu_gate_up[eidx].to(device, non_blocking=True)
                dn = cpu_down[eidx].to(device, non_blocking=True)
                expert_weights[eidx] = (gu, dn)
                cache.put(layer_idx, eidx, gu, dn)
                needs_sync = True

        if needs_sync:
            torch.cuda.synchronize()

        # Phase 2: compute with all experts already on GPU
        for expert_idx_tensor in expert_hit:
            eidx = expert_idx_tensor[0].item()
            if eidx == num_experts or eidx not in expert_weights:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            current_state = hidden_states[token_idx]
            gate_up_gpu, down_gpu = expert_weights[eidx]

            gate, up = nn.functional.linear(current_state, gate_up_gpu).chunk(2, dim=-1)
            current_hidden_states = act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_gpu)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    module.forward = patched_forward
    logger.debug("Patched layer %d experts for JIT offloading", layer_idx)


def load_moe_offloaded(model_path: str, device: str = "cuda",
                       max_cached_experts: int = 16,
                       use_nf4: bool = True) -> Tuple[nn.Module, ExpertCache]:
    """Load an MoE model with expert offloading.

    Foundation (shared MLP, attention, router) loads to GPU.
    Experts load to CPU. JIT forward hook transfers top-8 per pass.

    Args:
        model_path: Path to the model directory
        device: Target device for foundation layers
        max_cached_experts: Max expert slices cached on GPU (per-layer × expert)
        use_nf4: Whether to use NF4 quantization for foundation layers

    Returns:
        (model, expert_cache) tuple
    """
    import json
    from pathlib import Path
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # Build the split device map
    device_map = build_moe_device_map(model_path)

    logger.info("Loading MoE model with expert offloading (NF4=%s, cache=%d)",
                use_nf4, max_cached_experts)
    start = time.time()

    # Strategy: Load bf16 to CPU, then manually .to(cuda) non-expert modules.
    #
    # NF4 + split device_map is blocked by transformers. accelerate dispatch
    # has compatibility issues with Params4bit. So we do it ourselves:
    # 1. Load bf16 to CPU (whole model in system RAM)
    # 2. Walk the module tree, move everything EXCEPT experts to GPU
    # 3. Experts stay on CPU; our patched forward handles JIT transfer
    #
    # bf16 foundation on GPU: ~4.5GB (attn + shared MLP + router + norms + embeds)
    # bf16 experts on CPU: ~45GB (128 experts × 30 layers in system RAM)
    logger.info("Loading bf16 model to CPU (low_cpu_mem_usage)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    # Manually move foundation to GPU, skip experts
    # Walk named_parameters for complete coverage (catches raw params like layer_scalar)
    logger.info("Moving foundation to GPU (skipping experts)...")
    _moved = 0
    _skipped = 0
    for name, param in list(model.named_parameters()):
        if ".experts." in name or ".experts" == name.split(".")[-1]:
            _skipped += 1
            continue
        if "vision_tower" in name or "embed_vision" in name:
            _skipped += 1
            continue
        if param.device.type != "cuda":
            # Move parameter to GPU in-place
            param.data = param.data.to("cuda")
            _moved += 1

    # Also move any buffers (non-parameter tensors like running stats)
    for name, buf in list(model.named_buffers()):
        if ".experts." in name:
            continue
        if "vision_tower" in name or "embed_vision" in name:
            continue
        if buf.device.type != "cuda":
            buf.data = buf.data.to("cuda")

    logger.info("Moved %d parameters to GPU, %d expert params remain on CPU", _moved, _skipped)
    torch.cuda.empty_cache()

    model.eval()
    elapsed = time.time() - start

    # Report memory usage
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / (1024**2)
        logger.info("MoE model loaded in %.1fs — GPU: %.0fMB (foundation only)", elapsed, gpu_mb)

    # Create expert cache and patch the forward methods
    expert_cache = ExpertCache(max_cached=max_cached_experts)
    patch_experts_forward(model, expert_cache)

    return model, expert_cache
