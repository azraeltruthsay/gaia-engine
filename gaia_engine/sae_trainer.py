"""
SAE Atlas Trainer — builds interpretable feature maps of GAIA's weight geometry.

Phase 1: Record activations from diverse prompts at target layers
Phase 2: Train a Sparse Autoencoder to decompose activations into features
Phase 3: Store the feature dictionary as an atlas

The atlas maps neuron indices to interpretable features, enabling:
- SAE-guided ROME (calibrated weight surgery)
- Drift detection (compare atlases across training passes)
- Real-time feature monitoring (polygraph with labels)
- Precision abliteration (find and suppress specific refusal circuits)

Usage:
    from gaia_engine.sae_trainer import SAETrainer

    trainer = SAETrainer(model, tokenizer, device="cuda")
    trainer.record_activations(prompts, layers=[6, 12, 18, 23])
    atlas = trainer.train_sae(hidden_size=2048, num_features=4096)
    trainer.save_atlas("/shared/atlas/core")
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("GAIA.SAE")


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for decomposing model activations into interpretable features.

    Architecture: input → encoder (expand) → ReLU → [top-k] → decoder (compress)
    The encoder expands hidden_size → num_features (overcomplete basis).

    Sparsity modes:
      - L1 (k=None): ReLU activations, sparsity from an external L1 penalty
        (sparsity_weight in train_sae). Tends toward dense/polysemantic.
      - top-k (k=int): keep only the k strongest features per sample, zero the
        rest. Enforces L0=k DIRECTLY — no penalty to tune — giving sparse,
        discriminative (monosemantic-leaning) features. The modern fix.
    """

    def __init__(self, hidden_size: int, num_features: int, k: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.k = k if (k and 0 < k < num_features) else None

        self.encoder = nn.Linear(hidden_size, num_features)
        self.decoder = nn.Linear(num_features, hidden_size)

        # Initialize decoder as transpose of encoder (tied weights init)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = F.relu(self.encoder(x))
        if self.k is not None:
            # Keep the top-k features per row; zero everything else.
            vals, idx = torch.topk(encoded, self.k, dim=-1)
            encoded = torch.zeros_like(encoded).scatter_(-1, idx, vals)
        return encoded

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode → [top-k] → decode. Returns (reconstructed, encoded)."""
        encoded = self._encode(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activation strengths (top-k applied) without reconstructing."""
        return self._encode(x)


class SAETrainer:
    """Records activations and trains SAE to build feature atlases."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Recorded activations: {layer_idx: [tensor, tensor, ...]}
        self.activations: Dict[int, List[torch.Tensor]] = {}

        # Trained SAEs: {layer_idx: SparseAutoencoder}
        self.saes: Dict[int, SparseAutoencoder] = {}

        # Feature labels: {layer_idx: {feature_idx: label}}
        self.feature_labels: Dict[int, Dict[int, str]] = {}

    def record_activations(self, prompts: List[str], layers: List[int],
                            system_prompt: str = "You are GAIA, a sovereign AI.",
                            batch_size: int = 1) -> Dict:
        """Record hidden state activations for a corpus of prompts.

        Captures the last token's hidden state at each target layer
        for every prompt. These become training data for the SAE.
        """
        self.model.eval()
        self.activations = {l: [] for l in layers}
        total_tokens = 0

        logger.info("Recording activations for %d prompts at layers %s", len(prompts), layers)
        start = time.time()

        from gaia_engine.core import ChatFormatter
        _fmt = ChatFormatter(self.tokenizer)

        for i, prompt_text in enumerate(prompts):
            full = (_fmt.format_system(system_prompt) + "\n"
                    + _fmt.format_message("user", prompt_text) + "\n"
                    + _fmt.assistant_prefix(enable_thinking=True))
            ids = self.tokenizer.encode(full, return_tensors="pt").to(self.device)
            total_tokens += ids.shape[1]

            with torch.no_grad():
                out = self.model(ids, output_hidden_states=True)

            hidden_states = out.hidden_states

            for layer_idx in layers:
                if layer_idx < len(hidden_states):
                    # Capture ALL token positions (not just last) for richer data
                    hs = hidden_states[layer_idx][0].detach().cpu()  # [seq_len, hidden_size]
                    self.activations[layer_idx].append(hs)

            if (i + 1) % 50 == 0:
                logger.info("  Recorded %d/%d prompts (%d tokens)",
                            i + 1, len(prompts), total_tokens)

        elapsed = time.time() - start
        stats = {
            "prompts": len(prompts),
            "tokens": total_tokens,
            "layers": layers,
            "elapsed_s": round(elapsed, 1),
            "activations_per_layer": {
                l: sum(a.shape[0] for a in acts)
                for l, acts in self.activations.items()
            },
        }
        logger.info("Recording complete: %d prompts, %d tokens, %.1fs", len(prompts), total_tokens, elapsed)
        return stats

    def record_activations_gguf(self, prompts: List[str], layers: List[int],
                                 backend=None, n_embd: Optional[int] = None,
                                 capture_all_tokens: bool = False,
                                 system_prompt: str = "You are GAIA, a sovereign AI.",
                                 chat_format: bool = True) -> Dict:
        """Record per-layer residual activations from the GGUF/llama.cpp backend.

        The GGUF twin of record_activations(): captures the residual stream via
        gaia_cpp's capture_hidden and populates self.activations in the SAME
        {layer_idx: [tensor]} shape that train_sae() consumes — so we can build SAE
        atlases on the *production* (quantized, CPU) activations, not just the
        safetensors GPU model.

        Capture mechanics (gaia_cpp): generate with max_tokens=0 so the prompt
        prefill states survive (the autoregressive loop would otherwise overwrite
        the capture with the last generated token). NOTE: the current
        hidden_state_capture.hpp grabs only the LAST prompt token's state per layer
        (one n_embd vector), so each prompt yields 1 sample/layer — use a large
        corpus, or upgrade the C++ capture to all-token for ~seq_len× more samples.
        If that upgrade lands, pass `n_embd` and this recorder reshapes per-token.

        Layer indexing: gaia_cpp 'l_out-N' == transformers hidden_states[N+1]; pick
        `layers` consistently with the safetensors run when comparing atlases.

        Args:
            prompts: corpus.
            layers: ggml layer indices to keep (others captured are discarded).
            backend: a gaia_cpp.LlamaCppBackend-like object exposing
                generate(prompt, max_tokens, capture_hidden) -> result with
                .hidden_states {layer:int -> 1-D float array} and .prompt_tokens.
                Defaults to self.model.
        """
        import numpy as np

        backend = backend or self.model
        if not hasattr(backend, "generate"):
            raise TypeError("record_activations_gguf needs a gaia_cpp backend with .generate(); "
                            "got %r" % type(backend))

        # xzi: opt into all-token capture (n_embd × n_tokens per prompt → ~seq_len×
        # more samples). Needs the gaia_cpp all-token build; resolve n_embd from the
        # backend so the per-token reshape is exact.
        if capture_all_tokens and hasattr(backend, "set_capture_all_tokens"):
            backend.set_capture_all_tokens(True)
            if n_embd is None and hasattr(backend, "n_embd"):
                try:
                    n_embd = int(backend.n_embd())
                except Exception:
                    pass

        self.activations = {l: [] for l in layers}
        total_tokens = 0
        logger.info("Recording GGUF activations for %d prompts at layers %s", len(prompts), layers)
        start = time.time()

        # Chat-format to match production prompting (best-effort).
        _fmt = None
        if chat_format and self.tokenizer is not None:
            try:
                from gaia_engine.core import ChatFormatter
                _fmt = ChatFormatter(self.tokenizer)
            except Exception:
                _fmt = None

        for i, prompt_text in enumerate(prompts):
            if _fmt is not None:
                try:
                    full = (_fmt.format_system(system_prompt) + "\n"
                            + _fmt.format_message("user", prompt_text) + "\n"
                            + _fmt.assistant_prefix(enable_thinking=True))
                except Exception:
                    full = prompt_text
            else:
                full = prompt_text

            try:
                # max_tokens=0 → prefill only; capture holds the prompt states.
                result = backend.generate(full, max_tokens=0, capture_hidden=True)
            except Exception:
                logger.warning("GGUF generate failed on prompt %d; skipping", i, exc_info=True)
                continue

            n_tok = int(getattr(result, "prompt_tokens", 0) or 0)
            total_tokens += n_tok
            hs = getattr(result, "hidden_states", None) or {}

            for layer_idx in layers:
                flat = hs.get(layer_idx)
                if flat is None:
                    continue
                flat = np.asarray(flat, dtype=np.float32).reshape(-1)
                if flat.size == 0:
                    continue
                # Layout: only reshape per-token when n_embd is KNOWN and the
                # capture is genuinely all-token (flat == n_embd * n_tok). Size
                # alone can't disambiguate — a last-token n_embd vector can falsely
                # divide by a small n_tok — so default to last-token (one n_embd
                # vector), the current gaia_cpp behavior.
                if n_embd and n_tok > 0 and flat.size == n_embd * n_tok:
                    arr = flat.reshape(n_tok, n_embd)
                else:
                    arr = flat.reshape(1, flat.size)
                self.activations[layer_idx].append(torch.from_numpy(arr.copy()))

            if (i + 1) % 50 == 0:
                logger.info("  Recorded %d/%d prompts (%d tokens)", i + 1, len(prompts), total_tokens)

        # Restore the backend's default (last-token) capture so we don't affect
        # the polygraph or other consumers after recording.
        if capture_all_tokens and hasattr(backend, "set_capture_all_tokens"):
            try:
                backend.set_capture_all_tokens(False)
            except Exception:
                pass
        elapsed = time.time() - start
        stats = {
            "prompts": len(prompts),
            "tokens": total_tokens,
            "layers": layers,
            "backend": "gguf",
            "elapsed_s": round(elapsed, 1),
            "activations_per_layer": {
                l: sum(a.shape[0] for a in acts) for l, acts in self.activations.items()
            },
        }
        logger.info("GGUF recording complete: %d prompts, %d tokens, %.1fs",
                    len(prompts), total_tokens, elapsed)
        return stats

    def train_sae(self, layers: Optional[List[int]] = None,
                   num_features: int = 4096,
                   sparsity_weight: float = 0.01,
                   lr: float = 1e-3,
                   epochs: int = 50,
                   batch_size: int = 256,
                   top_k: Optional[int] = None) -> Dict:
        """Train Sparse Autoencoders on recorded activations.

        One SAE per layer, each learning an overcomplete basis of features.

        top_k: if set, each SAE keeps only the k strongest features per sample
            (L0=k enforced directly — no sparsity_weight tuning). Gives sparse,
            discriminative features; the fix for the dense/polysemantic atlas.
        """
        # Re-enable gradients — core.py disables them globally for inference,
        # but SAE training needs backprop through the autoencoder parameters.
        torch.set_grad_enabled(True)

        if layers is None:
            layers = list(self.activations.keys())

        results = {}

        for layer_idx in layers:
            acts_list = self.activations.get(layer_idx, [])
            if not acts_list:
                logger.warning("No activations for layer %d", layer_idx)
                continue

            # Concatenate all activations for this layer
            all_acts = torch.cat(acts_list, dim=0)  # [total_tokens, hidden_size]
            hidden_size = all_acts.shape[1]
            n_samples = all_acts.shape[0]

            logger.info("Training SAE for layer %d: %d samples × %d dims → %d features",
                        layer_idx, n_samples, hidden_size, num_features)

            # Ensure activations are regular float32 for SAE training
            # (BnB NF4 models may produce bfloat16 or non-standard dtypes)
            all_acts = all_acts.float()

            # Normalize activations
            mean = all_acts.mean(dim=0)
            std = all_acts.std(dim=0).clamp(min=1e-6)
            all_acts_norm = (all_acts - mean) / std

            # Create SAE in float32
            sae = SparseAutoencoder(hidden_size, num_features, k=top_k).to(dtype=torch.float32, device=self.device)
            optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

            # Move data to device — ensure contiguous float32, detached from model graph
            all_acts_device = all_acts_norm.detach().clone().to(self.device)

            # Train
            start = time.time()
            for epoch in range(epochs):
                # Shuffle
                perm = torch.randperm(n_samples)
                total_loss = 0
                total_recon = 0
                total_sparse = 0
                n_batches = 0

                for batch_start in range(0, n_samples, batch_size):
                    batch_idx = perm[batch_start:batch_start + batch_size]
                    batch = all_acts_device[batch_idx]

                    reconstructed, encoded = sae(batch)

                    # Reconstruction loss
                    recon_loss = F.mse_loss(reconstructed, batch)

                    # Sparsity loss (L1 on encoded features)
                    sparse_loss = sparsity_weight * encoded.abs().mean()

                    loss = recon_loss + sparse_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_recon += recon_loss.item()
                    total_sparse += sparse_loss.item()
                    n_batches += 1

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    avg_loss = total_loss / n_batches
                    avg_recon = total_recon / n_batches
                    avg_sparse = total_sparse / n_batches
                    with torch.no_grad():
                        _, enc = sae(all_acts_device[:1000])
                        active = (enc.mean(dim=0) > 0.01).sum().item()   # features ever-active
                        l0 = (enc > 1e-6).float().sum(dim=-1).mean().item()  # features/sample (the real measure)
                    logger.info("  Layer %d epoch %d/%d: loss=%.4f (recon=%.4f sparse=%.4f) L0=%.1f active=%d/%d",
                                layer_idx, epoch + 1, epochs, avg_loss, avg_recon, avg_sparse,
                                l0, active, num_features)

            elapsed = time.time() - start
            sae.eval()
            self.saes[layer_idx] = sae

            # Compute final stats
            with torch.no_grad():
                _, final_enc = sae(all_acts_device)
                active_features = (final_enc.mean(dim=0) > 0.01).sum().item()
                l0_per_sample = (final_enc > 1e-6).float().sum(dim=-1).mean().item()
                top_features = final_enc.mean(dim=0).topk(10)

            # Store normalization params for inference
            sae._norm_mean = mean
            sae._norm_std = std

            results[layer_idx] = {
                "samples": n_samples,
                "features": num_features,
                "active_features": active_features,
                "l0_per_sample": round(l0_per_sample, 2),
                "top_k": top_k,
                "final_loss": round(total_loss / n_batches, 4),
                "training_time_s": round(elapsed, 1),
                "top_feature_indices": top_features.indices.tolist(),
                "top_feature_strengths": [round(v, 4) for v in top_features.values.tolist()],
            }

        return results

    def save_atlas(self, path: str):
        """Save trained SAEs as a feature atlas."""
        atlas_dir = Path(path)
        atlas_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx, sae in self.saes.items():
            # Save SAE model
            torch.save({
                "encoder_weight": sae.encoder.weight.data.cpu(),
                "encoder_bias": sae.encoder.bias.data.cpu(),
                "decoder_weight": sae.decoder.weight.data.cpu(),
                "decoder_bias": sae.decoder.bias.data.cpu(),
                "norm_mean": sae._norm_mean,
                "norm_std": sae._norm_std,
                "hidden_size": sae.hidden_size,
                "num_features": sae.num_features,
                "labels": self.feature_labels.get(layer_idx, {}),
            }, atlas_dir / f"layer_{layer_idx}.pt")

        # Save metadata
        meta = {
            "layers": list(self.saes.keys()),
            "timestamp": time.time(),
            "model": getattr(self.model, 'name_or_path', 'unknown'),
        }
        (atlas_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        logger.info("Atlas saved to %s (%d layers)", path, len(self.saes))

    def load_atlas(self, path: str):
        """Load a previously saved SAE atlas from disk.

        Populates self.saes and self.feature_labels so analyze_prompt() works.
        """
        atlas_dir = Path(path)
        if not atlas_dir.exists():
            raise FileNotFoundError(f"Atlas directory not found: {path}")

        loaded = 0
        for sae_file in sorted(atlas_dir.glob("layer_*.pt")):
            layer_idx = int(sae_file.stem.split("_")[1])
            checkpoint = torch.load(sae_file, map_location=self.device, weights_only=True)

            sae = SparseAutoencoder(checkpoint["hidden_size"], checkpoint["num_features"])
            sae.encoder.weight.data = checkpoint["encoder_weight"].to(self.device)
            sae.encoder.bias.data = checkpoint["encoder_bias"].to(self.device)
            sae.decoder.weight.data = checkpoint["decoder_weight"].to(self.device)
            sae.decoder.bias.data = checkpoint["decoder_bias"].to(self.device)
            # Norm params stay on CPU (analyze_prompt does hs.cpu() - norm)
            sae._norm_mean = checkpoint.get("norm_mean", torch.zeros(checkpoint["hidden_size"])).cpu().float()
            sae._norm_std = checkpoint.get("norm_std", torch.ones(checkpoint["hidden_size"])).cpu().float()
            sae = sae.to(self.device).eval()

            self.saes[layer_idx] = sae
            labels = checkpoint.get("labels", {})
            if labels:
                self.feature_labels[layer_idx] = labels
            loaded += 1

        logger.info("Atlas loaded from %s (%d layers)", path, loaded)

    def analyze_prompt(self, prompt: str, layer_idx: int,
                        system_prompt: str = "You are GAIA, a sovereign AI.",
                        top_k: int = 20) -> Dict:
        """Analyze which SAE features activate for a specific prompt.

        This is the "polygraph with labels" — instead of raw neuron indices,
        you get interpretable feature activations.
        """
        if layer_idx not in self.saes:
            return {"error": f"No SAE for layer {layer_idx}"}

        sae = self.saes[layer_idx]

        from gaia_engine.core import ChatFormatter
        _fmt = ChatFormatter(self.tokenizer)
        full = (_fmt.format_system(system_prompt) + "\n"
                + _fmt.format_message("user", prompt) + "\n"
                + _fmt.assistant_prefix(enable_thinking=True))
        ids = self.tokenizer.encode(full, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
            hs = out.hidden_states[layer_idx][0, -1]  # Last token

            # Normalize
            hs_norm = (hs.cpu() - sae._norm_mean) / sae._norm_std
            features = sae.get_feature_activations(hs_norm.to(self.device))

        # Top active features
        top = features.topk(top_k)

        result = {
            "layer": layer_idx,
            "prompt": prompt[:80],
            "top_features": [],
        }
        for i in range(top_k):
            fidx = top.indices[i].item()
            strength = round(top.values[i].item(), 4)
            label = self.feature_labels.get(layer_idx, {}).get(str(fidx), f"feature_{fidx}")
            result["top_features"].append({
                "index": fidx,
                "strength": strength,
                "label": label,
            })

        return result
