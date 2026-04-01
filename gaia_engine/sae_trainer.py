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

    Architecture: input → encoder (expand) → ReLU → decoder (compress)
    The encoder expands hidden_size → num_features (overcomplete basis).
    Sparsity is enforced via L1 penalty on the encoded representation.
    """

    def __init__(self, hidden_size: int, num_features: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features

        self.encoder = nn.Linear(hidden_size, num_features)
        self.decoder = nn.Linear(num_features, hidden_size)

        # Initialize decoder as transpose of encoder (tied weights init)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode → decode.

        Returns: (reconstructed, encoded)
        """
        encoded = F.relu(self.encoder(x))
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activation strengths without reconstructing."""
        return F.relu(self.encoder(x))


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

        for i, prompt_text in enumerate(prompts):
            full = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
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

    def train_sae(self, layers: Optional[List[int]] = None,
                   num_features: int = 4096,
                   sparsity_weight: float = 0.01,
                   lr: float = 1e-3,
                   epochs: int = 50,
                   batch_size: int = 256) -> Dict:
        """Train Sparse Autoencoders on recorded activations.

        One SAE per layer, each learning an overcomplete basis of features.
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
            sae = SparseAutoencoder(hidden_size, num_features).to(dtype=torch.float32, device=self.device)
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
                    # Count active features (> 0.1 mean activation)
                    with torch.no_grad():
                        _, enc = sae(all_acts_device[:1000])
                        active = (enc.mean(dim=0) > 0.01).sum().item()
                    logger.info("  Layer %d epoch %d/%d: loss=%.4f (recon=%.4f sparse=%.4f) active=%d/%d",
                                layer_idx, epoch + 1, epochs, avg_loss, avg_recon, avg_sparse,
                                active, num_features)

            elapsed = time.time() - start
            sae.eval()
            self.saes[layer_idx] = sae

            # Compute final stats
            with torch.no_grad():
                _, final_enc = sae(all_acts_device)
                active_features = (final_enc.mean(dim=0) > 0.01).sum().item()
                top_features = final_enc.mean(dim=0).topk(10)

            # Store normalization params for inference
            sae._norm_mean = mean
            sae._norm_std = std

            results[layer_idx] = {
                "samples": n_samples,
                "features": num_features,
                "active_features": active_features,
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

        full = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n")
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
