"""Adapter Surgeon — SAE-guided LoRA adapter editing.

Uses Sparse Autoencoder feature analysis to identify and correct
problematic feature directions in LoRA adapters WITHOUT retraining.

The key insight: a LoRA adapter contributes delta = B @ A @ x to each
layer's output. If SAE analysis shows this delta is suppressing a
protected feature (e.g., identity neuron #1838), we can project out
that component from the adapter's B matrix while preserving everything
else the adapter learned.

Math:
    f = SAE encoder weight for protected feature (normalized)
    B_new = B - alpha * (f @ f.T) @ B

    This nulls out B's projection onto the protected feature direction.
    alpha=1.0 means full protection, alpha=0.5 means partial.

This is novel: SAE-guided adapter surgery. No retraining needed.

Usage:
    from gaia_engine.adapter_surgeon import AdapterSurgeon

    surgeon = AdapterSurgeon(model, tokenizer, sae_atlas_path, device)
    diagnosis = surgeon.diagnose_adapter(adapter_path, protected_features, test_prompts)
    surgeon.apply_correction(adapter_path, diagnosis, output_path)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("GAIA.AdapterSurgeon")


@dataclass
class FeatureImpact:
    """How an adapter affects a specific SAE feature."""
    feature_index: int
    feature_label: str
    layer: int
    base_activation: float  # Without adapter
    adapter_activation: float  # With adapter
    delta: float  # Change caused by adapter
    protected: bool  # Is this a feature we want to preserve


@dataclass
class AdapterDiagnosis:
    """Full diagnosis of an adapter's effect on protected features."""
    adapter_path: str
    layer_impacts: Dict[int, List[FeatureImpact]]  # layer → impacts
    protected_regressions: List[FeatureImpact]  # Features that regressed
    beneficial_changes: List[FeatureImpact]  # Features that improved
    overall_score: float  # -1.0 (harmful) to 1.0 (beneficial)
    recommendation: str  # "safe", "correct", "reject"

    def to_dict(self) -> dict:
        return {
            "adapter_path": self.adapter_path,
            "protected_regressions": [
                {"feature": f.feature_index, "label": f.feature_label,
                 "layer": f.layer, "delta": round(f.delta, 4)}
                for f in self.protected_regressions
            ],
            "beneficial_changes": [
                {"feature": f.feature_index, "label": f.feature_label,
                 "layer": f.layer, "delta": round(f.delta, 4)}
                for f in self.beneficial_changes
            ],
            "overall_score": round(self.overall_score, 4),
            "recommendation": self.recommendation,
        }


class AdapterSurgeon:
    """Diagnose and surgically correct LoRA adapters using SAE analysis."""

    def __init__(
        self,
        model,
        tokenizer,
        sae_atlas_path: str,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sae_atlas_path = Path(sae_atlas_path)

        # Load SAE atlas
        self._saes = {}
        self._load_atlas()

    def _load_atlas(self) -> None:
        """Load pre-trained SAE models from atlas directory."""
        import torch

        meta_path = self.sae_atlas_path / "meta.json"
        if not meta_path.exists():
            logger.warning("No SAE atlas at %s", self.sae_atlas_path)
            return

        meta = json.loads(meta_path.read_text())
        for layer_idx in meta.get("layers", []):
            pt_path = self.sae_atlas_path / f"layer_{layer_idx}.pt"
            if pt_path.exists():
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                self._saes[layer_idx] = data
                logger.debug("Loaded SAE for layer %d (%d features)",
                            layer_idx, data["num_features"])

        logger.info("SAE atlas loaded: %d layers from %s",
                    len(self._saes), self.sae_atlas_path)

    def _get_activations(self, prompt: str, layers: List[int]) -> Dict[int, "torch.Tensor"]:
        """Get hidden state activations for a prompt at specified layers."""
        import torch

        from gaia_engine.core import ChatFormatter
        fmt = ChatFormatter(self.tokenizer)
        full = (fmt.format_system("You are GAIA, a sovereign AI created by Azrael.")
                + "\n" + fmt.format_message("user", prompt)
                + "\n" + fmt.assistant_prefix(enable_thinking=True))
        ids = self.tokenizer.encode(full, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)

        result = {}
        for layer_idx in layers:
            if layer_idx < len(out.hidden_states):
                # Last token's hidden state
                result[layer_idx] = out.hidden_states[layer_idx][0, -1].detach().cpu()

        return result

    def _sae_decompose(self, activation: "torch.Tensor", layer_idx: int) -> Dict[int, float]:
        """Decompose an activation vector into SAE feature strengths."""
        import torch
        import torch.nn.functional as F

        if layer_idx not in self._saes:
            return {}

        sae_data = self._saes[layer_idx]
        encoder_weight = sae_data["encoder_weight"]
        encoder_bias = sae_data["encoder_bias"]
        norm_mean = sae_data["norm_mean"]
        norm_std = sae_data["norm_std"]

        # Normalize
        act_norm = (activation - norm_mean) / norm_std

        # Encode
        features = F.relu(F.linear(act_norm, encoder_weight, encoder_bias))

        # Return as dict of feature_idx → strength
        result = {}
        for idx in range(features.shape[0]):
            val = features[idx].item()
            if val > 0.01:  # Only active features
                result[idx] = val

        return result

    def diagnose_adapter(
        self,
        adapter_path: str,
        protected_features: Dict[int, List[int]],  # layer → [feature_indices]
        test_prompts: List[str],
        feature_labels: Optional[Dict[str, str]] = None,
    ) -> AdapterDiagnosis:
        """Diagnose how an adapter affects protected features.

        Args:
            adapter_path: Path to the LoRA adapter
            protected_features: {layer_idx: [feature_indices]} to monitor
            test_prompts: Prompts to test with
            feature_labels: Optional {layer_feature: label} for readability

        Returns:
            AdapterDiagnosis with detailed impact analysis
        """
        import torch
        from peft import PeftModel

        if feature_labels is None:
            feature_labels = {}

        layers = list(protected_features.keys())
        logger.info("Diagnosing adapter: %s (%d layers, %d prompts)",
                    adapter_path, len(layers), len(test_prompts))

        # Phase 1: Capture base model activations (no adapter)
        logger.info("Phase 1: Base model activations...")
        base_features = {}  # prompt → layer → {feature: strength}
        for prompt in test_prompts:
            acts = self._get_activations(prompt, layers)
            base_features[prompt] = {}
            for layer_idx, act in acts.items():
                base_features[prompt][layer_idx] = self._sae_decompose(act, layer_idx)

        # Phase 2: Load adapter and capture activations
        logger.info("Phase 2: Adapter activations...")
        adapter_model = PeftModel.from_pretrained(self.model, adapter_path)
        adapter_model.eval()

        # Swap model temporarily
        orig_model = self.model
        self.model = adapter_model

        adapter_features = {}
        for prompt in test_prompts:
            acts = self._get_activations(prompt, layers)
            adapter_features[prompt] = {}
            for layer_idx, act in acts.items():
                adapter_features[prompt][layer_idx] = self._sae_decompose(act, layer_idx)

        # Restore original model
        self.model = orig_model
        del adapter_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 3: Compare
        logger.info("Phase 3: Analyzing feature impacts...")
        all_impacts = {}
        protected_regressions = []
        beneficial_changes = []

        for layer_idx, feature_indices in protected_features.items():
            layer_impacts = []

            for feat_idx in feature_indices:
                # Average activation across all prompts
                base_avg = np.mean([
                    base_features[p].get(layer_idx, {}).get(feat_idx, 0.0)
                    for p in test_prompts
                ])
                adapter_avg = np.mean([
                    adapter_features[p].get(layer_idx, {}).get(feat_idx, 0.0)
                    for p in test_prompts
                ])
                delta = adapter_avg - base_avg
                label_key = f"{layer_idx}_{feat_idx}"
                label = feature_labels.get(label_key, f"feature_{feat_idx}")

                impact = FeatureImpact(
                    feature_index=feat_idx,
                    feature_label=label,
                    layer=layer_idx,
                    base_activation=float(base_avg),
                    adapter_activation=float(adapter_avg),
                    delta=float(delta),
                    protected=True,
                )
                layer_impacts.append(impact)

                if delta < -0.1:  # Feature weakened
                    protected_regressions.append(impact)
                    logger.warning(
                        "REGRESSION: layer %d feature %d (%s): %.3f → %.3f (Δ%.3f)",
                        layer_idx, feat_idx, label, base_avg, adapter_avg, delta,
                    )
                elif delta > 0.1:  # Feature strengthened
                    beneficial_changes.append(impact)

            all_impacts[layer_idx] = layer_impacts

        # Score: negative if regressions dominate
        reg_magnitude = sum(abs(r.delta) for r in protected_regressions)
        ben_magnitude = sum(abs(b.delta) for b in beneficial_changes)
        total = reg_magnitude + ben_magnitude
        overall_score = (ben_magnitude - reg_magnitude) / max(total, 0.01)

        if not protected_regressions:
            recommendation = "safe"
        elif reg_magnitude < 0.5:
            recommendation = "correct"  # Minor regressions, correctable
        else:
            recommendation = "reject"  # Major regressions

        diagnosis = AdapterDiagnosis(
            adapter_path=adapter_path,
            layer_impacts=all_impacts,
            protected_regressions=protected_regressions,
            beneficial_changes=beneficial_changes,
            overall_score=overall_score,
            recommendation=recommendation,
        )

        logger.info(
            "Diagnosis: score=%.3f, regressions=%d, beneficial=%d → %s",
            overall_score, len(protected_regressions),
            len(beneficial_changes), recommendation,
        )
        return diagnosis

    def apply_correction(
        self,
        adapter_path: str,
        diagnosis: AdapterDiagnosis,
        output_path: str,
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """Surgically correct an adapter to preserve protected features.

        For each regressed feature, projects out the adapter's component
        along that feature direction from the B matrix.

        Args:
            adapter_path: Path to the LoRA adapter to correct
            diagnosis: Diagnosis from diagnose_adapter()
            output_path: Where to save the corrected adapter
            alpha: Protection strength (1.0 = full, 0.5 = partial)

        Returns:
            Dict with correction stats
        """
        import torch
        import shutil

        if not diagnosis.protected_regressions:
            logger.info("No regressions to correct — adapter is safe")
            return {"corrected": False, "reason": "no regressions"}

        logger.info("Correcting %d feature regressions (alpha=%.2f)...",
                    len(diagnosis.protected_regressions), alpha)

        # Copy adapter to output path
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        for f in Path(adapter_path).iterdir():
            if f.is_file() and f.name != "adapter_model.safetensors":
                shutil.copy2(f, out / f.name)

        # Load adapter weights
        from safetensors.torch import load_file, save_file

        adapter_weights = load_file(Path(adapter_path) / "adapter_model.safetensors")

        corrections_applied = 0

        for regression in diagnosis.protected_regressions:
            layer_idx = regression.layer
            feat_idx = regression.feature_index

            if layer_idx not in self._saes:
                continue

            # Get the SAE encoder direction for this feature
            sae_data = self._saes[layer_idx]
            encoder_weight = sae_data["encoder_weight"]  # [num_features, hidden_size]
            feature_direction = encoder_weight[feat_idx]  # [hidden_size]
            feature_direction = feature_direction / feature_direction.norm()  # Normalize

            # Find the adapter's B matrix for this layer
            # LoRA weight names: base_model.model.model.layers.{N}.self_attn.{q,v}_proj.lora_B.weight
            for key in adapter_weights:
                # Match layers approximately (LoRA layer numbering may differ)
                if f"layers.{layer_idx}." in key and "lora_B" in key:
                    B = adapter_weights[key]  # [out_features, rank]

                    # Project out the feature direction from B
                    # B_new = B - alpha * (f @ f.T) @ B
                    # Where f is [out_features] and B is [out_features, rank]
                    if B.shape[0] == feature_direction.shape[0]:
                        # Match dtypes for matmul
                        fd = feature_direction.to(B.dtype)
                        projection = alpha * torch.outer(fd, fd) @ B
                        B_corrected = B - projection

                        adapter_weights[key] = B_corrected
                        corrections_applied += 1

                        change_magnitude = projection.norm().item()
                        logger.info(
                            "Corrected %s: nulled feature %d direction (magnitude %.4f)",
                            key, feat_idx, change_magnitude,
                        )

        # Save corrected weights
        save_file(adapter_weights, out / "adapter_model.safetensors")

        result = {
            "corrected": True,
            "corrections_applied": corrections_applied,
            "regressions_targeted": len(diagnosis.protected_regressions),
            "alpha": alpha,
            "output_path": str(out),
        }

        logger.info("Adapter surgery complete: %d corrections applied to %s",
                    corrections_applied, output_path)
        return result
