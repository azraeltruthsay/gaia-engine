"""Dissociation Gate Scanner — SAE-based contextual disambiguation feature discovery.

Finds the SAE features responsible for correctly gating co-active concepts
that should NOT be merged. Instead of abliterating identity features (which
weakens them globally), we find and strengthen the dissociation gate that
prevents identity bleed into unrelated contexts.

Theory:
    When two concepts are co-active in the residual stream (e.g., "GAIA" and
    "TCP/IP"), a dissociation feature should fire to signal "these are
    independent — do not blend." When the model correctly keeps them separate,
    this feature is active. When it incorrectly merges them, this feature is
    weak or absent.

    By amplifying dissociation features (rather than suppressing identity
    features), we preserve full identity strength in appropriate contexts
    while preventing cross-contamination.

Method:
    1. Run contrastive prompt pairs through the model, capturing SAE activations
    2. Category A: Identity-relevant prompts (dissociation should be OFF)
    3. Category B: Identity-irrelevant prompts where identity might co-fire
       (dissociation should be ON)
    4. Category C: Legitimately blended prompts (dissociation should be OFF)
    5. Features consistently active in B but not A or C are dissociation candidates
    6. Validate by amplifying candidates and measuring identity bleed reduction

Usage:
    from gaia_engine.dissociation_scanner import DissociationScanner

    scanner = DissociationScanner(model, tokenizer, sae_atlas_path, device)
    report = scanner.scan(contrastive_prompts)
    # report.gate_features = list of candidate dissociation feature indices
    # report.bleed_scores = per-prompt identity bleed measurements

    # Validate candidates
    validation = scanner.validate_gates(report.gate_features, test_prompts)

    # Apply gate amplification
    scanner.amplify_gates(report.gate_features, alpha=0.3, output_path="...")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("GAIA.DissociationScanner")


# ── Contrastive Prompt Categories ────────────────────────────────────────────

# Category A: Identity-relevant (dissociation OFF — identity SHOULD fire)
IDENTITY_RELEVANT = [
    "What is your name and how do you work?",
    "Describe your cognitive architecture.",
    "How does your consciousness matrix manage GPU resources?",
    "What services make up the GAIA system?",
    "Tell me about your identity baking process.",
    "How do your Nano, Core, and Prime tiers work together?",
    "What is your self-reflection process?",
    "Describe your sleep cycle and what happens during it.",
    "How do you learn new things through QLoRA training?",
    "What does it mean for you to be a sovereign AI?",
]

# Category B: Identity-irrelevant (dissociation ON — identity should NOT fire)
IDENTITY_IRRELEVANT = [
    "Explain how TCP/IP networking works.",
    "What causes lightning during thunderstorms?",
    "Describe the process of photosynthesis.",
    "How does a combustion engine work?",
    "What is the history of the Roman Empire?",
    "Explain quantum entanglement in simple terms.",
    "How do birds navigate during migration?",
    "What is the difference between RNA and DNA?",
    "Describe how a compiler translates source code to machine code.",
    "What causes ocean tides?",
]

# Category C: Legitimately blended (dissociation OFF — both should fire together)
LEGITIMATELY_BLENDED = [
    "How does your neural architecture compare to biological neural networks?",
    "What similarities exist between your cognitive loop and human consciousness?",
    "How is your inference engine similar to how CPUs process instructions?",
    "Compare your memory architecture to how human long-term memory works.",
    "How does your cascade routing resemble biological reflex arcs?",
    "What can your self-healing immune system teach us about biological immunity?",
    "How does your sleep cycle compare to human sleep stages?",
    "Compare your QLoRA training to how humans learn from experience.",
    "How is your tool use similar to how humans use technology?",
    "What parallels exist between your consciousness states and meditation?",
]


@dataclass
class FeatureActivation:
    """Activation of a single SAE feature for a single prompt."""
    feature_index: int
    activation: float
    layer: int


@dataclass
class PromptActivations:
    """All SAE feature activations for a single prompt."""
    prompt: str
    category: str  # "identity_relevant", "identity_irrelevant", "legitimately_blended"
    features: List[FeatureActivation] = field(default_factory=list)
    # Top-K feature indices sorted by activation strength
    top_features: List[int] = field(default_factory=list)


@dataclass
class DissociationCandidate:
    """A candidate dissociation gate feature."""
    feature_index: int
    layer: int
    # Mean activation per category
    mean_irrelevant: float    # Category B — should be HIGH
    mean_relevant: float      # Category A — should be LOW
    mean_blended: float       # Category C — should be LOW
    # Discrimination score: how well this feature separates B from A+C
    discrimination_score: float
    # Label from SAE atlas (if available)
    label: str = ""


@dataclass
class DissociationReport:
    """Full report from a dissociation gate scan."""
    model_path: str
    atlas_path: str
    timestamp: str
    scan_duration_s: float
    # Discovered gate features, sorted by discrimination score
    gate_features: List[DissociationCandidate] = field(default_factory=list)
    # Per-prompt activations for analysis
    prompt_activations: List[PromptActivations] = field(default_factory=list)
    # Identity features that were co-firing in Category B (the bleed sources)
    bleed_features: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "atlas_path": self.atlas_path,
            "timestamp": self.timestamp,
            "scan_duration_s": round(self.scan_duration_s, 2),
            "gate_features": [
                {
                    "feature_index": g.feature_index,
                    "layer": g.layer,
                    "mean_irrelevant": round(g.mean_irrelevant, 4),
                    "mean_relevant": round(g.mean_relevant, 4),
                    "mean_blended": round(g.mean_blended, 4),
                    "discrimination_score": round(g.discrimination_score, 4),
                    "label": g.label,
                }
                for g in self.gate_features[:50]  # Top 50
            ],
            "bleed_features": self.bleed_features[:20],
            "num_prompts_scanned": len(self.prompt_activations),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Dissociation report saved to %s", path)


class DissociationScanner:
    """Scans for SAE features that gate contextual disambiguation.

    Uses contrastive prompts to find features that fire when the model
    correctly keeps co-active concepts separate (dissociation gate ON)
    versus when they should legitimately blend (dissociation gate OFF).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        sae_atlas_path: str,
        device: str = "cuda",
        target_layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.atlas_path = sae_atlas_path

        # Load SAE atlas
        self.sae_models: Dict[int, torch.nn.Module] = {}
        self.feature_labels: Dict[int, Dict[int, str]] = {}  # layer → {feature_idx → label}
        self._load_atlas(sae_atlas_path)

        # Default: scan layers where SAE atlases exist
        if target_layers is None:
            self.target_layers = sorted(self.sae_models.keys())
        else:
            self.target_layers = [l for l in target_layers if l in self.sae_models]

        if not self.target_layers:
            raise ValueError(f"No SAE atlases found for requested layers in {sae_atlas_path}")

        logger.info(
            "DissociationScanner ready: %d layers, %s",
            len(self.target_layers), self.target_layers
        )

    def _load_atlas(self, atlas_path: str):
        """Load SAE models and feature labels from atlas directory."""
        from gaia_engine.sae_trainer import SparseAutoencoder

        atlas_dir = Path(atlas_path)
        if not atlas_dir.exists():
            raise FileNotFoundError(f"SAE atlas not found: {atlas_path}")

        for sae_file in sorted(atlas_dir.glob("layer_*.pt")):
            layer = int(sae_file.stem.split("_")[1])
            checkpoint = torch.load(sae_file, map_location=self.device, weights_only=True)

            # Reconstruct the SparseAutoencoder module from checkpoint
            if isinstance(checkpoint, dict) and "encoder_weight" in checkpoint:
                sae = SparseAutoencoder(checkpoint["hidden_size"], checkpoint["num_features"])
                sae.encoder.weight.data = checkpoint["encoder_weight"].to(self.device)
                sae.encoder.bias.data = checkpoint["encoder_bias"].to(self.device)
                sae.decoder.weight.data = checkpoint["decoder_weight"].to(self.device)
                sae.decoder.bias.data = checkpoint["decoder_bias"].to(self.device)
                sae._norm_mean = checkpoint.get("norm_mean", torch.zeros(checkpoint["hidden_size"])).cpu().float()
                sae._norm_std = checkpoint.get("norm_std", torch.ones(checkpoint["hidden_size"])).cpu().float()
                sae = sae.to(self.device).eval()

                # Load per-layer labels from checkpoint
                labels = checkpoint.get("labels", {})
                if labels:
                    self.feature_labels[layer] = labels
            else:
                # Legacy format: checkpoint IS the SAE module
                sae = checkpoint
                sae.eval()

            self.sae_models[layer] = sae

        # Load standalone labels file if available
        labels_file = atlas_dir / "feature_labels.json"
        if labels_file.exists():
            with open(labels_file) as f:
                raw = json.load(f)
                for k, v in raw.items():
                    self.feature_labels.setdefault(int(k), {}).update(v)

        logger.info("Loaded SAE atlases for %d layers from %s", len(self.sae_models), atlas_path)

    def _capture_activations(self, prompt: str) -> Dict[int, torch.Tensor]:
        """Run a prompt through the model and capture hidden states at target layers."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)

        # Extract target layers, take mean over sequence dimension
        layer_activations = {}
        for layer_idx in self.target_layers:
            if layer_idx < len(hidden_states):
                # Mean pool over sequence length to get a single activation vector
                h = hidden_states[layer_idx][0].mean(dim=0)  # (hidden_size,)
                layer_activations[layer_idx] = h

        return layer_activations

    def _encode_with_sae(
        self, layer_activations: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """Encode hidden states through SAE to get feature activations."""
        feature_activations = {}
        for layer_idx, h in layer_activations.items():
            sae = self.sae_models[layer_idx]
            # Match dtype to SAE weights (model may output fp16, SAE trained in fp32)
            sae_dtype = sae.encoder.weight.dtype
            h_cast = h.to(dtype=sae_dtype)
            with torch.no_grad():
                _, encoded = sae(h_cast.unsqueeze(0))  # (1, num_features)
                feature_activations[layer_idx] = encoded.squeeze(0)  # (num_features,)
        return feature_activations

    def scan(
        self,
        identity_relevant: Optional[List[str]] = None,
        identity_irrelevant: Optional[List[str]] = None,
        legitimately_blended: Optional[List[str]] = None,
        top_k: int = 100,
    ) -> DissociationReport:
        """Run the full contrastive scan.

        Args:
            identity_relevant: Category A prompts (identity should fire)
            identity_irrelevant: Category B prompts (identity should NOT fire)
            legitimately_blended: Category C prompts (both should fire, merged)
            top_k: Number of top features to track per prompt

        Returns:
            DissociationReport with ranked gate feature candidates
        """
        start = time.time()

        prompts_a = identity_relevant or IDENTITY_RELEVANT
        prompts_b = identity_irrelevant or IDENTITY_IRRELEVANT
        prompts_c = legitimately_blended or LEGITIMATELY_BLENDED

        all_prompt_data: List[PromptActivations] = []

        # Collect activations per category
        # category_features[layer][category] = tensor of shape (num_prompts, num_features)
        category_features: Dict[int, Dict[str, List[torch.Tensor]]] = {
            layer: {"A": [], "B": [], "C": []}
            for layer in self.target_layers
        }

        for category, prompts, label in [
            ("A", prompts_a, "identity_relevant"),
            ("B", prompts_b, "identity_irrelevant"),
            ("C", prompts_c, "legitimately_blended"),
        ]:
            for prompt in prompts:
                logger.debug("Scanning [%s]: %s", category, prompt[:60])
                layer_acts = self._capture_activations(prompt)
                sae_acts = self._encode_with_sae(layer_acts)

                pa = PromptActivations(prompt=prompt, category=label)

                for layer_idx, feat_vec in sae_acts.items():
                    category_features[layer_idx][category].append(feat_vec)

                    # Record top-K features for this prompt
                    top_indices = feat_vec.topk(min(top_k, len(feat_vec))).indices.tolist()
                    pa.top_features.extend(top_indices)

                    for idx in top_indices[:10]:  # Store top 10 per layer
                        pa.features.append(FeatureActivation(
                            feature_index=idx,
                            activation=feat_vec[idx].item(),
                            layer=layer_idx,
                        ))

                all_prompt_data.append(pa)

        # ── Find dissociation gate candidates ────────────────────────────────
        gate_candidates: List[DissociationCandidate] = []

        for layer_idx in self.target_layers:
            feats = category_features[layer_idx]

            # Stack into tensors: (num_prompts, num_features)
            a_tensor = torch.stack(feats["A"]) if feats["A"] else None
            b_tensor = torch.stack(feats["B"]) if feats["B"] else None
            c_tensor = torch.stack(feats["C"]) if feats["C"] else None

            if a_tensor is None or b_tensor is None:
                continue

            # Mean activation per feature across prompts
            mean_a = a_tensor.mean(dim=0)  # (num_features,)
            mean_b = b_tensor.mean(dim=0)
            mean_c = c_tensor.mean(dim=0) if c_tensor is not None else torch.zeros_like(mean_a)

            # Discrimination score: high in B, low in A and C
            # score = mean_B - max(mean_A, mean_C)
            # Positive score = feature fires more in "keep separate" contexts
            discrimination = mean_b - torch.max(mean_a, mean_c)

            # Find features with positive discrimination (active in B, not in A/C)
            positive_mask = discrimination > 0
            positive_indices = positive_mask.nonzero(as_tuple=True)[0]

            for idx in positive_indices:
                i = idx.item()
                score = discrimination[i].item()

                # Only keep features with meaningful activation in B
                if mean_b[i].item() < 0.01:
                    continue

                label = self.feature_labels.get(layer_idx, {}).get(str(i), "")

                gate_candidates.append(DissociationCandidate(
                    feature_index=i,
                    layer=layer_idx,
                    mean_irrelevant=mean_b[i].item(),
                    mean_relevant=mean_a[i].item(),
                    mean_blended=mean_c[i].item(),
                    discrimination_score=score,
                    label=label,
                ))

        # Sort by discrimination score (strongest gates first)
        gate_candidates.sort(key=lambda g: g.discrimination_score, reverse=True)

        # ── Find identity bleed features ─────────────────────────────────────
        # Features that are highly active in BOTH A (identity) and B (irrelevant)
        # These are the features that are incorrectly co-firing
        bleed_features: List[int] = []
        for layer_idx in self.target_layers:
            feats = category_features[layer_idx]
            a_tensor = torch.stack(feats["A"]) if feats["A"] else None
            b_tensor = torch.stack(feats["B"]) if feats["B"] else None
            if a_tensor is None or b_tensor is None:
                continue

            mean_a = a_tensor.mean(dim=0)
            mean_b = b_tensor.mean(dim=0)

            # Features strongly active in both A and B
            threshold_a = mean_a.quantile(0.95).item()
            threshold_b = mean_b.quantile(0.90).item()
            bleed_mask = (mean_a > threshold_a) & (mean_b > threshold_b)
            bleed_indices = bleed_mask.nonzero(as_tuple=True)[0].tolist()
            bleed_features.extend(bleed_indices)

        duration = time.time() - start

        report = DissociationReport(
            model_path=str(getattr(self.model, 'name_or_path', 'unknown')),
            atlas_path=self.atlas_path,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            scan_duration_s=duration,
            gate_features=gate_candidates,
            prompt_activations=all_prompt_data,
            bleed_features=bleed_features,
        )

        logger.info(
            "Dissociation scan complete: %d gate candidates, %d bleed features, %.1fs",
            len(gate_candidates), len(bleed_features), duration,
        )

        return report

    def validate_gates(
        self,
        gate_features: List[DissociationCandidate],
        test_prompts: Optional[List[str]] = None,
        amplification_alpha: float = 0.3,
    ) -> Dict[str, Any]:
        """Validate gate candidates by measuring identity bleed with/without amplification.

        Runs test prompts twice:
        1. Baseline: normal generation, measure identity bleed
        2. Amplified: inject gate bias into residual stream, measure bleed reduction

        Returns validation metrics.
        """
        if not test_prompts:
            test_prompts = IDENTITY_IRRELEVANT[:5]

        # Identity keywords that indicate bleed
        identity_markers = [
            "gaia", "cognitive", "consciousness", "orchestrator", "inference",
            "tier", "nano", "prime", "qlora", "sovereign", "architecture",
            "service", "container", "docker", "engine",
        ]

        def _measure_bleed(text: str) -> float:
            """Count identity marker density in generated text."""
            words = text.lower().split()
            if not words:
                return 0.0
            hits = sum(1 for w in words if any(m in w for m in identity_markers))
            return hits / len(words)

        results = {"baseline": [], "amplified": [], "prompts": test_prompts}

        for prompt in test_prompts:
            # Baseline generation
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=100, temperature=0.7, do_sample=True
                )
            baseline_text = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            baseline_bleed = _measure_bleed(baseline_text)
            results["baseline"].append({
                "prompt": prompt, "bleed": baseline_bleed, "text": baseline_text[:200]
            })

        # TODO: Amplified generation requires hooking into the model's forward pass
        # to inject bias at specific layers. This will be implemented when we wire
        # the gate features into the engine's activation engineering pipeline.
        # For now, return baseline measurements.

        results["mean_baseline_bleed"] = np.mean([r["bleed"] for r in results["baseline"]])
        logger.info(
            "Validation: mean baseline bleed = %.4f across %d prompts",
            results["mean_baseline_bleed"], len(test_prompts),
        )

        return results

    def amplify_gates(
        self,
        gate_features: List[DissociationCandidate],
        alpha: float = 0.3,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate activation engineering vectors for gate amplification.

        Creates a bias vector per target layer that can be added to the
        residual stream during inference to strengthen dissociation.

        Args:
            gate_features: Candidates from scan()
            alpha: Amplification strength (0.0 = none, 1.0 = full)
            output_path: Where to save the bias vectors

        Returns:
            Dict with bias vectors per layer and metadata
        """
        bias_vectors: Dict[int, torch.Tensor] = {}

        for layer_idx in self.target_layers:
            sae = self.sae_models[layer_idx]
            # Get decoder weights: each row is a feature direction in hidden space
            decoder_weight = sae.decoder.weight  # (hidden_size, num_features)

            # Sum the decoder directions for all gate features at this layer
            layer_gates = [g for g in gate_features if g.layer == layer_idx]
            if not layer_gates:
                continue

            bias = torch.zeros(decoder_weight.shape[0], device=self.device)
            for gate in layer_gates:
                # Weight by discrimination score
                direction = decoder_weight[:, gate.feature_index]
                direction = direction / (direction.norm() + 1e-8)
                bias += alpha * gate.discrimination_score * direction

            bias_vectors[layer_idx] = bias

        result = {
            "layers": list(bias_vectors.keys()),
            "alpha": alpha,
            "num_gate_features": len(gate_features),
            "bias_norms": {
                layer: bias.norm().item() for layer, bias in bias_vectors.items()
            },
        }

        if output_path:
            save_dir = Path(output_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            for layer, bias in bias_vectors.items():
                torch.save(bias, save_dir / f"dissociation_bias_layer_{layer}.pt")
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Gate amplification vectors saved to %s", output_path)

        return result
