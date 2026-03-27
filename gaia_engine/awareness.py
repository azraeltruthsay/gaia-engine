"""
Dynamic Awareness System — situated cognition for GAIA.

Manages what's cognitively relevant RIGHT NOW based on temporal context,
geographic proximity, current task, and environmental signals.

Awareness packages are small text files (~100-200 tokens) stored in
/knowledge/awareness/. This system selects which ones to inject into
the KV prefix cache as a situational_awareness segment.

Curiosity signals are generated when awareness data is stale or has
low confidence, driving autonomous research during idle time.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.Awareness")

from gaia_engine.config import AWARENESS_DIR as _AWARENESS_DIR_STR
AWARENESS_DIR = Path(_AWARENESS_DIR_STR)

# Relevance categories with base weights
# Higher weight = more likely to be included in the awareness segment
CATEGORY_WEIGHTS = {
    "temporal": 0.9,      # Almost always relevant (time, season, holidays)
    "local": 0.8,         # High relevance (weather, local events)
    "operational": 0.7,   # Usually relevant (system state, recent work)
    "global": 0.3,        # Lower base relevance (world news, geopolitics)
}

# Staleness thresholds (seconds) — after this, a curiosity signal fires
STALENESS_THRESHOLDS = {
    "weather": 3600,           # 1 hour
    "local_news": 86400,       # 1 day
    "tech_news": 43200,        # 12 hours
    "holidays": 604800,        # 1 week
    "today": 43200,             # 12 hours (refresh twice daily)
    "season": 2592000,         # 30 days
    "system_state": 300,       # 5 minutes
    "recent_work": 3600,       # 1 hour
    "location": 999999999,     # Essentially never stale
    "geopolitics": 86400,      # 1 day
}


class AwarenessPackage:
    """A single awareness file with metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.category = path.parent.name
        self.content = ""
        self.content_hash = ""
        self.last_read = 0.0
        self.last_modified = 0.0
        self.token_estimate = 0
        self.confidence = 1.0  # 0.0 = unverified, 1.0 = fully verified
        self.reload()

    def reload(self):
        """Read content from disk."""
        if self.path.exists():
            self.content = self.path.read_text().strip()
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.last_modified = self.path.stat().st_mtime
            self.last_read = time.time()
            self.token_estimate = len(self.content) // 3 + 4  # rough estimate
        else:
            self.content = ""
            self.token_estimate = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_modified if self.last_modified > 0 else float("inf")

    @property
    def is_stale(self) -> bool:
        threshold = STALENESS_THRESHOLDS.get(self.name, 86400)
        return self.age_seconds > threshold

    @property
    def base_weight(self) -> float:
        return CATEGORY_WEIGHTS.get(self.category, 0.5)


class AwarenessManager:
    """Manages dynamic awareness — selects and composes relevant context.

    Scans the awareness directory, tracks staleness, selects relevant
    packages based on weighted scoring, and composes them into a single
    text block for KV prefix injection.
    """

    def __init__(self, awareness_dir: Optional[Path] = None, max_tokens: int = 800):
        self.awareness_dir = awareness_dir or AWARENESS_DIR
        self.max_tokens = max_tokens
        self.packages: Dict[str, AwarenessPackage] = {}
        self._curiosity_signals: List[dict] = []
        self._last_scan = 0.0
        self._scan_interval = 60.0  # re-scan directory every 60s

        self._scan_packages()

    def _scan_packages(self):
        """Scan awareness directory for .md files."""
        if not self.awareness_dir.exists():
            logger.warning("Awareness directory not found: %s", self.awareness_dir)
            return

        for md_file in self.awareness_dir.rglob("*.md"):
            key = f"{md_file.parent.name}/{md_file.stem}"
            if key not in self.packages:
                self.packages[key] = AwarenessPackage(md_file)
            else:
                # Reload if file changed
                pkg = self.packages[key]
                if md_file.stat().st_mtime > pkg.last_modified:
                    pkg.reload()

        self._last_scan = time.time()
        logger.debug("Awareness scan: %d packages", len(self.packages))

    def select_relevant(self, context: str = "", max_tokens: Optional[int] = None,
                         boost_categories: Optional[Dict[str, float]] = None) -> List[AwarenessPackage]:
        """Select the most relevant awareness packages for the current context.

        Args:
            context: Current task/conversation context (for relevance scoring)
            max_tokens: Override max token budget
            boost_categories: Extra weight for specific categories {"operational": 0.5}

        Returns:
            List of selected packages, sorted by relevance
        """
        # Re-scan if stale
        if time.time() - self._last_scan > self._scan_interval:
            self._scan_packages()

        budget = max_tokens or self.max_tokens
        boosts = boost_categories or {}

        # Score each package
        scored = []
        for key, pkg in self.packages.items():
            if not pkg.content:
                continue

            score = pkg.base_weight

            # Boost if category matches requested boosts
            if pkg.category in boosts:
                score += boosts[pkg.category]

            # Penalize stale content
            if pkg.is_stale:
                score *= 0.5

            # Penalize low confidence
            score *= pkg.confidence

            # Boost if context mentions related terms
            if context:
                context_lower = context.lower()
                name_lower = pkg.name.lower()
                if name_lower in context_lower or any(
                    term in context_lower for term in name_lower.split("_")
                ):
                    score += 0.3

            scored.append((score, pkg))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Select within token budget
        selected = []
        used_tokens = 0
        for score, pkg in scored:
            if used_tokens + pkg.token_estimate > budget:
                continue
            selected.append(pkg)
            used_tokens += pkg.token_estimate

        return selected

    def compose_awareness_text(self, context: str = "",
                                 boost_categories: Optional[Dict[str, float]] = None) -> str:
        """Compose selected awareness packages into a single text block.

        This text is designed to be injected as the 'situational_awareness'
        segment in the KV prefix cache.
        """
        selected = self.select_relevant(context, boost_categories=boost_categories)

        if not selected:
            return ""

        parts = ["[Situational Awareness]"]
        for pkg in selected:
            parts.append(f"[{pkg.category}/{pkg.name}] {pkg.content}")

        text = "\n".join(parts)
        logger.info("Awareness composed: %d packages, ~%d tokens",
                     len(selected), len(text) // 3)
        return text

    def get_curiosity_signals(self) -> List[dict]:
        """Generate curiosity signals for stale or missing awareness data.

        Returns a list of THOUGHT_SEED-style signals that the idle heartbeat
        or sleep cycle can act on.
        """
        signals = []

        for key, pkg in self.packages.items():
            if pkg.is_stale:
                threshold = STALENESS_THRESHOLDS.get(pkg.name, 86400)
                signals.append({
                    "type": "THOUGHT_SEED",
                    "category": "knowledge_gap",
                    "topic": f"Stale awareness: {key}",
                    "detail": f"{pkg.name} last updated {pkg.age_seconds / 3600:.1f}h ago "
                              f"(threshold: {threshold / 3600:.1f}h)",
                    "action": f"Research and update /knowledge/awareness/{key}.md",
                    "priority": pkg.base_weight,
                })

        # Check for expected but missing packages
        expected = ["temporal/weather", "temporal/holidays", "temporal/season",
                     "local/location", "local/local_news",
                     "operational/system_state", "operational/recent_work"]
        for key in expected:
            if key not in self.packages:
                signals.append({
                    "type": "THOUGHT_SEED",
                    "category": "knowledge_gap",
                    "topic": f"Missing awareness: {key}",
                    "detail": f"Expected awareness file not found",
                    "action": f"Create /knowledge/awareness/{key}.md",
                    "priority": 0.8,
                })

        self._curiosity_signals = signals
        return signals

    def status(self) -> dict:
        """Get awareness system status."""
        stale = [k for k, p in self.packages.items() if p.is_stale]
        return {
            "total_packages": len(self.packages),
            "stale_packages": stale,
            "curiosity_signals": len(self.get_curiosity_signals()),
            "packages": {
                key: {
                    "category": pkg.category,
                    "tokens": pkg.token_estimate,
                    "age_hours": round(pkg.age_seconds / 3600, 1),
                    "stale": pkg.is_stale,
                    "confidence": pkg.confidence,
                    "hash": pkg.content_hash[:8],
                }
                for key, pkg in self.packages.items()
            },
        }
