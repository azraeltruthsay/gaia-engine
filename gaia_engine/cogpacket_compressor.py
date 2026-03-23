"""
CogPacket Compressor — dynamic prompt compression using KV cache and SAE atlas.

Instead of stuffing 6K+ tokens of identity/tools/world state into every
request, this compressor checks what's already known:

1. KV Cache segments: if hash matches, skip entirely (already cached)
2. SAE atlas: if features are strong for this topic, skip (weight-baked)
3. Awareness system: operational facts already injected via awareness
4. Only include genuinely novel content

The result: a 6K system prompt compressed to ~200-500 tokens of
genuinely new information. Everything else is already in GAIA's head.

Usage:
    from gaia_engine.cogpacket_compressor import compress_system_prompt

    compressed = compress_system_prompt(
        full_prompt="You are GAIA... [6K tokens of context]",
        kv_cache=engine.prefix_cache,
        awareness=engine.awareness,
    )
    # compressed is ~200-500 tokens of genuinely novel content
"""

import hashlib
import logging
from typing import Dict, Optional, List

logger = logging.getLogger("GAIA.CogPacketCompressor")


# Prompt sections that are commonly repeated and can be cached
CACHEABLE_SECTIONS = {
    "identity": {
        "markers": ["GAIA PERSONA ANCHOR", "You are GAIA", "sovereign AI"],
        "description": "Core identity and persona instructions",
    },
    "epistemic_rules": {
        "markers": ["EPISTEMIC HONESTY", "ANTI-CONFABULATION", "Source Integrity"],
        "description": "Epistemic honesty and anti-confabulation rules",
    },
    "tools": {
        "markers": ["MCP:", "Available tools:", "EXECUTE:"],
        "description": "MCP tool list and capabilities",
    },
    "world_state": {
        "markers": ["World State", "Clock:", "Uptime:", "Immune System:"],
        "description": "Current system state snapshot",
    },
    "cheatsheets": {
        "markers": ["Reference Cheatsheets", "cheat_sheet.json"],
        "description": "Reference cheatsheet pointers",
    },
    "spinal_routing": {
        "markers": ["SPINAL ROUTING", "SKETCHPAD:", "USER_CHAT:"],
        "description": "Output routing directives",
    },
    "thought_seeds": {
        "markers": ["THOUGHT_SEED", "THOUGHT SEED DIRECTIVE"],
        "description": "Thought seed emission instructions",
    },
    "vital_organs": {
        "markers": ["VITAL ORGAN", "PROMOTION PROTOCOL"],
        "description": "Vital organ promotion protocol",
    },
}


def identify_sections(prompt: str) -> Dict[str, Dict]:
    """Identify which cacheable sections are present in a prompt.

    Returns dict of section_name → {start, end, content, tokens_est}
    """
    sections = {}
    prompt_lower = prompt.lower()

    for name, config in CACHEABLE_SECTIONS.items():
        for marker in config["markers"]:
            pos = prompt_lower.find(marker.lower())
            if pos >= 0:
                # Estimate section boundaries (from marker to next double-newline or 500 chars)
                end = prompt.find("\n\n", pos + len(marker))
                if end < 0 or end - pos > 2000:
                    end = min(pos + 2000, len(prompt))

                content = prompt[pos:end].strip()
                sections[name] = {
                    "start": pos,
                    "end": end,
                    "content": content,
                    "tokens_est": len(content) // 3,
                    "hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                    "description": config["description"],
                }
                break

    return sections


def compress_system_prompt(
    full_prompt: str,
    kv_cache=None,
    awareness=None,
    sae_confident_topics: Optional[List[str]] = None,
) -> str:
    """Compress a system prompt by removing sections already in KV cache or weights.

    Args:
        full_prompt: The full system prompt (potentially 6K+ tokens)
        kv_cache: PrefixCache instance (checks segment hashes)
        awareness: AwarenessManager instance (checks what's injected)
        sae_confident_topics: Topics SAE confirms are in weights (e.g., ["identity", "values"])

    Returns:
        Compressed prompt with only novel content
    """
    sections = identify_sections(full_prompt)

    if not sections:
        logger.debug("No cacheable sections found — returning full prompt")
        return full_prompt

    original_tokens = len(full_prompt) // 3
    skipped = []
    kept = []

    # Check each section
    for name, section in sections.items():
        skip_reason = None

        # Check 1: KV cache — is this section already cached?
        if kv_cache and hasattr(kv_cache, '_hashes'):
            cached_hash = kv_cache._hashes.get(name)
            if cached_hash and cached_hash == section["hash"]:
                skip_reason = "KV cache hit"

        # Check 2: SAE — are these features strong in weights?
        if not skip_reason and sae_confident_topics and name in sae_confident_topics:
            skip_reason = "SAE confirms weight-baked"

        # Check 3: Awareness — is this content in the awareness segment?
        if not skip_reason and awareness and name in ("world_state",):
            # World state is injected via awareness — don't duplicate
            if awareness.compose_awareness_text():
                skip_reason = "awareness system covers this"

        if skip_reason:
            skipped.append((name, section["tokens_est"], skip_reason))
        else:
            kept.append(name)

    # Build compressed prompt — remove skipped sections
    compressed = full_prompt
    for name, tokens, reason in sorted(skipped, key=lambda x: -sections[x[0]]["start"]):
        # Replace section with a brief marker
        section = sections[name]
        marker = f"[{name}: loaded from cache]"
        compressed = compressed[:section["start"]] + marker + compressed[section["end"]:]

    compressed_tokens = len(compressed) // 3
    savings = original_tokens - compressed_tokens
    savings_pct = (savings / original_tokens * 100) if original_tokens > 0 else 0

    logger.info(
        "CogPacket compressed: %d → %d tokens (saved %d, %.0f%%)",
        original_tokens, compressed_tokens, savings, savings_pct,
    )
    for name, tokens, reason in skipped:
        logger.debug("  Skipped '%s' (%d tokens): %s", name, tokens, reason)
    for name in kept:
        logger.debug("  Kept '%s' (%d tokens)", name, sections[name]["tokens_est"])

    return compressed


def get_compression_stats(full_prompt: str, kv_cache=None, awareness=None,
                           sae_confident_topics=None) -> dict:
    """Get compression statistics without actually compressing."""
    sections = identify_sections(full_prompt)
    total_tokens = len(full_prompt) // 3

    stats = {
        "total_tokens": total_tokens,
        "sections_found": len(sections),
        "sections": {},
    }

    for name, section in sections.items():
        can_skip = False
        reason = "no cache"

        if kv_cache and hasattr(kv_cache, '_hashes'):
            if kv_cache._hashes.get(name) == section["hash"]:
                can_skip = True
                reason = "KV cached"

        if sae_confident_topics and name in sae_confident_topics:
            can_skip = True
            reason = "SAE weight-baked"

        stats["sections"][name] = {
            "tokens": section["tokens_est"],
            "can_skip": can_skip,
            "reason": reason,
        }

    stats["compressible_tokens"] = sum(
        s["tokens"] for s in stats["sections"].values() if s["can_skip"]
    )
    stats["compression_pct"] = round(
        stats["compressible_tokens"] / max(1, total_tokens) * 100, 1
    )

    return stats
