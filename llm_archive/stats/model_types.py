"""What kind of model each recorded model string is, and an honest account of how little
of that can actually be read off the name.

Like `pricing.py`'s $/Mtok table, this is a hand-maintained lookup, not something derived
from the archive. "Legacy" and "reasoning tier" in particular are judgment calls — there is
no substring that reliably means "this generation is superseded" — so `MODEL_TYPES` only
carries entries that need to be told apart from the keyword-based default, seeded from what
this archive has actually recorded. Update it rather than trusting it.

Not to be confused with `chunk.model_tag` elsewhere in the codebase, which names the
*embedding* model version used to build search vectors.
"""

from __future__ import annotations

from . import pricing

TYPES = ("general", "reasoning", "vision", "image-generation", "audio", "embedding", "legacy")

# Explicit overrides: models the keyword heuristic below would get wrong, or that carry
# no distinguishing keyword at all.
MODEL_TYPES: dict[str, str] = {
    # explicit reasoning-tier variants
    "gpt-5.2-reasoning": "reasoning",
    "claude-4.5-sonnet-reasoning": "reasoning",
    "gemini-3-flash-thinking": "reasoning",
    # image generation
    "gemini-3-pro-image-preview": "image-generation",
    # superseded by the 4.x/5-series Claude, GPT-5 family, and Gemini 3 family
    # respectively, still in use elsewhere in this archive
    "claude-3.5-sonnet": "legacy",
    "gpt-4.1": "legacy",
    "gemini-2.5-flash": "legacy",
    "gemini-2.5-flash-lite": "legacy",
}

# Fallback only for model strings with no explicit entry above. Ordered so a more specific
# word (e.g. "vision") is checked before a more general one that might also appear.
_KEYWORDS = (
    ("image", "image-generation"),
    ("vision", "vision"),
    ("embed", "embedding"),
    ("audio", "audio"),
    ("whisper", "audio"),
    ("tts", "audio"),
    ("voice", "audio"),
    ("reasoning", "reasoning"),
    ("thinking", "reasoning"),
)


def classify(model: str | None) -> str | None:
    """Return one of `TYPES`, or None if `model` isn't a real recorded model."""
    key = pricing.normalise(model, MODEL_TYPES)
    if key is not None:
        return MODEL_TYPES[key]

    if not model:
        return None
    name = model.strip().lower()
    if name in ("<synthetic>", "unknown", "none"):
        return None

    for word, kind in _KEYWORDS:
        if word in name:
            return kind
    return "general"


def model_type_index(con) -> dict[str, list[tuple[str, int]]]:
    """Recorded `model_primary` values, grouped by classified type.

    Computed in Python because the type isn't a DB column — `classify` is a lookup table
    plus a keyword heuristic, not something SQL can express. Shared by the web UI's
    `/browse` facet and CLI/web `--model-type` export filters, so both mean the same thing.
    """
    rows = con.execute("""
        SELECT model_primary, COUNT(*) n FROM session
        WHERE model_primary IS NOT NULL AND model_primary != ''
        GROUP BY model_primary""").fetchall()
    index: dict[str, list[tuple[str, int]]] = {}
    for r in rows:
        kind = classify(r["model_primary"])
        if kind is not None:
            index.setdefault(kind, []).append((r["model_primary"], r["n"]))
    return index
