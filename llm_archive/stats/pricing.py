"""Model prices, and an honest account of what the resulting number means.

**This estimates list-price API cost, not what you were billed.** Most of this
archive was not billed per token at all:

  * Claude Code here runs on a Claude subscription, not metered API billing.
  * `copilot/*` models are covered by a GitHub Copilot subscription.
  * `kimi-k2.5-free` and other `:free` endpoints cost nothing.

So read the figure as *"what this volume of tokens would list for at API rates"* —
useful for comparing projects, models and months against each other, and misleading
if taken as money that left your account. `billed` marks the rows where a per-token
charge plausibly applied.

Claude rates are first-party API list prices. Cache multipliers come from the same
source: a cache **read** costs ~0.1x the input rate and a cache **write** ~1.25x.
That distinction dominates everything here — this archive holds 1.4 *billion*
cache-read tokens, which price at ~$715 rather than ~$7,150.

Non-Anthropic rates are best-effort and flagged `verified=False`. Update them
rather than trusting them.
"""

from __future__ import annotations

from dataclasses import dataclass

CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    verified: bool = True
    billed: bool = True          # False = covered by a subscription or free tier
    note: str = ""

    def read_rate(self) -> float:
        return (self.cache_read if self.cache_read is not None
                else self.input * CACHE_READ_MULTIPLIER)

    def write_rate(self) -> float:
        return (self.cache_write if self.cache_write is not None
                else self.input * CACHE_WRITE_MULTIPLIER)


# Anthropic first-party API list prices.
_CLAUDE = {
    "claude-fable-5":   Price(10.0, 50.0),
    "claude-mythos-5":  Price(10.0, 50.0),
    "claude-opus-5":    Price(5.0, 25.0),
    "claude-opus-4-8":  Price(5.0, 25.0),
    "claude-opus-4-7":  Price(5.0, 25.0),
    "claude-opus-4-6":  Price(5.0, 25.0),
    "claude-sonnet-5":  Price(3.0, 15.0),
    "claude-sonnet-4-6": Price(3.0, 15.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
}

PRICES: dict[str, Price] = {
    **_CLAUDE,
    # Non-Anthropic — best effort, verify before trusting.
    "gpt-5-mini":     Price(0.25, 2.0, verified=False),
    "gpt-5":          Price(1.25, 10.0, verified=False),
    "gpt-5.2-codex":  Price(1.25, 10.0, verified=False),
    "gpt-5.3-codex":  Price(1.25, 10.0, verified=False),
    "gpt-4.1":        Price(2.0, 8.0, verified=False),
}

# Everything a Copilot subscription covers, and free endpoints: real usage, no
# per-token charge. Counted in token totals, excluded from spend.
SUBSCRIPTION_PREFIXES = ("copilot/",)
FREE_SUFFIXES = (":free", "-free")


def normalise(model: str | None, table: dict | None = None) -> str | None:
    """Map a recorded model string onto a key of `table` (default `PRICES`).

    Shared with `stats/model_types.py`, which classifies the same model strings
    against a different table and wants the same id-matching rules (Copilot
    prefix stripping, dot/dash normalisation, prefix matching).
    """
    if table is None:
        table = PRICES
    if not model:
        return None
    name = model.strip().lower()
    if name in ("<synthetic>", "unknown", "none"):
        return None

    # copilot/claude-haiku-4.5 -> claude-haiku-4.5, but remember it was Copilot
    for prefix in SUBSCRIPTION_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]

    if name in table:
        return name
    # Anthropic writes 4.5 in model ids as 4-5; VS Code records the dotted form.
    dashed = name.replace(".", "-")
    if dashed in table:
        return dashed
    # tolerate date suffixes and vendor prefixes
    for key in table:
        if dashed.startswith(key):
            return key
    return None


def lookup(model: str | None) -> tuple[Price | None, str]:
    """Return (price, reason). `reason` explains a None or unbilled price."""
    if not model:
        return None, "no model recorded"
    name = model.strip().lower()

    if any(name.startswith(p) for p in SUBSCRIPTION_PREFIXES):
        key = normalise(model)
        base = PRICES.get(key) if key else None
        return (Price(base.input if base else 0.0, base.output if base else 0.0,
                      verified=False, billed=False,
                      note="GitHub Copilot subscription"),
                "subscription")
    if any(name.endswith(s) for s in FREE_SUFFIXES):
        return Price(0.0, 0.0, billed=False, note="free endpoint"), "free"

    key = normalise(model)
    if key is None:
        return None, "no price on file"
    return PRICES[key], "list price"


def cost(model: str | None, tok_in: int = 0, tok_out: int = 0,
         cache_read: int = 0, cache_write: int = 0) -> tuple[float, bool]:
    """Return (usd, billed). Unpriced models contribute 0 and are reported."""
    price, _ = lookup(model)
    if price is None:
        return 0.0, False
    usd = (
        (tok_in or 0) * price.input
        + (tok_out or 0) * price.output
        + (cache_read or 0) * price.read_rate()
        + (cache_write or 0) * price.write_rate()
    ) / 1_000_000
    return usd, price.billed


def coverage(models: list[str | None]) -> dict:
    """How much of the archive we can actually price — report this, don't hide it."""
    known = unpriced = subscription = free = 0
    missing: set[str] = set()
    for model in models:
        price, reason = lookup(model)
        if price is None:
            unpriced += 1
            if model:
                missing.add(model)
        elif reason == "subscription":
            subscription += 1
        elif reason == "free":
            free += 1
        else:
            known += 1
    return {"priced": known, "subscription": subscription, "free": free,
            "unpriced": unpriced, "missing_models": sorted(missing)}
