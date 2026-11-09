"""Policy: how strict the firewall is and what it does with a flagged answer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hfw.types import ClaimVerdict, Label

PRESETS: dict[str, dict[str, Any]] = {
    # Only block on clear contradictions; let unsupported claims through with a note.
    "lenient": {"block_on": ["contradicted"], "threshold": 0.9, "min_score": 0.0},
    # Block contradictions, annotate unsupported claims.
    "balanced": {"block_on": ["contradicted"], "threshold": 0.8, "min_score": 0.0},
    # Every factual claim must be backed by tool evidence.
    "strict": {"block_on": ["contradicted", "unsupported"], "threshold": 0.7, "min_score": 0.8},
}


@dataclass
class Policy:
    strictness: str = "balanced"
    # Which tiers may run. 0 = exact, 1 = NLI model, 2 = LLM judge.
    tiers: list[int] = field(default_factory=lambda: [0, 1, 2])
    block_on: list[str] = field(default_factory=lambda: ["contradicted"])
    # action on a flagged answer: annotate | block | rewrite
    action: str = "block"
    # Tier-1 confidence needed to decide without the judge.
    threshold: float = 0.8
    # Block if the share of supported checkable claims is below this.
    min_score: float = 0.0
    top_k: int = 3

    @classmethod
    def preset(cls, strictness: str = "balanced", **overrides: Any) -> Policy:
        base = dict(PRESETS.get(strictness, PRESETS["balanced"]))
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(strictness=strictness, **base)

    def update(
        self,
        strictness: str | None = None,
        tiers: list[int] | None = None,
        block_on: list[str] | None = None,
        action: str | None = None,
        threshold: float | None = None,
    ) -> Policy:
        if strictness:
            if strictness not in PRESETS:
                raise ValueError(f"strictness must be one of {list(PRESETS)}")
            p = PRESETS[strictness]
            self.strictness = strictness
            self.block_on, self.threshold, self.min_score = list(p["block_on"]), p["threshold"], p["min_score"]
        if tiers is not None:
            bad = [t for t in tiers if t not in (0, 1, 2)]
            if bad or not tiers:
                raise ValueError("tiers must be a non-empty subset of [0, 1, 2]")
            self.tiers = sorted(set(tiers))
        if block_on is not None:
            valid = {l.value for l in Label}
            if any(b not in valid for b in block_on):
                raise ValueError(f"block_on values must be in {sorted(valid)}")
            self.block_on = list(block_on)
        if action is not None:
            if action not in ("annotate", "block", "rewrite"):
                raise ValueError("action must be annotate, block or rewrite")
            self.action = action
        if threshold is not None:
            if not 0.5 <= threshold <= 1.0:
                raise ValueError("threshold must be between 0.5 and 1.0")
            self.threshold = threshold
        return self

    def decide(self, verdicts: list[ClaimVerdict]) -> tuple[bool, float, str]:
        """Return (should_block, grounding_score, message)."""
        checkable = [v for v in verdicts if v.label != Label.NOT_CHECKABLE]
        supported = sum(v.label == Label.SUPPORTED for v in checkable)
        score = supported / len(checkable) if checkable else 1.0
        hits = [v for v in verdicts if v.label.value in self.block_on]
        block = bool(hits) or (bool(checkable) and score < self.min_score)
        n_contra = sum(v.label == Label.CONTRADICTED for v in verdicts)
        n_unsup = sum(v.label == Label.UNSUPPORTED for v in verdicts)
        if not checkable:
            msg = "No checkable factual claims."
        elif not (n_contra or n_unsup):
            msg = f"All {len(checkable)} factual claims are supported by tool evidence."
        else:
            parts = []
            if n_contra:
                parts.append(f"{n_contra} contradicted by tool evidence")
            if n_unsup:
                parts.append(f"{n_unsup} not supported by any tool output")
            msg = f"{len(checkable)} factual claims checked: " + ", ".join(parts) + "."
        if block:
            msg += " Do NOT send this draft as-is: fix the flagged claims (see corrections and evidence) or call rewrite_grounded."
        return block, score, msg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
