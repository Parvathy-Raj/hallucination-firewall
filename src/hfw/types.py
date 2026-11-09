"""Shared data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    NOT_CHECKABLE = "not_checkable"

    @property
    def icon(self) -> str:
        return {
            Label.SUPPORTED: "✅",
            Label.CONTRADICTED: "❌",
            Label.UNSUPPORTED: "⚠️",
            Label.NOT_CHECKABLE: "➖",
        }[self]

    @property
    def is_flag(self) -> bool:
        return self in (Label.CONTRADICTED, Label.UNSUPPORTED)


@dataclass
class Claim:
    """One atomic statement taken from a draft answer."""

    id: str
    text: str
    kind: str = "factual"  # factual | opinion | hedge
    sentence: str = ""  # the draft sentence the claim came from
    sentence_index: int = 0


@dataclass
class EvidenceRef:
    chunk_id: str
    evidence_id: str
    source: str  # tool name that produced it
    text: str
    score: float = 0.0


@dataclass
class ClaimVerdict:
    claim: Claim
    label: Label
    confidence: float
    tier: int  # tier that made the final call (-1 = no tier ran)
    reason: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    # A suggested replacement for a contradicted span, e.g. ("AED 500,000", "AED 375,000").
    correction: tuple[str, str] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)  # what each tier said

    def to_dict(self, max_evidence: int = 2, snippet_chars: int = 400) -> dict[str, Any]:
        return {
            "claim_id": self.claim.id,
            "claim": self.claim.text,
            "kind": self.claim.kind,
            "verdict": self.label.value,
            "icon": self.label.icon,
            "confidence": round(self.confidence, 3),
            "decided_by_tier": self.tier,
            "reason": self.reason,
            "correction": (
                {"wrong": self.correction[0], "evidence_says": self.correction[1]}
                if self.correction
                else None
            ),
            "evidence": [
                {
                    "chunk_id": e.chunk_id,
                    "tool": e.source,
                    "snippet": e.text[:snippet_chars],
                }
                for e in self.evidence[:max_evidence]
            ],
        }


@dataclass
class Report:
    verdicts: list[ClaimVerdict]
    action: str  # pass | annotate | block | rewrite
    blocked: bool
    score: float  # share of checkable claims that are supported
    message: str
    stats: dict[str, Any] = field(default_factory=dict)
    rewritten: str | None = None

    @property
    def flagged(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.label.is_flag]

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for v in self.verdicts:
            counts[v.label.value] = counts.get(v.label.value, 0) + 1
        out = {
            "action": self.action,
            "blocked": self.blocked,
            "grounding_score": round(self.score, 3),
            "message": self.message,
            "counts": counts,
            "claims": [v.to_dict() for v in self.verdicts],
            "stats": self.stats,
        }
        if self.rewritten is not None:
            out["rewritten_answer"] = self.rewritten
        return out


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
