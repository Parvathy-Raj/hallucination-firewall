"""Tiered verification with confidence routing.

    claim ─▶ retrieve top-k evidence ─▶ tier 0 (exact) ─▶ tier 1 (NLI) ─▶ tier 2 (LLM judge)
                                         │ confident?        │ confident?
                                         └──── stop ◀────────┘

Each tier either decides or passes the claim on. The judge only sees the
claims the cheap tiers could not settle.
"""

from __future__ import annotations

import time

from hfw.ledger import Ledger
from hfw.llm import LLM
from hfw.policy import Policy
from hfw.text import tokenize
from hfw.types import Claim, ClaimVerdict, EvidenceRef, Label
from hfw.verify import exact
from hfw.verify.judge import judge
from hfw.verify.nli import NLIModel


class Verifier:
    def __init__(self, ledger: Ledger, llm: LLM, nli_model: str = ""):
        self.ledger = ledger
        self.llm = llm
        self.nli_model_name = nli_model
        self.timings: dict[str, float] = {"tier0": 0.0, "tier1": 0.0, "tier2": 0.0, "retrieval": 0.0}
        self.tier_counts: dict[int, int] = {-1: 0, 0: 0, 1: 0, 2: 0}

    @property
    def nli(self) -> NLIModel | None:
        return NLIModel.get(self.nli_model_name) if self.nli_model_name else None

    def verify(self, claim: Claim, policy: Policy) -> ClaimVerdict:
        v = self._verify(claim, policy)
        self.tier_counts[v.tier] = self.tier_counts.get(v.tier, 0) + 1
        return v

    def _verify(self, claim: Claim, policy: Policy) -> ClaimVerdict:
        trace: list[dict] = []
        if claim.kind != "factual":
            return ClaimVerdict(claim, Label.NOT_CHECKABLE, 1.0, -1, f"{claim.kind}, not a factual claim")

        t = time.perf_counter()
        # The judge gets a little more context than the cheap tiers.
        hits = self.ledger.search(claim.text, k=policy.top_k + 2)
        self.timings["retrieval"] += time.perf_counter() - t
        if not hits:
            return ClaimVerdict(
                claim, Label.UNSUPPORTED, 0.9, -1, "no tool output has been recorded this session"
            )
        all_passages = [c.text for c, _ in hits]
        passages = all_passages[: policy.top_k]
        refs = [EvidenceRef(c.id, c.evidence_id, c.source, c.text, s) for c, s in hits]

        def ordered(first: int | None) -> list[EvidenceRef]:
            if first is None or not 0 <= first < len(refs):
                return refs
            return [refs[first]] + [r for i, r in enumerate(refs) if i != first]

        # ---- tier 0: exact checks ------------------------------------------------
        r0 = None
        if 0 in policy.tiers:
            t = time.perf_counter()
            r0 = exact.check(claim.text, passages)
            self.timings["tier0"] += time.perf_counter() - t
            trace.append({"tier": 0, "result": r0.verdict, "confidence": round(r0.confidence, 3), "reason": r0.reason})
            if r0.verdict == "contradicted" and r0.confidence >= 0.75:
                return ClaimVerdict(
                    claim, Label.CONTRADICTED, r0.confidence, 0, r0.reason,
                    ordered(r0.chunk_index), r0.correction, trace,
                )
        number_missing = r0 is not None and r0.verdict in ("missing", "contradicted")

        # ---- tier 1: NLI model -----------------------------------------------------
        r1 = None
        if 1 in policy.tiers and self.nli is not None:
            t = time.perf_counter()
            r1 = self.nli.score(claim.text, passages)
            self.timings["tier1"] += time.perf_counter() - t
            trace.append({"tier": 1, "entail": round(r1.entail, 3), "contradict": round(r1.contra, 3)})
            if r1.contra >= policy.threshold and r1.contra > r1.entail:
                corr = r0.correction if r0 and r0.verdict == "contradicted" else None
                return ClaimVerdict(
                    claim, Label.CONTRADICTED, r1.contra, 1,
                    "the fact-checking model finds the evidence contradicts this claim"
                    + (f" ({r0.reason})" if corr else ""),
                    ordered(r1.best_index), corr, trace,
                )
            if r1.entail >= policy.threshold and not number_missing:
                return ClaimVerdict(
                    claim, Label.SUPPORTED, r1.entail, 1,
                    "the fact-checking model finds the evidence entails this claim",
                    ordered(r1.best_index), None, trace,
                )

        # ---- tier 2: LLM judge -----------------------------------------------------
        if 2 in policy.tiers and self.llm.available:
            t = time.perf_counter()
            r2 = judge(self.llm, claim.text, all_passages)
            self.timings["tier2"] += time.perf_counter() - t
            if r2 is not None and r2.confidence > 0:
                trace.append({"tier": 2, "label": r2.label, "confidence": r2.confidence, "reason": r2.reason})
                first = r2.evidence_indices[0] if r2.evidence_indices else None
                correction = r2.correction or (r0.correction if r0 and r0.verdict == "contradicted" else None)
                return ClaimVerdict(
                    claim, Label(r2.label), r2.confidence, 2, r2.reason or "LLM judge",
                    ordered(first), correction if r2.label == "contradicted" else None, trace,
                )

        # ---- fallback when no tier was confident -------------------------------------
        if r0 is not None and r0.verdict == "contradicted":
            return ClaimVerdict(claim, Label.CONTRADICTED, r0.confidence, 0, r0.reason,
                                ordered(r0.chunk_index), r0.correction, trace)
        if r1 is not None:
            if r1.contra >= 0.5 and r1.contra > r1.entail:
                return ClaimVerdict(claim, Label.CONTRADICTED, r1.contra, 1,
                                    "evidence leans towards contradicting this claim (low confidence)",
                                    ordered(r1.best_index), None, trace)
            if r1.entail >= 0.5 and not number_missing:
                return ClaimVerdict(claim, Label.SUPPORTED, r1.entail, 1,
                                    "evidence leans towards supporting this claim (low confidence)",
                                    ordered(r1.best_index), None, trace)
            return ClaimVerdict(claim, Label.UNSUPPORTED, max(r1.neutral, 0.5), 1,
                                r0.reason if number_missing and r0 else "no tool output states this",
                                refs, None, trace)
        # Tier 0 only: numbers decide; otherwise fall back to word coverage.
        if r0 is not None and r0.verdict == "missing":
            return ClaimVerdict(claim, Label.UNSUPPORTED, r0.confidence, 0, r0.reason, refs, None, trace)
        cov = _coverage(claim.text, passages[0])
        if r0 is not None and r0.verdict == "match" and cov >= 0.5:
            return ClaimVerdict(claim, Label.SUPPORTED, 0.7, 0, r0.reason, refs, None, trace)
        if cov >= 0.75:
            return ClaimVerdict(claim, Label.SUPPORTED, cov * 0.8, 0, "claim wording is covered by the evidence", refs, None, trace)
        return ClaimVerdict(claim, Label.UNSUPPORTED, 0.5, 0, "could not confirm the claim from tool evidence", refs, None, trace)


def _coverage(claim: str, passage: str) -> float:
    ct = set(tokenize(claim))
    if not ct:
        return 0.0
    return len(ct & set(tokenize(passage))) / len(ct)
