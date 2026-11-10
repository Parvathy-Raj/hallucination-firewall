"""The Firewall facade used by the MCP server, the CLI and the Python SDK.

SDK use, where verification is enforced in code rather than left to the agent:

    fw = Firewall()
    fw.record("crm.lookup", tool_output)       # or decorate tools with @fw.watch
    answer = fw.guard(draft)                   # raises HallucinationBlocked if policy blocks
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hfw.claims import extract_claims
from hfw.config import Settings
from hfw.ledger import Ledger
from hfw.llm import LLM
from hfw.policy import Policy
from hfw.rewrite import rewrite
from hfw.types import Claim, ClaimVerdict, Label, Report
from hfw.verify.pipeline import Verifier


class HallucinationBlocked(Exception):
    def __init__(self, report: Report):
        self.report = report
        super().__init__(report.message)


class Firewall:
    def __init__(
        self,
        settings: Settings | None = None,
        policy: Policy | None = None,
        ledger: Ledger | None = None,
        llm: LLM | None = None,
        persist: bool = False,
    ):
        self.settings = settings or Settings()
        self.policy = policy or Policy()
        # `is None` checks: an empty Ledger is falsy (it defines __len__).
        if ledger is None:
            ledger = Ledger(
                self.settings.ledger_path if persist else None, embed_model=self.settings.embed_model
            )
        self.ledger = ledger
        self.llm = llm if llm is not None else LLM.from_settings(self.settings)
        self.verifier = Verifier(self.ledger, self.llm, self.settings.nli_model)
        self.history: dict[str, ClaimVerdict] = {}  # claim_id -> verdict (for explain_verdict)
        self.reports: list[Report] = []
        self._run = 0

    # -- evidence -----------------------------------------------------------------------
    def record(self, source: str, output: Any, args: dict[str, Any] | None = None) -> None:
        self.ledger.add(source, output, args)

    def watch(self, fn: Callable | None = None, *, name: str | None = None):
        """Decorator: record every return value of a tool function as evidence."""

        def deco(f: Callable):
            tool = name or f.__name__

            @functools.wraps(f)
            def wrapper(*a, **kw):
                out = f(*a, **kw)
                self.record(tool, out, kw or None)
                return out

            return wrapper

        return deco(fn) if fn else deco

    # -- verification -------------------------------------------------------------------
    def _extractor_mode(self) -> str:
        return self.settings.extractor

    def verify(self, draft: str, policy: Policy | None = None, claims: list[Claim] | None = None) -> Report:
        policy = policy or self.policy
        self.ledger.sync()
        t0 = time.perf_counter()
        usage0 = self.llm.usage.snapshot()
        self._run += 1
        claims = claims if claims is not None else extract_claims(draft, self.llm, self._extractor_mode())
        t_extract = time.perf_counter() - t0
        verdicts = []
        for c in claims:
            c.id = f"r{self._run}.{c.id}"
            v = self.verifier.verify(c, policy)
            self.history[c.id] = v
            verdicts.append(v)
        block, score, msg = policy.decide(verdicts)
        if not len(self.ledger):
            msg = (
                "No tool outputs are recorded yet, so nothing can be verified. Call your tools first "
                "(or use record_evidence), then verify again. " + msg
            )
        usage1 = self.llm.usage.snapshot()
        stats = {
            "claims": len(claims),
            "evidence_items": len(self.ledger),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "extract_ms": round(t_extract * 1000, 1),
            "decided_by_tier": {
                str(t): sum(v.tier == t for v in verdicts) for t in (0, 1, 2) if any(v.tier == t for v in verdicts)
            },
            "llm_calls": usage1["llm_calls"] - usage0["llm_calls"],
            "est_cost_usd": round(usage1["est_cost_usd"] - usage0["est_cost_usd"], 6),
        }
        rewritten = None
        flagged = any(v.label.is_flag for v in verdicts)
        if not flagged and not block:
            action = "pass"
        elif policy.action == "rewrite":
            action, rewritten = "rewrite", rewrite(draft, verdicts)
        elif block and policy.action == "block":
            action = "block"
        else:
            action = "annotate"
        report = Report(verdicts, action, action == "block", score, msg, stats, rewritten)
        self.reports.append(report)
        self._mark_verified(report)
        return report

    def verify_claim(self, claim: str, policy: Policy | None = None) -> ClaimVerdict:
        self.ledger.sync()
        self._run += 1
        c = Claim(id=f"r{self._run}.c1", text=claim, kind="factual", sentence=claim)
        v = self.verifier.verify(c, policy or self.policy)
        self.history[c.id] = v
        return v

    def rewrite_grounded(self, draft: str, unsupported: str = "hedge") -> tuple[str, Report]:
        report = self.verify(draft)
        return rewrite(draft, report.verdicts, unsupported), report

    def guard(self, draft: str) -> str:
        """Return the draft if it passes; the rewrite if policy.action == 'rewrite'; else raise."""
        report = self.verify(draft)
        if report.blocked:
            raise HallucinationBlocked(report)
        return report.rewritten or draft

    # -- introspection ------------------------------------------------------------------
    def explain(self, claim_id: str) -> dict[str, Any]:
        v = self.history.get(claim_id)
        if v is None:
            known = list(self.history)[-10:]
            raise KeyError(f"unknown claim_id {claim_id!r}; recent ids: {known}")
        d = v.to_dict(max_evidence=4, snippet_chars=1200)
        d["tier_trace"] = v.trace
        d["sentence_in_draft"] = v.claim.sentence
        return d

    def session_report(self) -> str:
        lines = ["# Hallucination Firewall: session report", ""]
        lines.append(f"Evidence items recorded: {len(self.ledger)}")
        lines.append(f"Answers verified: {len(self.reports)}")
        all_v = [v for r in self.reports for v in r.verdicts]
        counts = {l: sum(v.label == l for v in all_v) for l in Label}
        lines.append(
            "Claims: " + ", ".join(f"{l.icon} {l.value} {n}" for l, n in counts.items())
        )
        lines.append(f"LLM usage: {json.dumps(self.llm.usage.snapshot())}")
        lines.append("")
        for i, r in enumerate(self.reports, 1):
            lines.append(f"## Answer {i}: {r.action} (grounding {r.score:.0%})")
            for v in r.verdicts:
                lines.append(f"- {v.label.icon} `{v.claim.id}` {v.claim.text}  ")
                if v.label.is_flag:
                    lines.append(f"  {v.reason}")
            lines.append("")
        return "\n".join(lines)

    def _mark_verified(self, report: Report) -> None:
        """Leave a marker for the Claude Code Stop hook."""
        try:
            d: Path = self.settings.state_dir
            d.mkdir(parents=True, exist_ok=True)
            (d / "last_verify.json").write_text(
                json.dumps({"ts": time.time(), "action": report.action, "blocked": report.blocked}),
                encoding="utf-8",
            )
        except OSError:
            pass
