"""Tier 2: LLM judge. Only called when tiers 0 and 1 are not confident."""

from __future__ import annotations

from dataclasses import dataclass

from hfw.llm import LLM, LLMError

JUDGE_SYSTEM = """You are a strict fact-checker. Decide whether a CLAIM is supported by the
EVIDENCE passages, which are outputs from tools an AI agent called. Use only the evidence,
never your own knowledge.

Labels:
- "supported": the evidence states or directly implies the claim, including every number,
  date and name in it. Paraphrase, simple arithmetic, reasonable inferences and generic
  framing ("a great choice for events") are fine.
- "contradicted": the evidence states something incompatible with the claim (a different
  number, date, name, negation, or scope).
- "unsupported": the claim adds specific content (a fact, number, name, attribute, event or
  cause) that the evidence does not contain.

Return JSON only:
{"label": "supported|contradicted|unsupported", "confidence": 0.0-1.0,
 "evidence": [passage numbers used], "reason": "one sentence",
 "correction": {"wrong": "span from the claim", "evidence_says": "span from evidence"} or null}"""


@dataclass
class JudgeResult:
    label: str
    confidence: float
    reason: str
    evidence_indices: list[int]
    correction: tuple[str, str] | None


def judge(llm: LLM, claim: str, passages: list[str]) -> JudgeResult | None:
    if not llm.available:
        return None
    ev = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    try:
        data = llm.complete_json(JUDGE_SYSTEM, f"EVIDENCE:\n{ev}\n\nCLAIM: {claim}", 1200)
    except LLMError as e:
        return JudgeResult("unsupported", 0.0, f"judge failed: {e}", [], None)
    if not isinstance(data, dict):
        return None
    label = str(data.get("label", "unsupported")).lower().strip()
    if label not in ("supported", "contradicted", "unsupported"):
        label = "unsupported"
    try:
        conf = float(data.get("confidence", 0.7))
    except (TypeError, ValueError):
        conf = 0.7
    idx = []
    for x in data.get("evidence", []) or []:
        try:
            i = int(x) - 1
            if 0 <= i < len(passages):
                idx.append(i)
        except (TypeError, ValueError):
            continue
    corr = data.get("correction")
    correction = None
    if isinstance(corr, dict) and corr.get("wrong") and corr.get("evidence_says"):
        correction = (str(corr["wrong"]), str(corr["evidence_says"]))
    return JudgeResult(label, max(0.0, min(conf, 1.0)), str(data.get("reason", "")), idx, correction)
