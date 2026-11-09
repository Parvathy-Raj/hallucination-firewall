"""Claim extraction: split a draft answer into atomic, checkable claims.

Two extractors:
* LLM: decomposes each sentence into single-fact claims and labels each one
  factual / opinion / hedge.
* Heuristic: sentence split plus clause split, with keyword rules for the label.
  Needs no model, so the firewall still works offline.
"""

from __future__ import annotations

import re

from hfw.llm import LLM, LLMError
from hfw.text import split_sentences

HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|perhaps|likely|unlikely|probably|i think|i believe|"
    r"it seems|appears to|suggests?|roughly|i'?m not sure|uncertain)\b",
    re.IGNORECASE,
)
OPINION_RE = re.compile(
    r"\b(i recommend|we recommend|you should|consider|best|worst|great|excellent|"
    r"terrible|important to|i'?d suggest|in my (?:view|opinion)|good idea|beautiful|"
    r"interesting)\b",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"^(here(?:'s| is| are)|let me|i (?:hope|can|will|'ll)|feel free|in summary|to summarize|"
    r"sure[,!]|certainly[,!]|great question|"
    r"(?:based on|according to|from) the (?:given |provided |above |retrieved )?(?:tool|search|results?|passages?|information|context|documents?|sources?|data)[^.]*:$|"
    r"according to (?:my|the) (?:search|tools?)[^,]*,?$|sources?:|references?:)",
    re.IGNORECASE,
)
CITATION_ONLY_RE = re.compile(r"^\W*(\[\d+\]\W*)+$")


def classify(text: str) -> str:
    t = text.strip()
    if t.endswith(":") or (META_RE.match(t) and not re.search(r"\d", t)):
        return "meta"
    if t.endswith("?"):
        return "opinion"
    if OPINION_RE.search(t) and not re.search(r"\d", t):
        return "opinion"
    if HEDGE_RE.search(t):
        return "hedge"
    return "factual"


def _clean(s: str) -> str:
    s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", s)  # bullets / numbering
    s = re.sub(r"[*_`#>]+", "", s)  # markdown emphasis
    return " ".join(s.split())


def _split_clauses(sentence: str) -> list[str]:
    """Split on '; ' and ', and/while/whereas' when both sides look like full clauses."""
    parts = re.split(r";\s+|,\s+(?:and|but|while|whereas)\s+(?=[a-z]*\s*\w+\s+(?:is|are|was|were|has|have|had|must|will|can|grew|fell|rose)\b)", sentence)
    parts = [p.strip(" ,;") for p in parts if p and len(p.split()) >= 3]
    return parts or [sentence]


def heuristic_claims(draft: str) -> list[tuple[str, str, str, int]]:
    """Return (claim_text, kind, sentence, sentence_index) tuples."""
    out = []
    for si, sent in enumerate(split_sentences(draft)):
        s = _clean(sent)
        if not s or CITATION_ONLY_RE.match(s) or len(s.split()) < 3:
            continue
        if (META_RE.match(s) and not re.search(r"\d", s)) or s.endswith(":"):
            continue
        for clause in _split_clauses(s):
            out.append((clause, classify(clause), sent, si))
    return out


EXTRACT_SYSTEM = """You split an AI assistant's draft answer into atomic claims for fact-checking.

Rules:
- Each claim states exactly ONE fact and must be understandable on its own: resolve pronouns
  and carry over the subject, time period and units from the sentence ("it grew 12%" ->
  "Acme's revenue grew 12% in Q3 2024").
- Keep every number, date, name and unit exactly as written in the draft. Never fix or add facts.
- Label each claim: "factual" (verifiable statement), "opinion" (advice, judgement,
  recommendation), or "hedge" (explicitly uncertain: may/might/probably).
- Skip greetings, filler, offers of help and pure formatting.
- "sentence" is the index of the numbered sentence the claim came from.

Return JSON only: {"claims": [{"text": "...", "kind": "factual", "sentence": 0}]}"""


def llm_claims(draft: str, llm: LLM) -> list[tuple[str, str, str, int]]:
    sents = split_sentences(draft)
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    data = llm.complete_json(EXTRACT_SYSTEM, f"Draft answer, one sentence per line:\n{numbered}", 2000)
    items = data.get("claims", []) if isinstance(data, dict) else data
    out = []
    for it in items:
        if not isinstance(it, dict) or not str(it.get("text", "")).strip():
            continue
        kind = str(it.get("kind", "factual")).lower()
        if kind not in ("factual", "opinion", "hedge"):
            kind = "factual"
        try:
            si = int(it.get("sentence", 0))
        except (TypeError, ValueError):
            si = 0
        si = max(0, min(si, len(sents) - 1)) if sents else 0
        out.append((it["text"].strip(), kind, sents[si] if sents else "", si))
    return out


def extract_claims(draft: str, llm: LLM | None = None, mode: str = "auto"):
    """Return list[Claim]. mode: auto | llm | heuristic | sentences."""
    from hfw.types import Claim

    if mode == "sentences":  # one claim per sentence (used for span-labelled benchmarks)
        raw = [
            (_clean(s), classify(_clean(s)), s, i)
            for i, s in enumerate(split_sentences(draft))
            if len(_clean(s).split()) >= 3
        ]
    elif mode in ("llm", "auto") and llm is not None and llm.available:
        try:
            raw = llm_claims(draft, llm)
        except LLMError:
            raw = heuristic_claims(draft)
        if not raw and draft.strip():
            raw = heuristic_claims(draft)
    else:
        raw = heuristic_claims(draft)
    return [
        Claim(id=f"c{i + 1}", text=t, kind=k, sentence=s, sentence_index=si)
        for i, (t, k, s, si) in enumerate(raw)
    ]
