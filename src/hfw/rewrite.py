"""Grounded rewrite: fix contradicted figures, hedge or drop unsupported claims, add citations.

The rewrite is deterministic on purpose: an LLM rewriter could add new
hallucinations of its own. Corrections come from the evidence spans that
tier 0 / tier 2 identified.
"""

from __future__ import annotations

import re

from hfw.types import ClaimVerdict, Label

_SEVERITY = {Label.CONTRADICTED: 3, Label.UNSUPPORTED: 2, Label.SUPPORTED: 1, Label.NOT_CHECKABLE: 0}


def _locate(draft: str, sentence: str) -> tuple[int, int] | None:
    words = sentence.split()
    if not words:
        return None
    pat = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pat, draft)
    return (m.start(), m.end()) if m else None


def rewrite(draft: str, verdicts: list[ClaimVerdict], unsupported: str = "hedge") -> str:
    """unsupported: 'hedge' keeps the sentence with a marker, 'remove' drops it."""
    sources: dict[str, int] = {}
    source_lines: list[str] = []

    def cite(v: ClaimVerdict) -> str:
        if not v.evidence:
            return ""
        e = v.evidence[0]
        if e.chunk_id not in sources:
            sources[e.chunk_id] = len(sources) + 1
            snippet = " ".join(e.text.split())[:180]
            source_lines.append(f"[{sources[e.chunk_id]}] {e.source}: \"{snippet}…\"")
        return f"[{sources[e.chunk_id]}]"

    # group verdicts by the draft sentence they came from
    by_sentence: dict[int, list[ClaimVerdict]] = {}
    for v in verdicts:
        by_sentence.setdefault(v.claim.sentence_index, []).append(v)

    edits: list[tuple[int, int, str]] = []
    for _, vs in sorted(by_sentence.items()):
        sentence = vs[0].claim.sentence or vs[0].claim.text
        loc = _locate(draft, sentence)
        if loc is None:
            continue
        start, end = loc
        text = draft[start:end]
        worst = max(vs, key=lambda v: _SEVERITY[v.label]).label
        cites: list[str] = []
        fixed_all = True
        for v in vs:
            if v.label == Label.CONTRADICTED:
                if v.correction and v.correction[0] in text:
                    text = text.replace(v.correction[0], v.correction[1], 1)
                    cites.append(cite(v))
                else:
                    fixed_all = False
            elif v.label == Label.SUPPORTED:
                cites.append(cite(v))
        if worst == Label.CONTRADICTED and not fixed_all:
            refs = " ".join(c for c in (cite(v) for v in vs if v.label == Label.CONTRADICTED) if c)
            text = f"~~{text.rstrip()}~~ *(removed: contradicted by the tool evidence {refs})*"
        elif worst == Label.UNSUPPORTED and not any(v.label == Label.CONTRADICTED for v in vs):
            if unsupported == "remove":
                text = ""
            else:
                text = _append(text, "*(not confirmed by any tool output)*")
        cite_str = "".join(dict.fromkeys(c for c in cites if c))
        if cite_str and text and not text.startswith("~~"):
            text = _append(text, cite_str)
        edits.append((start, end, text))

    out = draft
    for start, end, text in sorted(edits, reverse=True):
        out = out[:start] + text + out[end:]
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if source_lines:
        out += "\n\nSources (tool outputs):\n" + "\n".join(source_lines)
    return out


def _append(sentence: str, tag: str) -> str:
    m = re.search(r"([.!?:;]*)(\s*)$", sentence)
    punct = m.group(1) if m else ""
    core = sentence[: len(sentence) - len(m.group(0))] if m else sentence
    return f"{core} {tag}{punct}"
