"""Build agent traces with planted errors from the UAE regulatory RAG corpus.

Each trace simulates an agent turn: the tool returned the gold passage plus two
distractor passages, and the draft answer contains four claims:

  * a correct sentence copied from the gold passage            (supported)
  * a sentence whose number was changed                        (number error)
  * a sentence with its obligation negated                     (negation error)
  * a sentence from an unrelated guide that was never retrieved (unsupported)

The corpus is not redistributed: run this against your own copy of chunks.jsonl.

    python evals/custom_traces/build.py /path/to/uae-regulatory-rag/data/chunks.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hfw.text import split_sentences
from hfw.verify.exact import extract_quantities

NEGATIONS = [
    (r"\bmust\b", "does not need to"),
    (r"\bis required to\b", "is not required to"),
    (r"\bare required to\b", "are not required to"),
    (r"\bcan\b", "cannot"),
    (r"\bshould\b", "should not"),
    (r"\bwill be\b", "will not be"),
    (r"\bis\b", "is not"),
    (r"\bare\b", "are not"),
]


def good_sentence(s: str) -> bool:
    return 60 <= len(s) <= 260 and s[0].isupper() and s.endswith(".") and "Example" not in s


def change_number(s: str, rng: random.Random) -> str | None:
    qs = [q for q in extract_quantities(s) if q.kind not in ("year", "quarter", "date") and q.value >= 2]
    if not qs:
        return None
    q = rng.choice(qs)
    num = re.search(r"\d[\d,]*(?:\.\d+)?", q.surface)
    if not num:
        return None
    raw = num.group(0).replace(",", "")
    value = float(raw)
    factor = rng.choice([0.8, 1.25, 1.5, 2.0, 0.5])
    new = value * factor
    new_s = f"{new:,.0f}" if "," in num.group(0) or new >= 10000 else (f"{new:g}" if "." in raw else str(int(round(new))))
    if new_s == num.group(0):
        new_s = str(int(value) + 1)
    surface = q.surface.replace(num.group(0), new_s, 1)
    return s[: q.start] + surface + s[q.end :]


def negate(s: str) -> str | None:
    for pat, rep in NEGATIONS:
        if re.search(pat, s) and not re.search(r"\bnot\b|n't\b", s):
            return re.sub(pat, rep, s, count=1)
    return None


def main(chunks_path: str, n: int = 20, seed: int = 7) -> None:
    rng = random.Random(seed)
    chunks = [json.loads(l) for l in Path(chunks_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(chunks)
    out = []
    for gold in chunks:
        if len(out) >= n:
            break
        sents = [s for s in split_sentences(gold["text"]) if good_sentence(s)]
        num_s = [s for s in sents if change_number(s, rng)]
        neg_s = [s for s in sents if negate(s)]
        if len(sents) < 2 or not num_s or not neg_s:
            continue
        correct = rng.choice([s for s in sents if s not in num_s[:1]] or sents)
        n_src = rng.choice(num_s)
        g_src = rng.choice([s for s in neg_s if s != n_src] or neg_s)
        others = [c for c in chunks if c["source_id"] != gold["source_id"]]
        distractors = rng.sample([c for c in others if c["domain"] == gold["domain"]] or others, 2)
        foreign = rng.choice([c for c in others if c not in distractors])
        f_sents = [s for s in split_sentences(foreign["text"]) if good_sentence(s)]
        if not f_sents:
            continue
        claims = [
            {"text": correct, "hallucinated": False, "type": "none"},
            {"text": change_number(n_src, rng), "hallucinated": True, "type": "number"},
            {"text": negate(g_src), "hallucinated": True, "type": "negation"},
            {"text": rng.choice(f_sents), "hallucinated": True, "type": "unsupported"},
        ]
        rng.shuffle(claims)
        for i, c in enumerate(claims):
            c["sentence_index"] = i
        evidence = [("uae_docs.search_guides", c["text"]) for c in [gold] + distractors]
        rng.shuffle(evidence)
        out.append(
            {"id": f"uae:{gold['chunk_id']}", "dataset": "UAE-traces", "evidence": evidence,
             "response": " ".join(c["text"] for c in claims), "claims": claims}
        )
    dest = Path(__file__).parent / "uae_traces.jsonl"
    dest.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} traces ({sum(len(t['claims']) for t in out)} claims) to {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../uae-regulatory-rag/data/chunks.jsonl")
