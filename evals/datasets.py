"""Benchmark loaders. Every example becomes the same shape:

    {"id", "dataset", "evidence": [(source, text)], "response",
     "claims": [{"text", "sentence_index", "hallucinated": bool, "type": str}]}

Claims are response sentences, so span-level human labels map onto them directly
and every tier configuration sees exactly the same claims.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import httpx

from hfw.claims import _clean
from hfw.text import split_sentences

DATA = Path(__file__).parent / "data"
RAGTRUTH = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{}"
HALUEVAL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return dest


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    out, pos = [], 0
    for s in split_sentences(text):
        pat = r"\s+".join(re.escape(w) for w in s.split())
        m = re.compile(pat).search(text, pos)
        if not m:
            continue
        out.append((s, m.start(), m.end()))
        pos = m.end()
    return out


def ragtruth(n_hal: int = 10, n_clean: int = 4, seed: int = 0) -> list[dict]:
    resp = _download(RAGTRUTH.format("response.jsonl"), DATA / "ragtruth_response.jsonl")
    src = _download(RAGTRUTH.format("source_info.jsonl"), DATA / "ragtruth_source_info.jsonl")
    sources = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        sources[d["source_id"]] = d
    by_task: dict[str, list[dict]] = {}
    for line in resp.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("split") != "test" or d["source_id"] not in sources:
            continue
        by_task.setdefault(sources[d["source_id"]]["task_type"], []).append(d)
    rng = random.Random(seed)
    examples = []
    for task, items in sorted(by_task.items()):
        # mostly responses with hallucinations (positives are rare at sentence level), plus
        # clean ones so false flags on fully correct answers are measured too
        hal = [d for d in items if d.get("labels")]
        clean = [d for d in items if not d.get("labels")]
        rng.shuffle(hal)
        rng.shuffle(clean)
        for d in hal[:n_hal] + clean[:n_clean]:
            s = sources[d["source_id"]]
            info = s["source_info"]
            if task == "QA" and isinstance(info, dict):
                evidence = [("search_results", info.get("passages", ""))]
            elif isinstance(info, (dict, list)):
                evidence = [("data_lookup", json.dumps(info, ensure_ascii=False))]
            else:
                evidence = [("fetch_document", str(info))]
            claims = []
            for si, (sent, a, b) in enumerate(_sentence_spans(d["response"])):
                text = _clean(sent)
                if len(text.split()) < 3:
                    continue
                overlapping = [l for l in d.get("labels", []) if l["start"] < b and l["end"] > a]
                claims.append(
                    {
                        "text": text,
                        "sentence_index": si,
                        "hallucinated": bool(overlapping),
                        "type": overlapping[0]["label_type"] if overlapping else "none",
                    }
                )
            if claims:
                examples.append(
                    {"id": f"ragtruth:{d['id']}", "dataset": f"RAGTruth-{task}", "evidence": evidence,
                     "response": d["response"], "claims": claims}
                )
    return examples


def halueval(n: int = 30, seed: int = 0) -> list[dict]:
    path = _download(HALUEVAL, DATA / "halueval_qa.json")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples = []
    for i, r in enumerate(rows[:n]):
        for kind, ans, hal in (("right", r["right_answer"], False), ("hallucinated", r["hallucinated_answer"], True)):
            claim = f'The answer to "{r["question"]}" is {ans.rstrip(".")}.'
            examples.append(
                {"id": f"halueval:{i}:{kind}", "dataset": "HaluEval-QA",
                 "evidence": [("wiki_search", r["knowledge"])], "response": claim,
                 "claims": [{"text": claim, "sentence_index": 0, "hallucinated": hal,
                             "type": "qa_wrong_answer" if hal else "none"}]}
            )
    return examples


def uae_traces(path: Path | None = None) -> list[dict]:
    p = path or Path(__file__).parent / "custom_traces" / "uae_traces.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
