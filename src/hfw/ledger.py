"""Evidence ledger: every tool output seen this session, chunked and indexed.

Retrieval is hybrid: BM25 always, plus dense embeddings (bge-small) when
sentence-transformers is installed. The two rankings are merged with reciprocal
rank fusion.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Plus

from hfw.text import normalize_tool_output, split_sentences, tokenize


@dataclass
class Evidence:
    id: str
    source: str  # tool name, e.g. "uae_docs__search"
    text: str
    args: dict[str, Any] = field(default_factory=dict)
    session: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class Chunk:
    id: str
    evidence_id: str
    source: str
    text: str


def chunk_text(text: str, max_chars: int = 600, overlap_sents: int = 1) -> list[str]:
    sents = split_sentences(text)
    if not sents:
        return []
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for s in sents:
        if cur and size + len(s) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
            size = sum(len(x) + 1 for x in cur)
        cur.append(s)
        size += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    # de-duplicate (overlap can create identical tail chunks)
    seen: set[str] = set()
    return [c for c in chunks if not (c in seen or seen.add(c))]


class _Embedder:
    _instances: dict[str, _Embedder] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge" in model_name.lower()
            else ""
        )

    @classmethod
    def get(cls, model_name: str) -> _Embedder | None:
        if not model_name:
            return None
        with cls._lock:
            if model_name not in cls._instances:
                try:
                    cls._instances[model_name] = cls(model_name)
                except Exception:  # missing extra or model download failure
                    cls._instances[model_name] = None  # type: ignore[assignment]
            return cls._instances[model_name]

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        )


class Ledger:
    """In-memory evidence store, optionally mirrored to a JSONL file."""

    def __init__(self, path: Path | str | None = None, embed_model: str = ""):
        self.path = Path(path) if path else None
        self.embed_model = embed_model
        self.evidence: dict[str, Evidence] = {}
        self.chunks: list[Chunk] = []
        self._bm25: BM25Plus | None = None
        self._emb: np.ndarray | None = None
        self._dirty = True
        self._file_pos = 0
        self._lock = threading.RLock()

    # -- writing --------------------------------------------------------------------
    def add(
        self,
        source: str,
        output: Any,
        args: dict[str, Any] | None = None,
        session: str = "",
        persist: bool = True,
    ) -> Evidence | None:
        text = normalize_tool_output(output).strip()
        if not text:
            return None
        ev = Evidence(id=uuid.uuid4().hex[:8], source=source, text=text, args=args or {}, session=session)
        with self._lock:
            self._ingest(ev)
            if persist and self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
                self._file_pos = self.path.stat().st_size
        return ev

    def _ingest(self, ev: Evidence) -> None:
        if ev.id in self.evidence:
            return
        self.evidence[ev.id] = ev
        for i, c in enumerate(chunk_text(ev.text)):
            self.chunks.append(Chunk(id=f"{ev.id}:{i}", evidence_id=ev.id, source=ev.source, text=c))
        self._dirty = True

    def sync(self) -> None:
        """Pick up entries other processes (e.g. the Claude Code hook) appended to the file."""
        if not self.path or not self.path.exists():
            return
        with self._lock:
            size = self.path.stat().st_size
            if size < self._file_pos:  # file was reset
                self.clear(keep_file=True)
                self._file_pos = 0
            if size == self._file_pos:
                return
            with self.path.open("r", encoding="utf-8") as f:
                f.seek(self._file_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self._ingest(Evidence(**d))
                    except (json.JSONDecodeError, TypeError):
                        continue
                self._file_pos = f.tell()

    def clear(self, keep_file: bool = False) -> None:
        with self._lock:
            self.evidence.clear()
            self.chunks.clear()
            self._dirty = True
            if not keep_file and self.path and self.path.exists():
                self.path.write_text("", encoding="utf-8")
                self._file_pos = 0

    # -- reading --------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.evidence)

    def _build(self) -> None:
        if not self._dirty:
            return
        corpus = [tokenize(c.text) or ["_"] for c in self.chunks]
        self._bm25 = BM25Plus(corpus) if corpus else None
        self._emb = None
        emb = _Embedder.get(self.embed_model)
        if emb is not None and self.chunks:
            self._emb = emb.encode([c.text for c in self.chunks])
        self._dirty = False

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        with self._lock:
            if not self.chunks:
                return []
            self._build()
            n = len(self.chunks)
            rankings: list[np.ndarray] = []
            if self._bm25 is not None:
                bm = np.asarray(self._bm25.get_scores(tokenize(query) or ["_"]))
                rankings.append(np.argsort(-bm))
            dense = None
            if self._emb is not None:
                emb = _Embedder.get(self.embed_model)
                q = emb.encode([query], query=True)[0]  # type: ignore[union-attr]
                dense = self._emb @ q
                rankings.append(np.argsort(-dense))
            # reciprocal rank fusion
            rrf = np.zeros(n)
            for order in rankings:
                for rank, idx in enumerate(order):
                    rrf[idx] += 1.0 / (60 + rank)
            top = np.argsort(-rrf)[: min(k, n)]
            return [(self.chunks[i], float(rrf[i])) for i in top]

    def all_text(self) -> str:
        return "\n".join(c.text for c in self.chunks)

    def summary(self, max_chars: int = 300) -> list[dict[str, Any]]:
        return [
            {
                "evidence_id": e.id,
                "tool": e.source,
                "args": e.args,
                "chars": len(e.text),
                "preview": e.text[:max_chars],
            }
            for e in self.evidence.values()
        ]
