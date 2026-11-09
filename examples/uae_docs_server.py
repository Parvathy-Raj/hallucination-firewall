"""Example downstream MCP server: keyword search over UAE tax guide passages.

Used for the demo. It searches the chunk file produced by the UAE regulatory RAG
project (set UAE_DOCS_CHUNKS=/path/to/chunks.jsonl). Without it, it falls back to
examples/sample_docs.jsonl: a few short passages paraphrased from the public
UAE Federal Tax Authority guides.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from rank_bm25 import BM25Okapi

from hfw.text import tokenize

HERE = Path(__file__).parent
path = Path(os.environ.get("UAE_DOCS_CHUNKS") or HERE / "sample_docs.jsonl")
DOCS = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
INDEX = BM25Okapi([tokenize(f'{d.get("heading", "")} {d["text"]}') for d in DOCS])

mcp = MCPServer("uae-tax-docs", instructions="Search UAE Federal Tax Authority guidance (VAT, corporate tax).")


@mcp.tool()
def search_guides(query: str, top_k: int = 3) -> str:
    """Search UAE FTA tax guides. Returns the most relevant passages with their source and page."""
    scores = INDEX.get_scores(tokenize(query))
    top = sorted(range(len(DOCS)), key=lambda i: -scores[i])[: max(1, min(top_k, 8))]
    out = []
    for rank, i in enumerate(top, 1):
        d = DOCS[i]
        src = d.get("source_title", d.get("source", "FTA guide"))
        page = f', p. {d["page_start"]}' if d.get("page_start") else ""
        out.append(f'[{rank}] {src}{page} - {d.get("heading", "")}\n{d["text"]}')
    return "\n\n".join(out)


if __name__ == "__main__":
    mcp.run("stdio")
