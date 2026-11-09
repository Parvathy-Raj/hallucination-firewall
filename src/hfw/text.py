"""Text helpers: sentence splitting, tokenizing, flattening JSON tool outputs."""

from __future__ import annotations

import json
import re
from typing import Any

_ABBREV = {
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "inc", "ltd", "co", "no", "art",
    "sec", "fig", "approx", "u.s", "u.k", "st", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec", "para", "cl", "p", "pp", "llc",
}

_SENT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9•\-*])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Bullets and line breaks also end a sentence."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)|\n(?=#)", text):
        block = " ".join(block.split())
        if not block:
            continue
        parts = _SENT_RE.split(block)
        buf = ""
        for part in parts:
            buf = f"{buf} {part}".strip() if buf else part
            last_word = buf.rstrip(".!?\"')]").split(" ")[-1].lower().rstrip(".")
            if last_word in _ABBREV or re.fullmatch(r"[a-z]", last_word):
                continue
            out.append(buf)
            buf = ""
        if buf:
            out.append(buf)
    return [s.strip() for s in out if s.strip()]


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOPWORDS = set(
    ["a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "with", "by", "as", "at", "from", "into", "than", "then", "so", "such", "can", "could", "should", "would", "will", "may", "might", "must", "not", "no", "do", "does", "did", "has", "have", "had", "he", "she", "they", "them", "their", "there", "here", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "each", "other", "some", "more", "most", "also", "only", "very", "just", "i", "you", "we", "our", "your", "his", "her", "if", "about", "over", "under", "after", "before", "between", "within", "per"]
)


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def flatten_json(obj: Any, prefix: str = "") -> list[str]:
    """Turn a JSON value into 'path: value' lines so numbers keep their field names."""
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                lines.extend(flatten_json(v, key))
            else:
                lines.append(f"{key}: {v}")
    elif isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            lines.append(f"{prefix}: {', '.join(map(str, obj))}" if prefix else ", ".join(map(str, obj)))
        else:
            for i, v in enumerate(obj):
                lines.extend(flatten_json(v, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {obj}" if prefix else str(obj))
    return lines


def normalize_tool_output(raw: Any) -> str:
    """Make a readable text from whatever a tool returned (str, JSON str, dict, MCP content)."""
    if isinstance(raw, str):
        s = raw.strip()
        if s[:1] in "{[":
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return raw
            return _from_parsed(parsed)
        return raw
    return _from_parsed(raw)


def _from_parsed(parsed: Any) -> str:
    # MCP-style content lists: [{"type": "text", "text": "..."}]
    if isinstance(parsed, list) and parsed and all(
        isinstance(p, dict) and p.get("type") == "text" for p in parsed
    ):
        return "\n".join(normalize_tool_output(p.get("text", "")) for p in parsed)
    if isinstance(parsed, dict) and set(parsed) <= {"content", "isError", "structuredContent"}:
        return normalize_tool_output(parsed.get("content", ""))
    # Long text fields are kept as prose; everything else becomes path: value lines.
    if isinstance(parsed, (dict, list)):
        return "\n".join(flatten_json(parsed))
    return str(parsed)
