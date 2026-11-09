"""Settings, read from environment variables (all prefixed HFW_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # LLM used for claim extraction and the tier-2 judge.
    # provider: auto | anthropic | openai | none
    #   "openai" means any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, vLLM...).
    llm_provider: str = field(default_factory=lambda: _env("HFW_LLM_PROVIDER", "auto"))
    llm_model: str = field(default_factory=lambda: _env("HFW_LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: _env("HFW_LLM_BASE_URL", ""))
    llm_api_key_env: str = field(default_factory=lambda: _env("HFW_LLM_API_KEY_ENV", ""))
    llm_rpm: float = field(default_factory=lambda: _env_float("HFW_LLM_RPM", 0))
    # Claim extraction: llm | heuristic | auto (llm when one is configured)
    extractor: str = field(default_factory=lambda: _env("HFW_EXTRACTOR", "auto"))

    # Tier-1 model: any Hugging Face NLI sequence classifier with entailment/contradiction
    # labels. The default was trained on MNLI + FEVER + ANLI, so it suits fact checking.
    nli_model: str = field(
        default_factory=lambda: _env(
            "HFW_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
    )
    # Dense retrieval model ("" disables embeddings; BM25 only).
    embed_model: str = field(
        default_factory=lambda: _env("HFW_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # Where tool outputs are recorded. The Claude Code hook writes to the same file.
    ledger_path: Path = field(
        default_factory=lambda: Path(_env("HFW_LEDGER", ".hfw/ledger.jsonl"))
    )
    # Downstream MCP servers to proxy (Claude-style {"mcpServers": {...}} JSON file).
    proxy_config: str = field(default_factory=lambda: _env("HFW_PROXY_CONFIG", ""))
    # Where verify_answer records that it ran (read by the Stop hook).
    state_dir: Path = field(default_factory=lambda: Path(_env("HFW_STATE_DIR", ".hfw")))


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_env() -> None:
    """Load HFW_DOTENV (if set) and ./.env. Lets you point at an existing .env elsewhere."""
    if os.environ.get("HFW_DOTENV"):
        load_dotenv(os.environ["HFW_DOTENV"])
    load_dotenv(".env")
