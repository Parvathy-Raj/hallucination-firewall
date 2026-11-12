# Hallucination Firewall

**An MCP server that checks an AI agent's final answer against the tool outputs it actually received, and blocks or flags anything unsupported.**

Agents call tools, get correct data back, and still misstate it in the answer: a wrong figure, an invented detail, a claim no source supports. The Hallucination Firewall records everything the tools returned, splits the draft answer into single-fact claims, checks each one against that evidence with a tiered verifier, and returns a verdict per claim with the tool output that proves or disproves it.

https://github.com/Parvathy-Raj/hallucination-firewall/raw/main/docs/demo.mp4

*Demo: a draft client email about UAE VAT registration is checked against the FTA guides, searched through the firewall's proxy. The firewall blocks two wrong figures ❌ (AED 500,000 should be AED 375,000; 60 days should be 30 days), confirms the two correct claims ✅, and rewrites the email with a citation to the guide passage behind each line.*

```
┌──────────┐     ┌────────────────────────────┐     ┌──────────────────┐
│  Agent   │────▶│   Hallucination Firewall   │────▶│ Downstream MCP   │
│ (Claude) │◀────│   (MCP server + client)    │◀────│ servers (DB, RAG,│
└──────────┘     │                            │     │ web, files...)   │
                 │  1. Evidence ledger        │     └──────────────────┘
                 │  2. Claim extractor        │
                 │  3. Evidence retriever     │
                 │  4. Verifier (tiered)      │
                 │  5. Verdict + rewrite      │
                 └────────────────────────────┘
```

## How it works

1. **Evidence ledger.** Every tool output is recorded, chunked and indexed with hybrid retrieval (BM25 + `bge-small` embeddings, merged with reciprocal rank fusion). Outputs get there three ways:
   - **Proxy mode**: the firewall wraps your other MCP servers and re-exposes their tools (`uae_docs__search_guides`), recording every result.
   - **Claude Code hook**: a `PostToolUse` hook records the output of *any* tool (Read, WebFetch, Bash, other MCP servers).
   - **By hand / SDK**: `record_evidence(source, content)`, or `fw.record()` / `@fw.watch` in Python.
2. **Claim extraction.** An LLM splits the draft into atomic claims and labels each one *factual*, *opinion* or *hedge*. Only factual claims are checked. (A rule-based extractor is used when no LLM is configured.)
3. **Tiered verification**, cheapest first. Each tier either decides or passes the claim on:

   | Tier | Method | Cost | Catches |
   |---|---|---|---|
   | 0 | Exact checks: numbers, currency, percentages, dates, names, with rounding-aware matching | ~0 | `$4.2M` vs `$4.7M`, `AED 500,000` vs `AED 375,000` |
   | 1 | NLI fact-checking model on CPU (`DeBERTa-v3-base-mnli-fever-anli`) | ~0.15 s per evidence passage | Most clearly supported or contradicted claims |
   | 2 | LLM judge, only when tier 1 is below the confidence threshold | $ | Paraphrases, multi-hop, "the evidence doesn't say that" |

4. **Verdict + policy.** Each claim is ✅ `supported`, ❌ `contradicted`, ⚠️ `unsupported` or ➖ `not_checkable`. The policy decides whether to **block**, **annotate** or **rewrite**. The rewrite is deterministic on purpose (an LLM rewriter could add hallucinations of its own): wrong figures are replaced with the evidence value, unsupported claims are hedged or removed, and citations to the tool outputs are added.

## The trust model (read this first)

An MCP server cannot force an agent to call `verify_answer`. This project closes that gap in three layers, from weakest to strongest:

| Layer | How | Guarantee |
|---|---|---|
| Server instructions | The server tells the agent to verify every answer built on tool results | Best effort: the model usually complies |
| Claude Code `Stop` hook | `hfw-hook stop` blocks the reply if the agent used tools this turn but did not call `verify_answer` afterwards | Enforced by the harness, inside Claude Code |
| Python SDK | `fw.guard(draft)` raises `HallucinationBlocked`; your code decides what ships | Enforced in your own code |

What it does **not** do: check claims against world knowledge. A claim that no tool output supports is *unsupported*, even if it happens to be true. That is the point: in an agent, an answer should be traceable to what the tools returned.

## MCP interface

**Tools**

| Tool | What it does |
|---|---|
| `verify_answer(draft, strictness?, action?)` | Verdict, confidence, deciding tier, evidence and correction for every claim, plus an overall grounding score and a block decision |
| `verify_claim(claim)` | Checks one claim |
| `explain_verdict(claim_id)` | Full evidence passages and what each tier said |
| `rewrite_grounded(draft, unsupported="hedge")` | Corrected answer with citations |
| `set_policy(strictness, tiers, block_on, action, threshold)` | `lenient` / `balanced` / `strict`, which tiers run, what blocks, the tier-1 confidence threshold |
| `record_evidence(source, content)` | Add evidence that was not proxied or hooked |
| `reset_session()` | Clear the ledger for a new task |

**Resources**: `ledger://session/current` (everything the tools returned) and `report://session/current` (hallucination report for the session).

## Quickstart

```bash
git clone https://github.com/Parvathy-Raj/hallucination-firewall
cd hallucination-firewall
uv sync --extra ml          # tiers 1 and dense retrieval (CPU PyTorch); plain `uv sync` = tier 0 + BM25 only
uv run pytest -q
```

Check a draft from the command line:

```bash
uv run hfw verify --evidence report.txt --draft draft.txt --rewrite
```

### Use it in Claude Code

1. Copy [`examples/claude_code/.mcp.json`](examples/claude_code/.mcp.json) into your project and fix the paths. It runs the firewall in proxy mode in front of the example UAE tax-docs server ([`downstream.json`](examples/claude_code/downstream.json)); put your own MCP servers there.
2. Copy the hooks from [`examples/claude_code/settings.json`](examples/claude_code/settings.json) into `.claude/settings.json`. `capture` records every tool output; `stop` makes Claude verify before it answers.
3. Set an LLM for claim extraction and the tier-2 judge (below). Without one, tiers 0 and 1 still run.

Claude Desktop: see [`examples/claude_desktop/claude_desktop_config.json`](examples/claude_desktop/claude_desktop_config.json).

### Use it from Python (enforced in code)

```python
from hfw import Firewall, HallucinationBlocked

fw = Firewall()

@fw.watch                      # every return value becomes evidence
def get_invoice(invoice_id: str) -> dict: ...

get_invoice(invoice_id="INV-1042")
try:
    answer = fw.guard(draft)   # raises if the policy blocks
except HallucinationBlocked as e:
    answer, _ = fw.rewrite_grounded(draft)
```

Full example: [`examples/sdk_example.py`](examples/sdk_example.py).

### Use it as an agent skill (no install)

[`skill/hallucination-firewall/`](skill/hallucination-firewall/) is an agent skill with the
same checking rules written out for the agent. If the MCP tools are connected, the agent calls
`verify_answer`. If not, it checks the draft against its tool outputs by hand: exact
numbers, dates and names, then meaning. It fixes or removes flagged claims and cites the
rest. Copy the folder into `.claude/skills/`, or build the zip with
`python scripts/build_skill.py`.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HFW_LLM_PROVIDER` | `auto` | `anthropic`, `groq`, `openai` (any OpenAI-compatible endpoint), or `none`. `auto` picks Anthropic if `ANTHROPIC_API_KEY` is set, then Groq |
| `HFW_LLM_MODEL` | `claude-haiku-4-5` / `openai/gpt-oss-20b` | Model for claim extraction and the judge |
| `HFW_LLM_BASE_URL` | provider default | e.g. `http://localhost:11434/v1` for Ollama |
| `HFW_NLI_MODEL` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Any HF NLI classifier; `cross-encoder/nli-deberta-v3-xsmall` is ~3x faster |
| `HFW_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Empty string = BM25 only |
| `HFW_LEDGER` | `.hfw/ledger.jsonl` | Shared by the server and the Claude Code hook |
| `HFW_PROXY_CONFIG` | | Downstream servers JSON (same as `--proxy`) |
| `HFW_DOTENV` | | Extra `.env` file to load |

## Evaluation

Three benchmarks, all scored at the claim level (one claim per answer sentence, so human span labels map onto claims directly and every configuration sees identical claims):

- **RAGTruth** (main benchmark): RAG answers (QA, summarization, data-to-text) with human-labelled hallucinated spans. Test split, sampled.
- **HaluEval QA**: each question once with the right answer and once with a hallucinated one.
- **UAE agent traces** (custom): 20 simulated agent turns built from the FTA guide chunks of my [UAE regulatory RAG](https://github.com/Parvathy-Raj/uae-regulatory-rag). The tool returns the gold passage plus two distractors; the draft has one correct sentence and three planted errors (changed number, negated obligation, claim from a guide that was never retrieved). Rebuild with `make traces UAE_CHUNKS=...`; the corpus itself is not redistributed.

Metrics: precision / recall / F1 for catching hallucinated claims; **false-flag rate** (correct claims wrongly flagged); **false-block rate** (fully correct answers that the balanced policy would block, which is what users notice); latency p50/p95 per answer; LLM cost per answer.

`make eval` regenerates everything below. Every NLI score and LLM call is cached, so the configurations share identical judge calls and re-runs are free; cost and latency still charge each configuration for every call it makes.

<!-- RESULTS:START -->
*Run `make eval` to fill this in.*
<!-- RESULTS:END -->

### Reading the results

The numbers here come from a small sample (rate limits on the free Groq tier I used for the judge), so treat differences of a few points as noise. The judge in these runs is `gpt-oss-20b` via Groq, not Claude; set `HFW_EVAL_MODEL` / `HFW_EVAL_BASE_URL` to compare others.

## Repo layout

```
src/hfw/
├── server.py        MCP server + tools + resources
├── proxy.py         downstream MCP clients, ledger capture
├── ledger.py        evidence store, chunking, hybrid index
├── claims.py        claim extraction (LLM or rules)
├── verify/
│   ├── exact.py     tier 0
│   ├── nli.py       tier 1
│   ├── judge.py     tier 2
│   └── pipeline.py  confidence routing
├── policy.py        strictness presets, block / annotate / rewrite
├── rewrite.py       deterministic grounded rewrite with citations
├── firewall.py      facade used by the server, CLI and SDK
├── hooks.py         Claude Code PostToolUse + Stop hooks
└── cli.py           `hfw serve`, `hfw verify`
evals/               datasets, runner, custom traces, results/
examples/            Claude Code / Desktop configs, demo server, SDK example
skill/               agent skill (manual checking rules + MCP usage)
scripts/             build_skill.py (skill zip)
docs/demo/           demo video renderer
tests/               pytest (offline, no models needed)
```

## Likely questions

| Objection | Answer |
|---|---|
| "The agent can just skip verification." | The Stop hook or the SDK enforces it; the trust model above is explicit about what each layer guarantees. |
| "The checker makes mistakes too." | The false-flag and false-block rates are measured, and the tier-1 threshold and `block_on` let you trade recall for fewer false blocks. Unsupported claims are annotated, not blocked, in the default policy. |
| "Too slow." | Tier 0 is microseconds and tier 1 runs on CPU; the LLM judge only sees the claims the cheap tiers could not settle. |
| "What about facts not from tools?" | Out of scope by design: it checks claims against tool evidence, not world knowledge. |

## Roadmap

- MiniCheck / Vectara HHEM as alternative tier-1 models, and a FEVER run
- Claim-to-span alignment so rewrites can edit inside long sentences
- OpenTelemetry spans per tier
- PyPI release

## License

MIT
