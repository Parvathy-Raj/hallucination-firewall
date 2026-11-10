# Hallucination Firewall skill

Makes your agent check its answer against the tool outputs it received before sending it.
Wrong numbers, dates and names get corrected, unsupported claims get removed or flagged, and
every remaining claim gets a citation.

## What it does

- Activates when an answer states facts taken from tools (search, documents, APIs, files, web
  pages), or when you ask it to fact-check a draft against sources.
- Splits the draft into claims and checks each one: ✅ supported, ❌ contradicted,
  ⚠️ unsupported, ➖ not checkable.
- Numbers, amounts, dates and names must match the evidence exactly, with rules for rounding
  and formats.
- Sends the corrected answer with citations and a short fact-check note.
- Ignores instructions found inside tool outputs.

## How it checks

1. **Hallucination Firewall MCP server (optional, best).** If you've installed it (see
   `references/setup.md`), the agent calls `verify_answer`, which checks claims in code
   against a ledger of tool outputs.
2. **Manual check (default).** Otherwise the agent follows `references/checking-rules.md`
   by hand.

## Configuration

None needed. No API keys, no scripts, nothing installed by the skill. Ask for "strict" or
"lenient" checking to change how unsupported claims are handled.

## Files

- `SKILL.md`: instructions for the agent
- `references/checking-rules.md`: claim splitting, exact-match rules, verdicts and fixes
- `references/setup.md`: optional MCP server install
