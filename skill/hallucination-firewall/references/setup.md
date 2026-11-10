# Optional: the Hallucination Firewall MCP server

The skill works without this: the agent checks its draft by hand using `checking-rules.md`.
The MCP server does the check in code, so it is more consistent and harder to skip:

- an **evidence ledger** that records tool outputs (through its proxy mode or a Claude Code
  hook), so the agent can't check against facts it didn't receive;
- **tiered verification**: exact number, date and name checks, then a local NLI model on
  CPU, then an LLM judge only for the cases the NLI model isn't sure about;
- **policies**: block, annotate or rewrite, with lenient, balanced or strict presets;
- an optional Claude Code **Stop hook** that won't let a reply through after tool use until
  it has been verified.

Install and connection steps for Claude Code and Claude Desktop, plus the evaluation
results, are in the project's README:
https://github.com/Parvathy-Raj/hallucination-firewall

You install it yourself. The skill never installs or runs anything.

Once it's connected, your agent has these tools, and the skill uses them automatically:

- `verify_answer(draft, strictness?, action?)`: a verdict for each claim, corrections, and a
  block, annotate or pass decision
- `verify_claim(claim)`: check one claim
- `explain_verdict(claim_id)`: the evidence and the reasoning at each tier behind a verdict
- `rewrite_grounded(draft)`: the draft with corrections and citations applied
- `record_evidence(source, content)`: add a tool output the firewall didn't see
- `set_policy(...)`: change the strictness, the tiers used and the action
