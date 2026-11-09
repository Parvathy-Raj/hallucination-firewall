"""MCP server exposing the Hallucination Firewall.

Run:  hfw serve            (stdio)
Env:  HFW_PROXY_CONFIG=path/to/downstream.json   to enable proxy mode
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import Tool as MCPTool

from hfw.config import Settings, load_env
from hfw.firewall import Firewall
from hfw.ledger import Ledger
from hfw.policy import Policy
from hfw.proxy import Proxy

INSTRUCTIONS = """Hallucination Firewall checks your final answer against the tool outputs you
actually received in this session.

REQUIRED WORKFLOW for any answer that states facts taken from tool results:
1. Gather evidence with your tools as usual. Tool outputs are recorded automatically
   (proxied tools, or the Claude Code hook). If a fact came from somewhere that was not
   recorded, call record_evidence with the source text first.
2. Write your draft answer, then call verify_answer(draft) BEFORE replying to the user.
3. If the result says blocked, do not send the draft. Fix every claim marked contradicted
   (use the correction / evidence given) and remove or hedge unsupported ones, or call
   rewrite_grounded(draft). Then verify again.
4. In your reply, mention when the firewall corrected something.
The firewall only checks claims against tool evidence, not against world knowledge."""


class FirewallServer(MCPServer):
    """MCPServer that also serves proxied downstream tools."""

    proxy: Proxy | None = None

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        return tools + (self.proxy.list() if self.proxy else [])

    def _tool_input_schema(self, name: str) -> dict[str, Any] | None:
        if self.proxy and self.proxy.owns(name):
            return self.proxy.schema(name)
        return super()._tool_input_schema(name)

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Any = None):  # type: ignore[override]
        if self.proxy and self.proxy.owns(name):
            return await self.proxy.call(name, arguments)
        return await super().call_tool(name, arguments, context)


def build_server(firewall: Firewall | None = None, proxy_config: str | None = None) -> FirewallServer:
    load_env()
    settings = Settings()
    fw = firewall or Firewall(
        settings,
        Policy.preset("balanced"),
        Ledger(settings.ledger_path, embed_model=settings.embed_model),
    )
    proxy_cfg = proxy_config if proxy_config is not None else settings.proxy_config

    @asynccontextmanager
    async def lifespan(server: FirewallServer):
        if proxy_cfg:
            server.proxy = Proxy(fw.ledger)
            await server.proxy.start(proxy_cfg)
        try:
            yield {}
        finally:
            if server.proxy:
                await server.proxy.stop()

    mcp = FirewallServer(
        "hallucination-firewall",
        instructions=INSTRUCTIONS,
        version="0.1.0",
        lifespan=lifespan,
    )
    mcp.firewall = fw  # type: ignore[attr-defined]

    def _summary(report_dict: dict[str, Any]) -> list[str]:
        lines = []
        for c in report_dict["claims"]:
            line = f'{c["icon"]} {c["claim_id"]} [{c["verdict"]}, tier {c["decided_by_tier"]}] {c["claim"]}'
            if c["verdict"] in ("contradicted", "unsupported"):
                line += f'  -> {c["reason"]}'
                if c["evidence"]:
                    line += f' (evidence: {c["evidence"][0]["tool"]})'
            lines.append(line)
        return lines

    @mcp.tool()
    def verify_answer(draft: str, strictness: str | None = None, action: str | None = None) -> dict[str, Any]:
        """Check every factual claim in a draft answer against recorded tool outputs.

        Call this before giving the user any answer based on tool results.
        Returns a verdict per claim (supported / contradicted / unsupported / not_checkable),
        the evidence behind it, a correction for wrong figures, and whether the draft is blocked.

        Args:
            draft: the full answer you intend to send.
            strictness: optional one-off override: lenient | balanced | strict.
            action: optional one-off override: annotate | block | rewrite.
        """
        policy = fw.policy
        if strictness or action:
            policy = Policy(**fw.policy.to_dict()).update(strictness=strictness, action=action)
        report = fw.verify(draft, policy).to_dict()
        report["summary"] = _summary(report)
        return report

    @mcp.tool()
    def verify_claim(claim: str) -> dict[str, Any]:
        """Check one standalone factual claim against recorded tool outputs."""
        return fw.verify_claim(claim).to_dict(max_evidence=3)

    @mcp.tool()
    def explain_verdict(claim_id: str) -> dict[str, Any]:
        """Show the full evidence passages and what each tier decided for a claim id (e.g. "r1.c2")."""
        return fw.explain(claim_id)

    @mcp.tool()
    def rewrite_grounded(draft: str, unsupported: str = "hedge") -> dict[str, Any]:
        """Return a corrected version of the draft: wrong figures replaced with the evidence
        values, unsupported claims hedged (or removed with unsupported="remove"), and citations
        to the tool outputs added."""
        text, report = fw.rewrite_grounded(draft, unsupported)
        d = report.to_dict()
        return {
            "rewritten_answer": text,
            "changes": [l for l in _summary(d) if not l.startswith("✅") and not l.startswith("➖")],
            "grounding_score_before": d["grounding_score"],
        }

    @mcp.tool()
    def set_policy(
        strictness: str | None = None,
        tiers: list[int] | None = None,
        block_on: list[str] | None = None,
        action: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Configure the firewall.

        Args:
            strictness: lenient | balanced | strict (presets for block_on and thresholds).
            tiers: which verifier tiers may run: 0 exact, 1 NLI model, 2 LLM judge.
            block_on: verdicts that block the answer, e.g. ["contradicted", "unsupported"].
            action: what to do with a flagged answer: annotate | block | rewrite.
            threshold: tier-1 confidence (0.5-1.0) needed to decide without the LLM judge.
        """
        fw.policy.update(strictness, tiers, block_on, action, threshold)
        return {"policy": fw.policy.to_dict(), "llm": fw.llm.provider, "nli_model": fw.settings.nli_model}

    @mcp.tool()
    def record_evidence(source: str, content: str) -> dict[str, Any]:
        """Add evidence by hand (e.g. text from a tool that is not proxied or hooked).
        Use the exact text the source returned, never your own summary."""
        ev = fw.ledger.add(source, content)
        return {"recorded": bool(ev), "evidence_id": ev.id if ev else None, "evidence_items": len(fw.ledger)}

    @mcp.tool()
    def reset_session() -> dict[str, Any]:
        """Clear the evidence ledger and verdict history (start of a new task)."""
        fw.ledger.clear()
        fw.history.clear()
        fw.reports.clear()
        return {"evidence_items": 0}

    @mcp.resource("ledger://session/current", mime_type="application/json")
    def ledger_resource() -> list[dict[str, Any]]:
        """Everything the tools returned this session."""
        fw.ledger.sync()
        return fw.ledger.summary()

    @mcp.resource("report://session/current", mime_type="text/markdown")
    def report_resource() -> str:
        """Hallucination report for this session."""
        return fw.session_report()

    return mcp


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
