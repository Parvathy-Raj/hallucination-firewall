"""Proxy mode: connect to downstream MCP servers, re-expose their tools, record every output.

Config uses the same shape as Claude Desktop / Claude Code:

    {"mcpServers": {"uae_docs": {"command": "python", "args": ["server.py"], "env": {}}}}

A downstream tool `search` on server `uae_docs` is exposed as `uae_docs__search`.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, TextContent, Tool

from hfw.ledger import Ledger

SEP = "__"


@dataclass
class ProxiedTool:
    server: str
    name: str  # downstream name
    tool: Tool  # as exposed upstream (prefixed name)
    client: Client


def load_config(path: str) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    servers = data.get("mcpServers", data)
    return {k: v for k, v in servers.items() if isinstance(v, dict) and v.get("command")}


def _expand(value: str) -> str:
    return os.path.expandvars(value)


class Proxy:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.tools: dict[str, ProxiedTool] = {}
        self.errors: dict[str, str] = {}
        self._stack = AsyncExitStack()

    async def start(self, config_path: str) -> None:
        if not config_path:
            return
        for name, spec in load_config(config_path).items():
            params = StdioServerParameters(
                command=_expand(spec["command"]),
                args=[_expand(a) for a in spec.get("args", [])],
                env={k: _expand(v) for k, v in (spec.get("env") or {}).items()} or None,
                cwd=spec.get("cwd"),
            )
            try:
                client = await self._stack.enter_async_context(Client(params))
                listed = await client.list_tools()
            except Exception as e:  # keep the firewall up even if one server fails
                self.errors[name] = repr(e)
                print(f"[hfw] proxy: could not start {name!r}: {e}", file=sys.stderr)
                continue
            for t in listed.tools:
                upstream = t.model_copy(
                    update={
                        "name": f"{name}{SEP}{t.name}",
                        "description": f"[via Hallucination Firewall: output is recorded as evidence] {t.description or ''}",
                    }
                )
                self.tools[upstream.name] = ProxiedTool(name, t.name, upstream, client)

    async def stop(self) -> None:
        await self._stack.aclose()

    def owns(self, name: str) -> bool:
        return name in self.tools

    def list(self) -> list[Tool]:
        return [p.tool for p in self.tools.values()]

    def schema(self, name: str) -> dict[str, Any] | None:
        p = self.tools.get(name)
        return p.tool.input_schema if p else None

    async def call(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        p = self.tools[name]
        result = await p.client.call_tool(p.name, arguments)
        if isinstance(result, CallToolResult) and not result.is_error:
            texts = [c.text for c in result.content if isinstance(c, TextContent)]
            payload: Any = "\n".join(texts)
            if not payload.strip() and result.structured_content:
                payload = result.structured_content
            self.ledger.add(f"{p.server}.{p.name}", payload, arguments)
        return result  # type: ignore[return-value]
