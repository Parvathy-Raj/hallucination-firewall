"""Claude Code hooks.

An MCP server cannot force the agent to call verify_answer. These hooks close
that gap inside Claude Code:

  hfw-hook capture   PostToolUse: append every tool output to the evidence ledger,
                     so evidence from built-in tools (Read, WebFetch, Bash) and other
                     MCP servers is checked too.
  hfw-hook stop      Stop: if the agent used tools this turn but did not call
                     verify_answer after its last tool call, block the reply and tell
                     it to verify first.

See examples/claude_code/settings.json for the wiring.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from hfw.ledger import Ledger

# Tools whose output is not evidence about the world.
SKIP_TOOLS = re.compile(
    os.environ.get(
        "HFW_HOOK_SKIP",
        r"^(Write|Edit|MultiEdit|NotebookEdit|TodoWrite|Task|Agent|ExitPlanMode|AskUserQuestion|ToolSearch|Skill)$"
        r"|^mcp__(hfw|hallucination[-_]firewall)__",
    )
)
VERIFY_TOOL = re.compile(r"__(verify_answer|rewrite_grounded)$")
FIREWALL_TOOL = re.compile(r"^mcp__(hfw|hallucination[-_]firewall)__")


def _ledger_path(cwd: str) -> Path:
    p = Path(os.environ.get("HFW_LEDGER", ".hfw/ledger.jsonl"))
    return p if p.is_absolute() else Path(cwd) / p


def _response_text(resp: Any) -> Any:
    if isinstance(resp, dict):
        # Read tool: {"file": {"content": ...}}; Bash: {"stdout": ..., "stderr": ...}
        if isinstance(resp.get("file"), dict) and "content" in resp["file"]:
            return resp["file"]["content"]
        if "stdout" in resp:
            return resp.get("stdout", "")
        if "content" in resp and isinstance(resp["content"], (str, list)):
            return resp["content"]
        if "result" in resp:
            return resp["result"]
    return resp


def capture(payload: dict[str, Any]) -> int:
    tool = payload.get("tool_name", "")
    if not tool or SKIP_TOOLS.search(tool):
        return 0
    out = _response_text(payload.get("tool_response"))
    if out in (None, "", [], {}):
        return 0
    ledger = Ledger(_ledger_path(payload.get("cwd") or os.getcwd()))
    args = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    ledger.add(tool, out, args, session=payload.get("session_id", ""))
    return 0


def _turn_tool_calls(transcript: Path) -> list[str]:
    """Tool names the assistant called since the last real user message."""
    calls: list[str] = []
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = entry.get("message") or {}
        content = msg.get("content")
        if entry.get("type") == "user":
            is_tool_result = isinstance(content, list) and any(
                isinstance(c, dict) and c.get("type") == "tool_result" for c in content
            )
            if not is_tool_result and not entry.get("isMeta"):
                calls = []  # new user turn
        elif entry.get("type") == "assistant" and isinstance(content, list):
            calls.extend(c.get("name", "") for c in content if isinstance(c, dict) and c.get("type") == "tool_use")
    return calls


def stop(payload: dict[str, Any]) -> int:
    if payload.get("stop_hook_active"):
        return 0  # we already blocked once this turn; do not loop
    tp = payload.get("transcript_path")
    if not tp or not Path(tp).exists():
        return 0
    calls = _turn_tool_calls(Path(tp))
    evidence_idx = [i for i, n in enumerate(calls) if not SKIP_TOOLS.search(n) and not FIREWALL_TOOL.search(n)]
    # proxied tools are served by the firewall itself but are evidence calls too
    evidence_idx += [i for i, n in enumerate(calls) if FIREWALL_TOOL.search(n) and n.count("__") >= 3]
    if not evidence_idx:
        return 0
    last_evidence = max(evidence_idx)
    verified_after = any(VERIFY_TOOL.search(n) for n in calls[last_evidence + 1 :])
    if verified_after:
        return 0
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": (
                    "Hallucination Firewall: you used tools this turn but did not verify your answer. "
                    "Call verify_answer with your full draft answer, fix anything it flags, then reply."
                ),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] not in ("capture", "stop"):
        print("usage: hfw-hook capture|stop  (reads the hook JSON on stdin)", file=sys.stderr)
        return 2
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    try:
        return capture(payload) if argv[0] == "capture" else stop(payload)
    except Exception as e:  # a hook must never break the agent
        print(f"[hfw-hook] {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
