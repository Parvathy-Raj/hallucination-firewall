import json
import sys
from pathlib import Path

from mcp import Client
from mcp.types import TextContent

from hfw import Firewall, Policy
from hfw.config import Settings
from hfw.hooks import capture, stop
from hfw.server import build_server

ROOT = Path(__file__).resolve().parents[1]


def _payload(result):
    if result.structured_content:
        sc = result.structured_content
        return sc.get("result", sc)
    return json.loads(next(c.text for c in result.content if isinstance(c, TextContent)))


async def test_server_tools(tmp_path, monkeypatch, evidence):
    monkeypatch.setenv("HFW_STATE_DIR", str(tmp_path))
    fw = Firewall(Settings(), Policy.preset("balanced", tiers=[0]))
    server = build_server(fw, proxy_config="")
    async with Client(server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"verify_answer", "verify_claim", "explain_verdict", "rewrite_grounded", "set_policy"} <= names
        await c.call_tool("record_evidence", {"source": "docs", "content": evidence})
        r = _payload(await c.call_tool("verify_answer", {"draft": "The registration threshold is AED 500,000."}))
        assert r["blocked"] and r["claims"][0]["verdict"] == "contradicted"
        assert r["summary"][0].startswith("❌")
        ex = _payload(await c.call_tool("explain_verdict", {"claim_id": r["claims"][0]["claim_id"]}))
        assert ex["evidence"]
        rw = _payload(await c.call_tool("rewrite_grounded", {"draft": "The registration threshold is AED 500,000."}))
        assert "AED 375,000" in rw["rewritten_answer"]
        pol = _payload(await c.call_tool("set_policy", {"strictness": "strict"}))
        assert pol["policy"]["block_on"] == ["contradicted", "unsupported"]
        res = await c.read_resource("report://session/current")
        assert "session report" in res.contents[0].text


async def test_proxy_mode_records_downstream_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("HFW_STATE_DIR", str(tmp_path))
    cfg = tmp_path / "downstream.json"
    cfg.write_text(json.dumps({"mcpServers": {"uae_docs": {
        "command": sys.executable,
        "args": [str(ROOT / "examples" / "uae_docs_server.py")],
        "env": {"PYTHONPATH": str(ROOT / "src")},
    }}}))
    fw = Firewall(Settings(), Policy.preset("balanced", tiers=[0]))
    server = build_server(fw, proxy_config=str(cfg))
    async with Client(server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert "uae_docs__search_guides" in names
        out = await c.call_tool("uae_docs__search_guides", {"query": "voluntary registration threshold"})
        assert "187,500" in out.content[0].text
        assert len(fw.ledger) == 1
        r = _payload(await c.call_tool("verify_answer", {"draft": "The voluntary registration threshold is AED 185,000."}))
        assert r["claims"][0]["verdict"] == "contradicted"
        assert r["claims"][0]["correction"]["evidence_says"] == "AED 187,500"


def test_capture_hook_writes_ledger(tmp_path):
    capture({"tool_name": "WebFetch", "cwd": str(tmp_path), "session_id": "s1",
             "tool_input": {"url": "https://x"}, "tool_response": "Q3 revenue was $4.2M."})
    capture({"tool_name": "Write", "cwd": str(tmp_path), "tool_response": "ignored"})
    capture({"tool_name": "mcp__hfw__verify_answer", "cwd": str(tmp_path), "tool_response": "ignored"})
    lines = (tmp_path / ".hfw" / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["source"] == "WebFetch"


def _transcript(tmp_path, tools):
    p = tmp_path / "t.jsonl"
    rows = [{"type": "user", "message": {"role": "user", "content": "question"}}]
    for name in tools:
        rows.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "id": "x", "input": {}}]}})
        rows.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_stop_hook_blocks_unverified_answer(tmp_path, capsys):
    stop({"transcript_path": _transcript(tmp_path, ["mcp__uae_docs__search_guides"])})
    assert json.loads(capsys.readouterr().out)["decision"] == "block"


def test_stop_hook_allows_verified_answer(tmp_path, capsys):
    stop({"transcript_path": _transcript(tmp_path, ["mcp__hfw__uae_docs__search_guides", "mcp__hfw__verify_answer"])})
    assert capsys.readouterr().out == ""


def test_stop_hook_allows_no_tool_turn_and_no_loop(tmp_path, capsys):
    stop({"transcript_path": _transcript(tmp_path, [])})
    stop({"transcript_path": _transcript(tmp_path, ["WebFetch"]), "stop_hook_active": True})
    assert capsys.readouterr().out == ""


def test_server_ledger_is_shared_with_hook(tmp_path, monkeypatch):
    """The server must persist to the same file the Claude Code hook appends to."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HFW_STATE_DIR", str(tmp_path / ".hfw"))
    server = build_server(proxy_config="")
    fw = server.firewall
    fw.policy.update(tiers=[0])
    capture({"tool_name": "WebFetch", "cwd": str(tmp_path), "tool_response": "Q3 revenue was $4.2M."})
    assert fw.verify("Q3 revenue was $4.7M.").blocked  # evidence from the hook is picked up
    fw.record("manual", "Headcount is 140.")
    assert (tmp_path / ".hfw" / "ledger.jsonl").read_text().count("\n") == 2
