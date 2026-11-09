"""Command line: `hfw serve` runs the MCP server; `hfw verify` checks a draft offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hfw.config import load_env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hfw", description="Hallucination Firewall")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the MCP server over stdio")
    s.add_argument("--proxy", default=None, help="downstream MCP servers JSON (proxy mode)")

    v = sub.add_parser("verify", help="verify a draft against evidence files")
    v.add_argument("--evidence", nargs="+", required=True, help="text/JSON files with tool outputs")
    v.add_argument("--draft", required=True, help="draft answer file, or - for stdin")
    v.add_argument("--strictness", default="balanced")
    v.add_argument("--tiers", default="0,1,2")
    v.add_argument("--rewrite", action="store_true")
    v.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()

    if args.cmd == "serve":
        from hfw.server import build_server

        build_server(proxy_config=args.proxy).run("stdio")
        return 0

    from hfw import Firewall, Policy

    pol = Policy.preset(args.strictness, tiers=[int(t) for t in args.tiers.split(",")])
    if args.rewrite:
        pol.action = "rewrite"
    fw = Firewall(policy=pol)
    for f in args.evidence:
        fw.record(Path(f).name, Path(f).read_text(encoding="utf-8"))
    draft = sys.stdin.read() if args.draft == "-" else Path(args.draft).read_text(encoding="utf-8")
    report = fw.verify(draft)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        for vd in report.verdicts:
            print(f"{vd.label.icon} [{vd.label.value:13}] tier {vd.tier:>2}  {vd.claim.text}")
            if vd.label.is_flag:
                print(f"      {vd.reason}")
        print(f"\n{report.action.upper()}: {report.message}")
        if report.rewritten:
            print("\n--- rewritten ---\n" + report.rewritten)
    return 1 if report.blocked else 0


if __name__ == "__main__":
    sys.exit(main())
