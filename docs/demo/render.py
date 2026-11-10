"""Render a Claude Code session log (stream-json) into a terminal-style demo video.

The video replays a real session: every tool call, tool result and answer comes
from the log; nothing is typed in by hand. Pacing and captions are added.

    claude -p "<prompt>" --output-format stream-json --verbose > session.jsonl
    python docs/demo/render.py session.jsonl --prompt "<prompt>" --out docs/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 768, 30
FONTS = Path("C:/Windows/Fonts")
MONO = ImageFont.truetype(str(FONTS / "CascadiaMono.ttf"), 16)
MONO_B = ImageFont.truetype(str(FONTS / "CascadiaMono.ttf"), 16)
SYM = ImageFont.truetype(str(FONTS / "seguisym.ttf"), 15)
UI = ImageFont.truetype(str(FONTS / "segoeui.ttf"), 19)
UI_B = ImageFont.truetype(str(FONTS / "segoeuib.ttf"), 19)
TITLE = ImageFont.truetype(str(FONTS / "segoeuib.ttf"), 44)
SUB = ImageFont.truetype(str(FONTS / "segoeui.ttf"), 22)

BG, CHROME, FG, DIM = "#0d1117", "#161b22", "#e6edf3", "#7d8590"
GREEN, RED, AMBER, BLUE, ORANGE = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#d18616"
SYMBOLS = set("⏺⎿✔✖⚠●▸›–")
CHAR_W = MONO.getbbox("M")[2]
LINE_H = 22
LEFT, TOP = 28, 58
TEXT_BOTTOM = H - 84
COLS = (W - 2 * LEFT) // CHAR_W
ROWS = (TEXT_BOTTOM - TOP) // LINE_H

ICON_MAP = {"✅": ("✔", GREEN), "❌": ("✖", RED), "⚠️": ("⚠", AMBER), "⚠": ("⚠", AMBER), "➖": ("–", DIM)}

Span = tuple[str, str]  # (text, color)


@dataclass
class Screen:
    lines: list[list[Span]] = field(default_factory=list)
    revealed: int = 0  # characters revealed among lines[pending_from:]
    pending_from: int = 0
    caption: str = ""
    step: str = ""


def wrap(spans: list[Span], indent: int = 0, first_prefix: list[Span] | None = None) -> list[list[Span]]:
    """Wrap a list of spans to the terminal width, keeping colors."""
    prefix = first_prefix or []
    plen = sum(len(t) for t, _ in prefix)
    width = COLS - max(plen, indent)
    words: list[Span] = []
    for text, color in spans:
        for part in re.split(r"(\s+)", text):
            if part:
                words.append((part, color))
    out: list[list[Span]] = []
    cur: list[Span] = list(prefix)
    cur_len = 0
    for text, color in words:
        if text.isspace():
            if cur_len and cur_len + 1 <= width:
                cur.append((" ", color))
                cur_len += 1
            continue
        while len(text) > width:  # very long token
            if cur_len:
                out.append(cur)
                cur, cur_len = [(" " * indent, FG)], 0
            out.append(cur + [(text[:width], color)])
            text = text[width:]
            cur, cur_len = [(" " * indent, FG)], 0
        if cur_len + len(text) > width:
            while cur and cur[-1][0] == " ":
                cur.pop()
            out.append(cur)
            cur, cur_len = [(" " * indent, FG)], 0
        cur.append((text, color))
        cur_len += len(text)
    out.append(cur)
    return out


def draw_spans(d: ImageDraw.ImageDraw, x: int, y: int, spans: list[Span], limit: int | None) -> int:
    shown = 0
    for text, color in spans:
        for ch in text:
            if limit is not None and shown >= limit:
                return shown
            if ch in SYMBOLS:
                d.text((x, y + 2), ch, font=SYM, fill=color)
            elif ch != " ":
                d.text((x, y), ch, font=MONO, fill=color)
            x += CHAR_W
            shown += 1
    return shown


def frame(s: Screen) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 38], fill=CHROME)
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
    d.text((W // 2, 19), "claude  ·  demo-ws  ·  MCP: hfw (Hallucination Firewall)", font=UI, fill=DIM, anchor="mm")
    visible = s.lines[-ROWS:]
    start_idx = len(s.lines) - len(visible)
    budget = s.revealed
    for i, spans in enumerate(visible):
        idx = start_idx + i
        y = TOP + i * LINE_H
        if idx < s.pending_from:
            draw_spans(d, LEFT, y, spans, None)
        else:
            if budget <= 0:
                break
            budget -= draw_spans(d, LEFT, y, spans, budget)
    # caption bar
    d.rectangle([0, H - 64, W, H], fill=CHROME)
    if s.step:
        d.rounded_rectangle([20, H - 48, 20 + UI_B.getlength(s.step) + 24, H - 16], radius=8, fill=ORANGE)
        d.text((32, H - 32), s.step, font=UI_B, fill="#0d1117", anchor="lm")
        x = 20 + UI_B.getlength(s.step) + 40
    else:
        x = 24
    d.text((x, H - 32), s.caption, font=UI, fill=FG, anchor="lm")
    return img


class Video:
    def __init__(self, out: Path):
        self.writer = imageio_ffmpeg.write_frames(
            str(out), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
            output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"],
        )
        self.writer.send(None)
        self.frames = 0

    def emit(self, img: Image.Image, n: int = 1) -> None:
        buf = np.asarray(img).tobytes()
        for _ in range(n):
            self.writer.send(buf)
            self.frames += 1

    def close(self) -> None:
        self.writer.close()


class Director:
    def __init__(self, video: Video):
        self.v = video
        self.s = Screen()

    def hold(self, seconds: float) -> None:
        self.v.emit(frame(self.s), max(1, int(seconds * FPS)))

    def say(self, step: str, caption: str) -> None:
        self.s.step, self.s.caption = step, caption

    def add(self, lines: list[list[Span]], cps: float = 400, max_seconds: float = 4.0) -> None:
        """Append lines and reveal them progressively at `cps` characters per second."""
        self.s.pending_from = len(self.s.lines)
        self.s.lines.extend(lines)
        total = sum(len(t) for line in lines for t, _ in line)
        seconds = min(max_seconds, total / cps) if total else 0
        n = max(1, int(seconds * FPS))
        for i in range(1, n + 1):
            self.s.revealed = int(total * i / n)
            self.v.emit(frame(self.s))
        self.s.pending_from = len(self.s.lines)
        self.s.revealed = 0

    def blank(self) -> None:
        self.s.lines.append([("", FG)])


def card(video: Video, title: str, subtitle: list[str], seconds: float) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 70), title, font=TITLE, fill=FG, anchor="mm")
    for i, line in enumerate(subtitle):
        d.text((W // 2, H // 2 - 5 + i * 36), line, font=SUB, fill=DIM if i else BLUE, anchor="mm")
    video.emit(img, int(seconds * FPS))


# ---- session log → screen ---------------------------------------------------------------
def tool_label(name: str) -> tuple[str, str]:
    m = re.match(r"mcp__([^_]+(?:_[^_]+)*?)__(.+)$", name)
    if m:
        return m.group(1), m.group(2)
    return "", name


def short_args(inp: dict) -> str:
    parts = []
    for k, v in inp.items():
        v = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else f'"{v}"'
        v = " ".join(v.split())
        tail = '…"' if len(v) > 70 else ""
        parts.append(f"{k}: {v[:70]}{tail}")
    return ", ".join(parts)


def result_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content or "")


def colored_summary_line(line: str) -> list[Span]:
    for icon, (sym, color) in ICON_MAP.items():
        if line.startswith(icon):
            rest = line[len(icon):].strip()
            m = re.match(r"(\S+) \[([^\]]+)\] (.*?)(  -> (.*))?$", rest)
            if m:
                spans: list[Span] = [(sym + " ", color), (m.group(2).split(",")[0] + "  ", color), (m.group(3), FG)]
                if m.group(5):
                    spans += [("  → " + m.group(5), color)]
                return spans
            return [(sym + " ", color), (rest, FG)]
    return [(line, FG)]


def render_result(name: str, text: str) -> list[list[Span]]:
    _, tool = tool_label(name)
    pre = [("  ⎿  ", DIM)]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        data = None
    if tool == "verify_answer" and isinstance(data, dict):
        head = "BLOCKED" if data.get("blocked") else data.get("action", "").upper()
        color = RED if data.get("blocked") else GREEN
        tiers = data.get("stats", {}).get("decided_by_tier", {})
        tier_txt = " · ".join(f"tier {t}: {n} claim{'s' if n != 1 else ''}" for t, n in sorted(tiers.items()))
        stats = data.get("stats", {})
        out = wrap([(f"{head}  ", color), (f"grounding score {data.get('grounding_score', 0):.0%}  ·  {tier_txt}"
                                           f"  ·  {stats.get('latency_ms', 0) / 1000:.1f}s", DIM)], 5, pre)
        for line in data.get("summary", []):
            out += wrap(colored_summary_line(line), 7, [("     ", FG)])
        for c in data.get("claims", []):
            if c.get("verdict") == "contradicted" and c.get("evidence"):
                ev = " ".join(c["evidence"][0]["snippet"].split())[:230]
                out += wrap([("evidence (" + c["evidence"][0]["tool"] + "): ", BLUE), (f'"{ev}…"', DIM)], 7, [("     ", FG)])
        return out
    if tool == "rewrite_grounded" and isinstance(data, dict):
        out: list[list[Span]] = []
        first = True
        for para in data.get("rewritten_answer", "").split("\n"):
            if not para.strip():
                continue
            spans: list[Span] = []
            for part in re.split(r"(\[\d+\])", para):
                spans.append((part, BLUE if re.fullmatch(r"\[\d+\]", part) else FG))
            out += wrap(spans, 5, pre if first else [("     ", FG)])
            first = False
        return out
    if tool in ("explain_verdict", "verify_claim") and isinstance(data, dict):
        return wrap([(f"{data.get('verdict', '')}: ", RED), (data.get("reason", ""), FG)], 5, pre)
    lines = [l for l in text.splitlines() if l.strip()]
    out = []
    for i, l in enumerate(lines[:3]):
        out += wrap([(l[:200], DIM)], 5, pre if i == 0 else [("     ", FG)])[:1]
    if len(lines) > 3:
        out.append([("     … +" + str(len(lines) - 3) + " lines (recorded in the evidence ledger)", DIM)])
    return out


CAPTIONS = {
    "search": ("Tool call", "Proxied through the firewall: the output is recorded in the evidence ledger"),
    "verify_answer": ("Verify", "Each claim is checked: tier 0 exact, tier 1 NLI model on CPU, tier 2 LLM judge"),
    "rewrite_grounded": ("Rewrite", "Wrong figures replaced with evidence values; citations to the tool output added"),
    "explain_verdict": ("Explain", "The exact evidence behind a verdict"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="docs/demo.mp4")
    ap.add_argument("--gif", default="docs/demo.gif")
    args = ap.parse_args()

    events = [json.loads(l) for l in Path(args.log).read_text(encoding="utf-8").splitlines() if l.strip().startswith("{")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    v = Video(out)
    card(v, "Hallucination Firewall", ["An MCP server that checks an agent's answer against its tool outputs",
                                        "Recorded from a real Claude Code session"], 3.2)
    d = Director(v)
    d.say("Task", "Ask Claude Code to check a draft client email against the UAE FTA guides")
    d.hold(0.6)
    prompt_lines = []
    for i, para in enumerate(args.prompt.split("\n")):
        prompt_lines += wrap([(para, FG)], 2, [("> ", DIM)] if i == 0 else [("  ", FG)])
    d.add(prompt_lines, cps=110, max_seconds=7)
    d.hold(1.2)

    names: dict[str, str] = {}
    final_text = ""
    for ev in events:
        if ev.get("type") == "assistant":
            for block in ev["message"].get("content", []):
                if block.get("type") == "text" and block["text"].strip():
                    d.blank()
                    lines: list[list[Span]] = []
                    for j, para in enumerate(block["text"].strip().split("\n")):
                        para = para.rstrip()
                        if not para:
                            lines.append([("", FG)])
                            continue
                        clean = para.replace("**", "")
                        for icon, (sym, _col) in ICON_MAP.items():
                            clean = clean.replace(icon, sym)
                        lines += wrap([(clean, FG)], 2, [("⏺ ", FG)] if j == 0 else [("  ", FG)])
                    d.say("Claude", "Writing")
                    d.add(lines, cps=260, max_seconds=6)
                    d.hold(0.8)
                    final_text = block["text"]
                elif block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
                    server, tool = tool_label(block["name"])
                    key = "search" if "search" in tool else tool
                    if key in CAPTIONS:
                        d.say(*CAPTIONS[key])
                    else:
                        d.say("Tool call", tool)
                    label = f"{server} - {tool}" if server else tool
                    d.blank()
                    args_txt = short_args(block.get("input", {}))
                    if tool in ("verify_answer", "rewrite_grounded"):
                        args_txt = f'draft: "{" ".join(block["input"].get("draft", "").split())[:90]}…"'
                    d.add(wrap([("⏺ ", GREEN), (label, FG), (f" ({args_txt})", DIM)], 2), cps=500, max_seconds=1.2)
                    d.hold(0.7)
        elif ev.get("type") == "user":
            content = ev.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        name = names.get(block.get("tool_use_id"), "")
                        lines = render_result(name, result_text(block.get("content")))
                        _, tool = tool_label(name)
                        d.add(lines, cps=600 if tool == "verify_answer" else 900, max_seconds=3)
                        d.hold(4.5 if tool in ("verify_answer", "rewrite_grounded") else 1.2)
    d.say("Done", "The corrected email cites the FTA guide passages it relies on")
    d.hold(4.0)
    card(v, "Hallucination Firewall", ["github.com/Parvathy-Raj/hallucination-firewall",
                                        "Tier 0 exact checks · Tier 1 NLI on CPU · Tier 2 LLM judge only when needed"], 3.5)
    v.close()
    print(f"wrote {out} ({v.frames / FPS:.1f}s)")
    if args.gif:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out), "-vf",
                        "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        args.gif], check=True)
        print(f"wrote {args.gif}")
    _ = final_text, textwrap


if __name__ == "__main__":
    main()
