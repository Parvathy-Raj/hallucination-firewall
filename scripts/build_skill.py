"""Build the Agensi skill zip from skill/hallucination-firewall/.

    python scripts/build_skill.py

Checks the SKILL.md frontmatter and writes dist/hallucination-firewall.zip with a single
top-level hallucination-firewall/ folder of text files, as Agensi requires.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = "hallucination-firewall"
SKILL_DIR = ROOT / "skill" / SKILL
DIST = ROOT / "dist" / f"{SKILL}.zip"
ALLOWED = {".md", ".txt", ".json", ".yaml", ".yml"}


def check_frontmatter() -> None:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    assert m, "SKILL.md must start with YAML frontmatter"
    fields = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    assert fields.get("name", "").strip() == SKILL, "name must match the folder name"
    desc = fields.get("description", "").strip()
    assert 0 < len(desc) <= 1024, f"description must be 1-1024 chars (is {len(desc)})"


def build_zip() -> list[str]:
    DIST.parent.mkdir(exist_ok=True)
    names = []
    with zipfile.ZipFile(DIST, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SKILL_DIR.rglob("*")):
            if f.is_dir():
                continue
            assert not f.is_symlink(), f"symlink not allowed: {f}"
            assert f.suffix in ALLOWED, f"unexpected file type: {f}"
            name = f"{SKILL}/{f.relative_to(SKILL_DIR).as_posix()}"
            z.write(f, name)
            names.append(name)
    return names


if __name__ == "__main__":
    check_frontmatter()
    files = build_zip()
    print(f"Wrote {DIST.relative_to(ROOT)} ({DIST.stat().st_size / 1024:.1f} KB):")
    for name in files:
        print("  " + name)
