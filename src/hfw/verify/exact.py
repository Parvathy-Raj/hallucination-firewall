"""Tier 0: exact checks for numbers, dates, percentages, currency and names.

Costs nothing and catches the most common agent error: a figure that changed
between the tool output and the answer ("$4.2M" vs "$4.7M").

For every quantity in the claim we look for the same normalized value in the
retrieved evidence. If it is missing and the best-matching evidence passage
has a quantity of the same kind in a similar context, that is a contradiction
with a concrete correction. Claims with no quantities or names are passed on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hfw.text import tokenize

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    )
}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9

SCALE = {
    "k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "mm": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12, "tn": 1e12, "trillion": 1e12,
    "lakh": 1e5, "crore": 1e7,
}
CURRENCY = {
    "$": "USD", "usd": "USD", "us$": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP",
    "aed": "AED", "dhs": "AED", "dh": "AED", "dirham": "AED", "dirhams": "AED", "₹": "INR",
    "inr": "INR", "rs": "INR", "sar": "SAR", "¥": "JPY", "jpy": "JPY",
}
WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90,
    "hundred": 100,
}
UNIT_WORDS = {
    "day": "day", "days": "day", "month": "month", "months": "month", "year": "year",
    "years": "year", "hour": "hour", "hours": "hour", "week": "week", "weeks": "week",
    "minute": "minute", "minutes": "minute", "kg": "kg", "km": "km", "employees": "people",
    "people": "people", "users": "users", "customers": "customers",
}

_CUR_PREFIX = r"(?P<cur>US\$|\$|€|£|₹|¥|AED|USD|EUR|GBP|INR|SAR|JPY|Dhs?\.?|Rs\.?)\s?"
_NUM = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_SCALE = r"(?:\s?(?P<scale>thousand|million|billion|trillion|mn|bn|tn|lakh|crore|[kmbt])\b)?"
_SUFFIX = (
    r"(?:\s?(?P<pct>%|percent\b|per\s?cent\b|pp\b|percentage points?\b)"
    r"|\s?(?P<cur2>AED|USD|EUR|GBP|INR|SAR|dirhams?|dollars?|euros?)\b"
    r"|\s(?P<unit>days?|months?|years?|hours?|weeks?|minutes?|kg|km|employees|people|users|customers)\b)?"
)
QTY_RE = re.compile(rf"(?:{_CUR_PREFIX})?{_NUM}{_SCALE}{_SUFFIX}", re.IGNORECASE)
WORD_QTY_RE = re.compile(
    r"\b(?P<w>" + "|".join(WORD_NUMS) + r")\s(?P<unit>days?|months?|years?|hours?|weeks?)\b", re.IGNORECASE
)
DATE_RE = re.compile(
    r"\b(?:(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s(?P<m1>[A-Za-z]{3,9})\.?,?\s(?P<y1>\d{4})"
    r"|(?P<m2>[A-Za-z]{3,9})\.?\s(?P<d2>\d{1,2})(?:st|nd|rd|th)?,?\s(?P<y2>\d{4})"
    r"|(?P<y3>\d{4})-(?P<mo3>\d{2})-(?P<d3>\d{2}))\b"
)
QUARTER_RE = re.compile(r"\b(?:Q(?P<q>[1-4])|(?P<qw>first|second|third|fourth) quarter)\b", re.IGNORECASE)
NAME_RE = re.compile(
    r"\b(?:[A-Z][\w&'-]*\w(?:\s+(?:of|de|and|&)?\s*[A-Z][\w&'-]*\w)*|[A-Z]{2,}[a-z]?)\b"
)


@dataclass(frozen=True)
class Qty:
    kind: str  # currency:AED | percent | year | date | quarter | count | unit:day ...
    value: float
    surface: str
    start: int
    end: int
    unit: float = 0.0  # place value of the last significant digit ("4.2M" -> 100,000)
    sig: int = 99  # significant digits as written


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def extract_quantities(text: str) -> list[Qty]:
    out: list[Qty] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    for m in DATE_RE.finditer(text):
        try:
            if m.group("y3"):
                y, mo, d = int(m.group("y3")), int(m.group("mo3")), int(m.group("d3"))
            elif m.group("y1"):
                mo = MONTHS.get(m.group("m1").lower().rstrip("."), 0)
                y, d = int(m.group("y1")), int(m.group("d1"))
            else:
                mo = MONTHS.get(m.group("m2").lower().rstrip("."), 0)
                y, d = int(m.group("y2")), int(m.group("d2"))
        except (TypeError, ValueError):
            continue
        if not mo:
            continue
        out.append(Qty("date", y * 10000 + mo * 100 + d, m.group(0), m.start(), m.end()))
        taken.append((m.start(), m.end()))

    for m in QUARTER_RE.finditer(text):
        q = m.group("q") or {"first": "1", "second": "2", "third": "3", "fourth": "4"}[m.group("qw").lower()]
        out.append(Qty("quarter", float(q), m.group(0), m.start(), m.end()))
        taken.append((m.start(), m.end()))

    for m in QTY_RE.finditer(text):
        a, b = m.start(), m.end()
        if not free(a, b):
            continue
        # skip parts of identifiers like "VATG001", "Q3" or section numbers "4.3.3"
        if a > 0 and (text[a - 1].isalnum() or text[a - 1] in "./_#"):
            continue
        if a > 1 and text[a - 1] == "-" and text[a - 2].isalnum():  # IDs like INV-1042
            continue
        if b < len(text) and text[b : b + 1].isalnum() and not m.group("scale"):
            continue
        if re.match(r"\.\d", text[b : b + 2]):  # section number like 4.3.3
            continue
        num = _to_float(m.group("num"))
        scale = (m.group("scale") or "").lower()
        # single-letter scales only count right after a currency or digit ("$4.2M", "5k")
        if scale and len(scale) == 1 and " " in m.group(0)[len(m.group("num")) :][:2]:
            scale = ""
        value = num * SCALE.get(scale, 1.0)
        cur = m.group("cur") or m.group("cur2")
        if cur:
            key = cur.lower().rstrip(".").rstrip("s") if cur.lower().startswith("dirham") else cur.lower().rstrip(".")
            kind = f"currency:{CURRENCY.get(key, CURRENCY.get(key.rstrip('s'), key.upper()))}"
            if cur.lower().startswith("dollar"):
                kind = "currency:USD"
            elif cur.lower().startswith("euro"):
                kind = "currency:EUR"
        elif m.group("pct"):
            kind = "percent"
        elif m.group("unit"):
            kind = f"unit:{UNIT_WORDS.get(m.group('unit').lower(), m.group('unit').lower())}"
        elif not scale and "," not in m.group("num") and "." not in m.group("num") and 1900 <= num <= 2100:
            kind = "year"
        else:
            kind = "count"
        unit, sig = _precision(m.group("num"), SCALE.get(scale, 1.0))
        out.append(Qty(kind, value, m.group(0).strip(), a, b, unit, sig))
        taken.append((a, b))

    for m in WORD_QTY_RE.finditer(text):
        if free(m.start(), m.end()):
            unit = UNIT_WORDS.get(m.group("unit").lower(), m.group("unit").lower())
            out.append(Qty(f"unit:{unit}", float(WORD_NUMS[m.group("w").lower()]), m.group(0), m.start(), m.end()))
    return sorted(out, key=lambda q: q.start)


def _precision(num: str, scale: float) -> tuple[float, int]:
    digits = num.replace(",", "")
    if "." in digits:
        whole, frac = digits.split(".")
        sig = len(whole.lstrip("0") + frac) or 1
        return 10 ** -len(frac) * scale, sig
    stripped = digits.rstrip("0")
    tz = len(digits) - len(stripped)
    return 10**tz * scale, max(1, len(stripped.lstrip("0")))


def _same(a: Qty, b: Qty) -> bool:
    if a.kind != b.kind:
        # a bare count may be written with or without its currency/unit in the evidence
        if not ({a.kind, b.kind} & {"count"}):
            return False
    if a.value == b.value:
        return True
    if a.kind in ("year", "date", "quarter") or b.kind in ("year", "date", "quarter"):
        return False
    # Rounding is fine when it is consistent with how precisely the coarser number was
    # written: "$4.2M" matches "$4,213,000", but "AED 185,000" does not match "AED 187,500"
    # and a one-digit "AED 400,000" must match exactly.
    coarse = a if a.unit >= b.unit else b
    if coarse.sig < 2:
        return False
    return abs(a.value - b.value) <= coarse.unit / 2 + 1e-9


def _context(text: str, q: Qty, window: int = 8) -> set[str]:
    left = tokenize(text[: q.start])[-window:]
    right = tokenize(text[q.end :])[:window]
    return set(left + right)


@dataclass
class ExactResult:
    verdict: str  # contradicted | match | missing | abstain
    confidence: float
    reason: str
    chunk_index: int | None = None
    correction: tuple[str, str] | None = None
    unknown_names: list[str] | None = None


def check(claim: str, passages: list[str]) -> ExactResult:
    """Compare quantities and names in `claim` with the evidence `passages` (best first)."""
    cq = [q for q in extract_quantities(claim) if q.kind != "quarter" or len(passages) > 0]
    names = _names(claim)
    evidence_text = "\n".join(passages)
    ev_lower = evidence_text.lower()
    ev_tokens = set(tokenize(evidence_text, drop_stop=False))
    unknown = [n for n in names if not set(tokenize(n, drop_stop=False)) <= ev_tokens]

    if not cq:
        if unknown and len(unknown) == len(names) and names:
            return ExactResult("missing", 0.5, f"names not found in any evidence: {', '.join(unknown)}", unknown_names=unknown)
        return ExactResult("abstain", 0.0, "no numbers, dates or names to check", unknown_names=unknown or None)

    pq = [extract_quantities(p) for p in passages]
    claim_tokens = set(tokenize(claim))
    missing: list[Qty] = []
    best_conflict: tuple[float, Qty, Qty, int] | None = None
    for q in cq:
        if any(_same(q, e) for qs in pq for e in qs):
            continue
        # also accept a literal surface match ("Q3", "2024-01-01")
        if q.surface.lower() in ev_lower:
            continue
        missing.append(q)
        qctx = _context(claim, q)
        for pi, qs in enumerate(pq):
            for e in qs:
                if not _comparable(q, e):
                    continue
                ectx = _context(passages[pi], e)
                overlap = len(qctx & ectx) / max(1, len(qctx))
                topical = len(claim_tokens & set(tokenize(passages[pi]))) / max(1, len(claim_tokens))
                score = overlap * 0.7 + topical * 0.3 - pi * 0.02
                if best_conflict is None or score > best_conflict[0]:
                    best_conflict = (score, q, e, pi)

    if not missing:
        return ExactResult("match", 0.9, "every number/date in the claim appears in the evidence")

    if best_conflict and best_conflict[0] >= 0.3:
        score, q, e, pi = best_conflict
        conf = min(0.97, 0.6 + score * 0.4)
        return ExactResult(
            "contradicted",
            conf,
            f'claim says "{q.surface}" but the evidence says "{e.surface}"',
            chunk_index=pi,
            correction=(q.surface, _restyle(q, e)),
        )
    names_txt = ", ".join(m.surface for m in missing)
    return ExactResult("missing", 0.6, f"not found in any evidence: {names_txt}", unknown_names=unknown or None)


def _comparable(q: Qty, e: Qty) -> bool:
    if q.kind == e.kind:
        return True
    pair = {q.kind, e.kind}
    # "AED 375,000" vs "375,000" in a table: comparable
    return "count" in pair and not (pair & {"year", "date", "quarter", "percent"})


_COMMON_CAPS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "In", "On", "For", "If", "Yes", "No",
    "However", "Also", "Note", "Under", "As", "At", "By", "To", "When", "Where", "While", "Each",
    "Any", "All", "Some", "Most", "Its", "Their", "There", "Here", "With", "From", "After", "Before",
    "Q1", "Q2", "Q3", "Q4", "I", "We", "You", "They", "He", "She", "Our", "Your", "But", "And", "Or",
}


def _names(text: str) -> list[str]:
    names = []
    for m in NAME_RE.finditer(text):
        n = m.group(0).strip()
        first_word = n.split()[0]
        if first_word in _COMMON_CAPS:
            n = " ".join(n.split()[1:])
            if not n:
                continue
        if m.start() == 0 and " " not in n and not n.isupper():
            continue  # sentence-initial capital, probably not a name
        if n.lower() in MONTHS or len(n) < 2:
            continue
        names.append(n)
    return names


def _restyle(q: Qty, e: Qty) -> str:
    """Write the evidence value in the claim's style: "AED 18,900" + "18400" -> "AED 18,400"."""
    if e.kind == q.kind or e.kind != "count":
        return e.surface
    num = re.search(r"\d[\d,]*(?:\.\d+)?", q.surface)
    if not num:
        return e.surface
    claim_num = num.group(0)
    scale = 1.0
    tail = q.surface[num.end():].strip().lower()
    for word, mult in SCALE.items():
        if tail.startswith(word) and (len(tail) == len(word) or not tail[len(word)].isalpha()):
            scale = mult
            break
    v = e.value / scale
    if "," in claim_num or (v >= 10000 and "." not in claim_num):
        new = f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"
    else:
        new = f"{v:g}"
    return q.surface[: num.start()] + new + q.surface[num.end():]
