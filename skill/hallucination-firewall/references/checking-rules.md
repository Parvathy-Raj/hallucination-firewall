# Checking rules (manual mode)

These are the rules the Hallucination Firewall MCP server applies in code, written out so an
agent can apply them by hand.

## 1. What counts as evidence

- Counts: outputs of tools called in this conversation (search results, file reads, web
  fetches, API or database responses, MCP tool results), and documents the user pasted or
  attached.
- Does not count: your own earlier messages, your training knowledge, the user's question,
  or the draft being checked.
- If there is no evidence at all, say so: "No tool output covers this, so I can't verify it."
  Don't pass the draft as checked.

## 2. Splitting a draft into claims

- One claim per checkable fact. "The threshold is AED 375,000 and you must apply within
  30 days" is two claims.
- Keep the subject in each claim, so it can be checked alone ("The deadline is 30 days",
  not "It is 30 days").
- Skip these (➖ not checkable):
  - opinions and advice ("you should register early");
  - hedges ("may", "might", "it depends", "typically", "usually");
  - questions;
  - lead-ins such as "Based on the documents:" or "Here is what I found:".
- A hedge doesn't cover a specific number: "may be around AED 500,000" is still checked
  against the evidence's number.

## 3. Numbers, dates and names: exact matching

Check these first. They are where most hallucinations hide, and the check is mechanical.

| In the claim | Matches the evidence if... |
|---|---|
| A number, amount or percentage | It is the same value. Formats may differ: "AED 18,400" = "18400 AED" = "AED 18.4k". |
| A rounded number ("about 18,000", "18k", "1.2M") | The evidence rounds to it at the claim's precision. "about AED 18,000" matches 18,400. "AED 185,000" does **not** match 187,500, because the claim states thousands. |
| A date | It is the same day, month or year at the claim's precision. "March 2026" matches 2026-03-15. "15 March" does not match 2026-03-16. |
| A quarter or period ("Q3", "previous 12 months", "30 days") | The same period and length. "60 days" vs "30 days" is a contradiction. |
| A name (person, company, product, law, place) | The same name appears in the evidence. A different name in the same role is a contradiction. |
| An ID ("INV-1042", "v2.3.1", "Article 50") | Treat it as a name, never as a quantity. |

If the claim's number is in the evidence for a **different** subject (the 30 days is the
appeal window, not the application deadline), that is not support. Read the sentence
around it.

More cases:

- **Spelling of names.** A different spelling is a different name: "Sarah Haddad" vs
  "Sara Haddad" is ❌. Fix it to the evidence's spelling. Case and accents don't count
  ("ACME" = "Acme").
- **Calculated values** (a ratio, sum, difference, or days between dates). Do the arithmetic
  from the evidence's values. If the result rounds to the claim at the claim's precision,
  it's ✅. Cite the outputs you used. "about 60%" for 61,200 / 100,000 is ✅. "about 50%" is ❌.
- **Time words.** "now", "currently" and "today" are fine when the evidence gives the current
  state. If the evidence carries a date, and it may be stale, keep the value and say "as of
  <that date>".

## 4. Meaning: everything else

For a claim with no number, date or name, or once those match, check the meaning:

- ✅ **supported**: the evidence states it, or it follows directly (a paraphrase, or
  a generic framing such as "the guide explains how to register").
- ❌ **contradicted**: the evidence states the opposite, or a different fact in the same
  slot. Watch for negation ("must" vs "need not", "includes" vs "excludes"), the wrong
  subject, and reversed conditions ("above" vs "below").
- ⚠️ **unsupported**: the claim adds specific content the evidence doesn't contain, even
  if it's probably true. Example: "Registration is free" when no output mentions fees.

**Missing conditions.** A claim that drops part of the evidence is still ✅ if what's left
is true ("the voluntary threshold is AED 187,500" when the evidence adds "over the previous
12 months"). It is ❌ if the dropped part changes when or to whom it applies. Examples: the
claim says "supplies" where the evidence says "supplies and imports", or it leaves out an
"or is expected to exceed" test. Fix it by restoring the condition.

When unsure between ✅ and ⚠️, choose ⚠️. When unsure between ⚠️ and ❌, choose ⚠️, and say
what the evidence does say.

## 5. Fixing

| Verdict | Fix |
|---|---|
| ❌ with one clear value | Swap in the evidence's value, in the claim's format, and cite it. Keep any condition that comes with the value ("within 30 days **of the end of the month** ..."). |
| ❌ without a single value | Rewrite the sentence to say what the evidence says, and cite it. |
| ⚠️ | Remove it. If the user needs it, keep it marked "(not confirmed by the sources)". |
| ✅ | Keep it, and add the citation [n]. |
| ➖ | Keep it, unless it rests on a claim you removed or changed. Then base it on what's left, or drop it ("paid late twice, so call them" becomes "39 days overdue, so call them"). |

After fixing, re-read the answer once: a fix can break the sentence around it.

If the evidence doesn't answer part of the user's question, say so in one line ("No tool
returned the account's payment history.") rather than filling the gap.

## 6. Strictness

Default (balanced): fix every ❌, and hedge or remove every ⚠️.

If the user asks for **strict** checking, remove every ⚠️ claim and say how many you
removed. For legal, tax, financial, medical or compliance answers, use strict unless the
user says otherwise. If they ask for **lenient** checking (drafts,
brainstorming with sources), fix the ❌ claims and just list the ⚠️ ones.
