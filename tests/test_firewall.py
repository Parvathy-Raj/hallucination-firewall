import json

import pytest

from hfw import Firewall, HallucinationBlocked, Label, Policy
from hfw.claims import extract_claims
from hfw.ledger import Ledger, chunk_text
from hfw.text import normalize_tool_output, split_sentences


def test_split_sentences_handles_abbreviations_and_bullets():
    s = split_sentences("Revenue rose 5%, e.g. in Dubai. It fell in Q4.\n- First point here\n- Second point")
    assert s == ["Revenue rose 5%, e.g. in Dubai.", "It fell in Q4.", "- First point here", "- Second point"]


def test_claim_kinds():
    claims = extract_claims(
        "Here is what I found. The threshold is AED 375,000. It might change next year. "
        "I recommend talking to an advisor."
    )
    kinds = [(c.text, c.kind) for c in claims]
    assert ("The threshold is AED 375,000.", "factual") in kinds
    assert any(k == "hedge" for _, k in kinds)
    assert any(k == "opinion" for _, k in kinds)
    assert not any("Here is what I found" in t for t, _ in kinds)


def test_json_tool_output_is_flattened():
    text = normalize_tool_output(json.dumps({"customer": {"name": "Acme", "arr_usd": 4200000}}))
    assert "customer.arr_usd: 4200000" in text


def test_mcp_content_list_is_unwrapped():
    assert normalize_tool_output([{"type": "text", "text": "hello world"}]) == "hello world"


def test_chunking_overlaps_and_dedupes():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) > 3 and all(len(c) <= 260 for c in chunks)


def test_ledger_search_and_persistence(tmp_path):
    p = tmp_path / "ledger.jsonl"
    l1 = Ledger(p)
    l1.add("a", "The VAT rate is 5 percent in the UAE.")
    l1.add("b", "Corporate tax applies at 9 percent above AED 375,000.")
    l2 = Ledger(p)
    l2.sync()
    assert len(l2) == 2
    top, _ = l2.search("corporate tax rate", k=1)[0]
    assert top.source == "b"


def test_verify_blocks_wrong_number(fw):
    r = fw.verify("The mandatory registration threshold is AED 500,000 over the previous 12 months.")
    assert r.blocked and r.action == "block"
    v = r.verdicts[0]
    assert v.label == Label.CONTRADICTED and v.tier == 0
    assert v.correction == ("AED 500,000", "AED 375,000")
    assert v.evidence[0].source == "docs.search"


def test_verify_passes_correct_answer(fw):
    r = fw.verify("The mandatory registration threshold is AED 375,000. You must apply within 30 days.")
    assert not r.blocked
    assert all(v.label == Label.SUPPORTED for v in r.verdicts), [v.reason for v in r.verdicts]


def test_unsupported_annotated_not_blocked_in_balanced(fw):
    r = fw.verify("The FTA charges a registration fee of AED 1,000.")
    assert r.verdicts[0].label == Label.UNSUPPORTED
    assert not r.blocked and r.action == "annotate"


def test_strict_blocks_unsupported(fw):
    r = fw.verify("The FTA charges a registration fee of AED 1,000.", Policy.preset("strict", tiers=[0]))
    assert r.blocked


def test_no_evidence_means_unsupported(tmp_path, monkeypatch):
    monkeypatch.setenv("HFW_STATE_DIR", str(tmp_path))
    f = Firewall(policy=Policy.preset("balanced", tiers=[0]))
    r = f.verify("Revenue was $4.2M in Q3.")
    assert r.verdicts[0].label == Label.UNSUPPORTED
    assert "No tool outputs" in r.message


def test_rewrite_fixes_number_and_cites(fw):
    text, _ = fw.rewrite_grounded(
        "The mandatory registration threshold is AED 500,000 over the previous 12 months. "
        "The FTA charges a registration fee of AED 1,000."
    )
    assert "AED 375,000" in text and "AED 500,000" not in text
    assert "[1]" in text and "Sources (tool outputs)" in text
    assert "not confirmed" in text


def test_guard_raises(fw):
    with pytest.raises(HallucinationBlocked) as e:
        fw.guard("The threshold is AED 400,000.")
    assert e.value.report.flagged


def test_watch_decorator_records(tmp_path, monkeypatch):
    monkeypatch.setenv("HFW_STATE_DIR", str(tmp_path))
    f = Firewall(policy=Policy.preset("balanced", tiers=[0]))

    @f.watch
    def lookup(customer: str) -> dict:
        return {"customer": customer, "arr_usd": 4_200_000}

    lookup(customer="Acme")
    assert len(f.ledger) == 1
    assert f.verify("Acme's ARR is $5.1M.").blocked


def test_explain_verdict(fw):
    r = fw.verify("The threshold is AED 500,000.")
    d = fw.explain(r.verdicts[0].claim.id)
    assert d["tier_trace"][0]["tier"] == 0 and d["evidence"]


def test_policy_validation():
    p = Policy()
    with pytest.raises(ValueError):
        p.update(tiers=[3])
    with pytest.raises(ValueError):
        p.update(action="delete")
    assert p.update(strictness="strict").block_on == ["contradicted", "unsupported"]
