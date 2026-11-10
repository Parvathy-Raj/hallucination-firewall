import pytest

from hfw.verify.exact import check, extract_quantities


@pytest.mark.parametrize(
    "text,kind,value",
    [
        ("revenue was $4.2M", "currency:USD", 4_200_000),
        ("a fee of AED 375,000", "currency:AED", 375_000),
        ("costs 20,000 dirhams", "currency:AED", 20_000),
        ("grew 12% last year", "percent", 12),
        ("grew 12.5 percent", "percent", 12.5),
        ("within 30 days", "unit:day", 30),
        ("within thirty days", "unit:day", 30),
        ("on 13 September 2018", "date", 20180913),
        ("on March 1, 2024", "date", 20240301),
        ("in 2023", "year", 2023),
        ("1.5 billion users", "unit:users", 1.5e9),
        ("in Q3", "quarter", 3),
    ],
)
def test_extract(text, kind, value):
    qs = extract_quantities(text)
    assert any(q.kind == kind and q.value == pytest.approx(value) for q in qs), qs


def test_ignores_identifiers_and_section_numbers():
    assert extract_quantities("see section 4.3.3 of VATG001") == []


def test_wrong_number_is_contradicted_with_correction(evidence):
    r = check("The VAT registration threshold is AED 500,000.", [evidence])
    assert r.verdict == "contradicted"
    assert r.correction == ("AED 500,000", "AED 375,000")


def test_right_number_matches(evidence):
    assert check("The VAT registration threshold is AED 375,000.", [evidence]).verdict == "match"


def test_rounding_tolerated():
    assert check("Revenue was $4.2M.", ["Revenue: $4,213,000 for the quarter"]).verdict == "match"


def test_no_numbers_abstains(evidence):
    assert check("Residents must register for VAT.", [evidence]).verdict == "abstain"


def test_date_mismatch():
    r = check("The contract was signed on 3 March 2024.", ["The contract was signed on 5 March 2024."])
    assert r.verdict == "contradicted"


def test_identifiers_are_not_quantities():
    assert extract_quantities("Invoice INV-1042 and order #77") == []


def test_json_field_context_picks_right_number():
    ev = ["invoice_id: INV-1042\namount_aed: 18400\ndue_date: 2026-12-01"]
    r = check("The amount due is AED 18,900.", ev)
    # the correction keeps the claim's own formatting
    assert r.verdict == "contradicted" and r.correction == ("AED 18,900", "AED 18,400")


def test_names_matched_word_by_word():
    r = check("Invoice INV-1042 is unpaid.", ["invoice_id: INV-1042\nstatus: unpaid"])
    assert r.verdict == "abstain"


def test_correction_prefers_same_obligation_threshold():
    """A mandatory-threshold claim must be corrected with the mandatory value, not the voluntary one."""
    voluntary = ("A person can voluntarily register for VAT if, at the end of any month, the total value of "
                 "the person's taxable supplies and imports in the previous 12 months exceeds the Voluntary "
                 "Registration Threshold of AED 187,500.")
    mandatory = ("A person resident in the UAE is required to register for VAT if the total value of their "
                 "taxable supplies made within the UAE and imports into the UAE exceeded AED 375,000 over the "
                 "previous 12-month period.")
    claim = "A person must register for VAT once their taxable supplies exceed AED 500,000 over the previous 12 months."
    r = check(claim, [voluntary, mandatory])
    assert r.verdict == "contradicted"
    assert r.correction == ("AED 500,000", "AED 375,000")
    r2 = check("You can register voluntarily once supplies exceed AED 200,000.", [mandatory, voluntary])
    assert r2.correction == ("AED 200,000", "AED 187,500")
