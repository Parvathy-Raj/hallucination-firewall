import os

import pytest

# Tests run offline: no LLM, no models (tier 0 + heuristics). Model tiers are covered by evals.
os.environ.setdefault("HFW_LLM_PROVIDER", "none")
os.environ.setdefault("HFW_NLI_MODEL", "")
os.environ.setdefault("HFW_EMBED_MODEL", "")

EVIDENCE = (
    "If a person is resident in the UAE, the person is required to register for VAT if the total "
    "value of their taxable supplies exceeds the Mandatory Registration Threshold of AED 375,000 "
    "over the previous 12-month period. A person must apply to register within 30 days."
)


@pytest.fixture
def evidence() -> str:
    return EVIDENCE


@pytest.fixture
def fw(tmp_path, monkeypatch):
    monkeypatch.setenv("HFW_STATE_DIR", str(tmp_path / ".hfw"))
    from hfw import Firewall, Policy
    from hfw.config import Settings

    f = Firewall(Settings(), Policy.preset("balanced", tiers=[0]))
    f.record("docs.search", EVIDENCE)
    return f
