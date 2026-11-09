"""Hallucination Firewall: check an agent's answer against the tool outputs it actually received."""

from hfw.firewall import Firewall, HallucinationBlocked
from hfw.ledger import Ledger
from hfw.policy import Policy
from hfw.types import Claim, ClaimVerdict, Label, Report

__all__ = [
    "Claim",
    "ClaimVerdict",
    "Firewall",
    "HallucinationBlocked",
    "Label",
    "Ledger",
    "Policy",
    "Report",
]
__version__ = "0.1.0"
