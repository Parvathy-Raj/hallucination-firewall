"""Enforce verification in code: the answer cannot leave the function unverified."""

from hfw import Firewall, HallucinationBlocked, Policy

fw = Firewall(policy=Policy.preset("balanced", action="block"))


@fw.watch
def get_invoice(invoice_id: str) -> dict:
    # stand-in for a real tool / API call
    return {"invoice_id": invoice_id, "amount_aed": 18_400, "due_date": "2026-12-01", "status": "unpaid"}


def answer(question: str) -> str:
    inv = get_invoice(invoice_id="INV-1042")
    draft = f"Invoice {inv['invoice_id']} is unpaid. The amount due is AED 18,900, due on 1 December 2026."
    try:
        return fw.guard(draft)
    except HallucinationBlocked as e:
        for v in e.report.flagged:
            print(f"{v.label.icon} {v.claim.text} -> {v.reason}")
        fixed, _ = fw.rewrite_grounded(draft)
        return fixed


if __name__ == "__main__":
    print(answer("What do I owe on INV-1042?"))
