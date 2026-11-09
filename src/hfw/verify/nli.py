"""Tier 1: a small NLI / fact-checking model on CPU.

Scores (evidence passage, claim) pairs and returns entailment and contradiction
probabilities. Default model: DeBERTa-v3-base fine-tuned on MNLI, FEVER and ANLI.
Any Hugging Face sequence classifier with "entailment"/"contradiction" labels works.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class NLIResult:
    entail: float
    contra: float
    neutral: float
    best_index: int  # passage index behind the deciding score


class NLIModel:
    _instances: dict[str, NLIModel | None] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # Some checkpoints ship fp16 weights, which are ~10x slower on CPU than fp32.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).float().eval()
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.i_ent = next(i for i, l in labels.items() if l.startswith("entail"))
        self.i_con = next((i for i, l in labels.items() if l.startswith("contra")), None)
        self.i_neu = next((i for i, l in labels.items() if l.startswith("neutral")), None)

    @classmethod
    def get(cls, model_name: str) -> NLIModel | None:
        with cls._lock:
            if model_name not in cls._instances:
                try:
                    cls._instances[model_name] = cls(model_name)
                except Exception as e:  # missing [ml] extra or download failure
                    import sys

                    print(f"[hfw] tier 1 disabled: could not load {model_name}: {e}", file=sys.stderr)
                    cls._instances[model_name] = None
            return cls._instances[model_name]

    def score(self, claim: str, passages: list[str]) -> NLIResult:
        if not passages:
            return NLIResult(0.0, 0.0, 1.0, -1)
        # Score each passage, plus the top two joined (claims that need two passages).
        premises = list(passages)
        if len(passages) >= 2:
            premises.append(passages[0] + " " + passages[1])
        with self.torch.inference_mode():
            enc = self.tok(
                premises,
                [claim] * len(premises),
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        ent = probs[:, self.i_ent]
        con = probs[:, self.i_con] if self.i_con is not None else 1 - ent
        bi_e = int(ent.argmax())
        bi_c = int(con.argmax())
        best_e, best_c = float(ent[bi_e]), float(con[bi_c])
        # Contradiction counts only from a passage that does not also entail the claim.
        if best_e >= best_c:
            idx = min(bi_e, len(passages) - 1)
            neu = float(probs[bi_e, self.i_neu]) if self.i_neu is not None else 1 - best_e
            return NLIResult(best_e, float(con[bi_e]), neu, idx)
        idx = min(bi_c, len(passages) - 1)
        neu = float(probs[bi_c, self.i_neu]) if self.i_neu is not None else 0.0
        return NLIResult(best_e, best_c, neu, idx)
