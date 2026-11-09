"""Run the benchmarks, write evals/results/*, and refresh the README results table.

    uv run python -m evals.run            (or: make eval)

Experiments
  * Tier ablation: tier 0, tiers 0+1, tiers 0+1+2 (routed), LLM judge on every claim.
  * Threshold sweep: tier-1 confidence threshold vs accuracy and share of claims sent to the judge.
  * Failure analysis: recall per hallucination type for each configuration.

Every NLI score and LLM call is cached in evals/data/, so configurations that make the
same call reuse it (and re-runs are free). Cost and latency still count each call the
configuration makes, using the recorded tokens and time of the original call.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.datasets import DATA, halueval, ragtruth, uae_traces
from hfw.claims import classify
from hfw.config import Settings, load_env
from hfw.ledger import Ledger
from hfw.llm import LLM
from hfw.policy import Policy
from hfw.types import Claim, Label
from hfw.verify.nli import NLIModel, NLIResult
from hfw.verify.pipeline import Verifier

RESULTS = Path(__file__).parent / "results"


def _key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


class _Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.data[d["k"]] = d

    def get(self, k: str) -> dict | None:
        return self.data.get(k)

    def put(self, k: str, d: dict) -> None:
        d["k"] = k
        self.data[k] = d
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


class CachedLLM(LLM):
    def __init__(self, *a, cache: _Cache, **kw):
        super().__init__(*a, **kw)
        self.cache = cache
        self.sim_seconds = 0.0

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        k = _key(self.model, system, user)
        hit = self.cache.get(k)
        if hit:
            self.usage.add(self.model, hit["in"], hit["out"])
            self.sim_seconds += hit["s"]
            return hit["text"]
        u0 = self.usage.snapshot()
        t = time.perf_counter()
        text = super().complete(system, user, max_tokens)
        u1 = self.usage.snapshot()
        self.cache.put(k, {"text": text, "s": time.perf_counter() - t,
                           "in": u1["input_tokens"] - u0["input_tokens"],
                           "out": u1["output_tokens"] - u0["output_tokens"]})
        return text


class CachedNLI:
    def __init__(self, name: str, cache: _Cache):
        self.name = name
        self.cache = cache
        self.sim_seconds = 0.0
        self._model: NLIModel | None = None

    def score(self, claim: str, passages: list[str]) -> NLIResult:
        k = _key(self.name, claim, *passages)
        hit = self.cache.get(k)
        if hit:
            self.sim_seconds += hit["s"]
            return NLIResult(hit["e"], hit["c"], hit["n"], hit["i"])
        if self._model is None:
            self._model = NLIModel.get(self.name)
        t = time.perf_counter()
        r = self._model.score(claim, passages)  # type: ignore[union-attr]
        self.cache.put(k, {"e": r.entail, "c": r.contra, "n": r.neutral, "i": r.best_index,
                           "s": time.perf_counter() - t})
        return r


class EvalVerifier(Verifier):
    def __init__(self, ledger, llm, nli):
        super().__init__(ledger, llm, "")
        self._cached_nli = nli

    @property
    def nli(self):  # type: ignore[override]
        return self._cached_nli


@dataclass
class Config:
    name: str
    tiers: list[int]
    threshold: float = 0.8


CONFIGS = [
    Config("LLM judge on every claim", [2]),
    Config("Tier 0 (exact)", [0]),
    Config("Tiers 0+1 (exact + NLI)", [0, 1]),
    Config("Tiers 0+1+2 (routed)", [0, 1, 2]),
]
SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def run_config(cfg: Config, examples: list[dict], llm: CachedLLM, nli: CachedNLI, embed: str) -> list[dict]:
    rows = []
    policy = Policy.preset("balanced", tiers=cfg.tiers, threshold=cfg.threshold)
    for ex in examples:
        ledger = Ledger(None, embed_model=embed)
        for src, text in ex["evidence"]:
            ledger.add(src, text)
        ver = EvalVerifier(ledger, llm, nli)
        c0, sim0, nsim0 = llm.usage.snapshot(), llm.sim_seconds, nli.sim_seconds
        t = time.perf_counter()
        preds = []
        for i, c in enumerate(ex["claims"]):
            claim = Claim(f"c{i}", c["text"], classify(c["text"]), c["text"], c["sentence_index"])
            v = ver.verify(claim, policy)
            preds.append({"label": v.label.value, "tier": v.tier, "gold": c["hallucinated"], "type": c["type"]})
        elapsed = (time.perf_counter() - t) + (llm.sim_seconds - sim0) + (nli.sim_seconds - nsim0)
        c1 = llm.usage.snapshot()
        rows.append({"id": ex["id"], "dataset": ex["dataset"], "preds": preds, "latency_s": elapsed,
                     "llm_calls": c1["llm_calls"] - c0["llm_calls"],
                     "cost": c1["est_cost_usd"] - c0["est_cost_usd"]})
    return rows


def metrics(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    judge = 0
    n_claims = 0
    clean_resp = clean_blocked = hal_resp = hal_caught = 0
    for r in rows:
        any_gold = any(p["gold"] for p in r["preds"])
        any_contra = any(p["label"] == Label.CONTRADICTED.value for p in r["preds"])
        any_flag = any(p["label"] in ("contradicted", "unsupported") for p in r["preds"])
        if any_gold:
            hal_resp += 1
            hal_caught += any_flag
        else:
            clean_resp += 1
            clean_blocked += any_contra
        for p in r["preds"]:
            n_claims += 1
            judge += p["tier"] == 2
            flag = p["label"] in ("contradicted", "unsupported")
            if p["gold"] and flag:
                tp += 1
            elif p["gold"]:
                fn += 1
            elif flag:
                fp += 1
            else:
                tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "claims": n_claims,
        "hallucinated_claims": tp + fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_flag_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_block_rate": clean_blocked / clean_resp if clean_resp else 0.0,
        "answer_recall": hal_caught / hal_resp if hal_resp else 0.0,
        "judge_share": judge / n_claims if n_claims else 0.0,
        "latency_p50_s": statistics.median(lat) if lat else 0.0,
        "latency_p95_s": lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))] if lat else 0.0,
        "cost_per_answer_usd": sum(r["cost"] for r in rows) / len(rows) if rows else 0.0,
        "llm_calls_per_answer": sum(r["llm_calls"] for r in rows) / len(rows) if rows else 0.0,
    }


def recall_by_type(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, list[int]] = {}
    for r in rows:
        for p in r["preds"]:
            if p["gold"]:
                out.setdefault(p["type"], [0, 0])
                out[p["type"]][0] += p["label"] in ("contradicted", "unsupported")
                out[p["type"]][1] += 1
    return {t: {"recall": a / b, "n": b} for t, (a, b) in sorted(out.items())}


def main() -> None:
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    s = Settings()
    model = os.environ.get("HFW_EVAL_MODEL", "openai/gpt-oss-20b")
    llm = CachedLLM(
        "openai", model, os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("HFW_EVAL_BASE_URL", "https://api.groq.com/openai/v1"),
        rpm=float(os.environ.get("HFW_EVAL_RPM", "8")),
        cache=_Cache(DATA / "cache_llm.jsonl"),
    )
    nli = CachedNLI(s.nli_model, _Cache(DATA / "cache_nli.jsonl"))

    examples = ragtruth() + halueval() + uae_traces()
    limit = int(os.environ.get("HFW_EVAL_LIMIT", "0"))
    if limit:  # quick smoke run: first N answers of each dataset
        seen: dict[str, int] = {}
        examples = [e for e in examples if (seen.__setitem__(e["dataset"], seen.get(e["dataset"], 0) + 1) or seen[e["dataset"]] <= limit)]
    datasets = sorted({e["dataset"] for e in examples})
    print(f"{len(examples)} answers, {sum(len(e['claims']) for e in examples)} claims: {datasets}")

    results: dict = {"judge_model": model, "nli_model": s.nli_model, "embed_model": s.embed_model,
                     "configs": {}, "sweep": [], "by_dataset": {}, "by_type": {}}
    all_rows: dict[str, list[dict]] = {}
    for cfg in CONFIGS:
        t = time.time()
        rows = run_config(cfg, examples, llm, nli, s.embed_model)
        all_rows[cfg.name] = rows
        m = metrics(rows)
        results["configs"][cfg.name] = m
        results["by_type"][cfg.name] = recall_by_type(rows)
        results["by_dataset"][cfg.name] = {
            d: metrics([r for r in rows if r["dataset"] == d]) for d in datasets
        }
        print(f"{cfg.name:28} F1 {m['f1']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"false-flag {m['false_flag_rate']:.3f}  judge {m['judge_share']:.0%}  ({time.time() - t:.0f}s)")
    for th in SWEEP:
        rows = run_config(Config(f"routed@{th}", [0, 1, 2], th), examples, llm, nli, s.embed_model)
        m = metrics(rows)
        results["sweep"].append({"threshold": th, **m})
        print(f"threshold {th:.2f}: F1 {m['f1']:.3f} judge share {m['judge_share']:.0%} cost/answer ${m['cost_per_answer_usd']:.5f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = render_markdown(results, len(examples))
    (RESULTS / "results.md").write_text(md, encoding="utf-8")
    plots(results)
    update_readme(md)
    print(f"\nwrote {RESULTS / 'results.md'}")


def render_markdown(r: dict, n_answers: int) -> str:
    judge = r["configs"]["LLM judge on every claim"]
    lines = [
        f"Judge: `{r['judge_model']}` · NLI: `{r['nli_model']}` · {n_answers} answers, "
        f"{judge['claims']} claims ({judge['hallucinated_claims']} hallucinated). "
        "Positive class = hallucinated claim; a claim counts as flagged when it is contradicted or unsupported.",
        "",
        "| Configuration | F1 | Precision | Recall | False-flag rate | False-block rate | Sent to judge | p50 / p95 latency per answer | LLM cost per answer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in r["configs"].items():
        lines.append(
            f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['false_flag_rate']:.1%} | {m['false_block_rate']:.1%} | {m['judge_share']:.0%} | "
            f"{m['latency_p50_s']:.1f}s / {m['latency_p95_s']:.1f}s | ${m['cost_per_answer_usd']:.5f} |"
        )
    lines += ["", "**F1 by dataset**", "", "| Configuration | " + " | ".join(r["by_dataset"][next(iter(r["by_dataset"]))]) + " |",
              "|---|" + "---|" * len(r["by_dataset"][next(iter(r["by_dataset"]))])]
    for name, per in r["by_dataset"].items():
        lines.append(f"| {name} | " + " | ".join(f"{m['f1']:.3f}" for m in per.values()) + " |")
    types = sorted({t for per in r["by_type"].values() for t in per})
    lines += ["", "**Recall by hallucination type** (failure analysis)", "",
              "| Configuration | " + " | ".join(f"{t} (n={next(iter(r['by_type'].values())).get(t, {}).get('n', 0)})" for t in types) + " |",
              "|---|" + "---|" * len(types)]
    for name, per in r["by_type"].items():
        lines.append(f"| {name} | " + " | ".join(f"{per[t]['recall']:.2f}" if t in per else "-" for t in types) + " |")
    lines += ["", "**Tier-1 confidence threshold sweep** (tiers 0+1+2)", "",
              "| Threshold | F1 | Sent to judge | Cost per answer |", "|---|---|---|---|"]
    for s in r["sweep"]:
        lines.append(f"| {s['threshold']} | {s['f1']:.3f} | {s['judge_share']:.0%} | ${s['cost_per_answer_usd']:.5f} |")
    lines += ["", "![accuracy vs cost](evals/results/accuracy_vs_cost.png)",
              "![threshold sweep](evals/results/threshold_sweep.png)"]
    return "\n".join(lines)


def plots(r: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ink, muted, accent = "#1f2328", "#6e7781", "#cf222e"
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, m in r["configs"].items():
        x = m["cost_per_answer_usd"] * 1000
        ax.scatter(x, m["f1"], s=60, color=accent if "routed" in name else ink, zorder=3)
        ax.annotate(name, (x, m["f1"]), textcoords="offset points", xytext=(6, 4), fontsize=8, color=ink)
    ax.set_xlabel("LLM cost per 1,000 answers (USD)")
    ax.set_ylabel("F1 (hallucinated claims)")
    ax.set_title("Accuracy vs cost by tier configuration", fontsize=11)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy_vs_cost.png", dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.4, 4))
    th = [s["threshold"] for s in r["sweep"]]
    ax1.plot(th, [s["f1"] for s in r["sweep"]], marker="o", color=ink, label="F1 (routed)")
    judge_f1 = r["configs"]["LLM judge on every claim"]["f1"]
    ax1.axhline(judge_f1, color=muted, ls="--", lw=1, label=f"LLM judge on every claim (F1 {judge_f1:.2f})")
    ax1.set_xlabel("Tier-1 confidence threshold")
    ax1.set_ylabel("F1")
    ax2 = ax1.twinx()
    ax2.bar(th, [s["judge_share"] for s in r["sweep"]], width=0.03, alpha=0.3, color=accent, label="share sent to judge")
    ax2.set_ylabel("Share of claims sent to the judge")
    ax2.set_ylim(0, 1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax1.set_title("Threshold sweep: accuracy vs judge usage", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "threshold_sweep.png", dpi=150)
    plt.close(fig)


def update_readme(md: str) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end)[1]
        readme.write_text(f"{before}{start}\n{md}\n{end}{after}", encoding="utf-8")


if __name__ == "__main__":
    main()
