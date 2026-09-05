"""Experiment 1 judge arms + metrics.

Verdicts everywhere: "faithful" | "unfaithful" | "unverifiable".
Judge A (NLI): argmax over entailment/neutral/contradiction.
Judge B (LLM): OpenRouter chat model.
Judge C (hybrid): NLI if top prob >= threshold, else LLM arbitration.
Judge D (baseline): no decomposition — judge full output vs context, one pass.
"""

import csv
import itertools
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .decompose import iter_claims
from .judge import CachedJudge, NLIJudge
from .llm_judge import GLM_FLASH, CostLog, OpenRouterJudge
from .ragtruth import load_ragtruth

NLI_VERDICT = {
    "entailment": "faithful",
    "neutral": "unverifiable",
    "contradiction": "unfaithful",
}


def nli_verdict(probs: dict) -> tuple[str, float]:
    top = max(probs, key=lambda k: probs[k])
    return NLI_VERDICT[top], probs[top]


class ArmC:
    """Hybrid: NLI first pass, LLM arbitration below confidence threshold."""

    def __init__(self, nli, llm, threshold: float = 0.85):  # duck-typed judges
        self.nli = nli
        self.llm = llm
        self.threshold = threshold
        self.escalated = 0
        self.total = 0

    def verdict(self, probs: dict, context: str, claim: str) -> str:
        self.total += 1
        v, conf = nli_verdict(probs)
        if conf >= self.threshold:
            return v
        self.escalated += 1
        return self.llm.verdict(context, claim)

    @property
    def proxy_ratio(self) -> float:
        return self.escalated / max(1, self.total)


def binary_metrics(golds: list[bool], preds_hallucinated: list[bool]) -> dict:
    """Hallucinated = positive class."""
    tp = sum(p and g for g, p in zip(golds, preds_hallucinated, strict=True))
    fp = sum(p and not g for g, p in zip(golds, preds_hallucinated, strict=True))
    fn = sum(not p and g for g, p in zip(golds, preds_hallucinated, strict=True))
    tn = sum(not p and not g for g, p in zip(golds, preds_hallucinated, strict=True))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / max(1, len(golds)), 4),
    }


def verdicts_to_binary(verdicts: list[str], neutral_as_hallucinated: bool) -> list[bool]:
    return [
        v == "unfaithful" or (neutral_as_hallucinated and v == "unverifiable") for v in verdicts
    ]


def calibration_bins(confs: list[float], correct: list[bool], n_bins: int = 10) -> dict:
    """Accuracy + mean confidence per probability bin (calibration plot data)."""
    bins = [{"lo": i / n_bins, "n": 0, "acc_sum": 0.0, "conf_sum": 0.0} for i in range(n_bins)]
    for c, ok in zip(confs, correct, strict=True):
        b = bins[min(n_bins - 1, int(c * n_bins))]
        b["n"] += 1
        b["acc_sum"] += float(ok)
        b["conf_sum"] += c
    ece = sum(
        (b["n"] / max(1, len(confs))) * abs(b["acc_sum"] / b["n"] - b["conf_sum"] / b["n"])
        for b in bins
        if b["n"]
    )
    out = [
        {
            **b,
            "accuracy": round(b["acc_sum"] / b["n"], 4),
            "mean_conf": round(b["conf_sum"] / b["n"], 4),
        }
        if b["n"]
        else {**b, "accuracy": None, "mean_conf": None}
        for b in bins
    ]
    for b in out:
        b.pop("acc_sum")
        b.pop("conf_sum")
    return {"bins": out, "ece": round(ece, 4)}


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.start

    @staticmethod
    def fmt(seconds: float) -> str:
        return f"{seconds:.1f}s" if seconds < 120 else f"{seconds / 60:.1f}min"


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))


def run_exp1(
    split: str = "test",
    limit: int | None = None,
    arms: str = "ABCD",
    repeat: int = 1,
    llm_model: str = GLM_FLASH,
    out_dir: Path = Path("results/exp1"),
    threshold: float = 0.85,
    batch_size: int = 32,
    align_threshold: float = 0.2,  # RAGTruth spans are sub-sentence; 0.5 starves gold
    max_workers: int = 8,
) -> dict:
    """Run Experiment 1. Returns summary dict; writes JSONL/CSV/JSON into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_log = CostLog()

    rows = load_ragtruth(split)
    if limit:
        rows = rows[:limit]
    claims = list(itertools.chain.from_iterable(iter_claims(r, align_threshold) for r in rows))
    gold_by_claim = {c["id"]: c["gold_hallucinated"] for c in claims}
    row_gold = {r["id"]: bool(r["spans"]) for r in rows}
    print(
        f"{len(rows)} rows -> {len(claims)} claims "
        f"(hallucinated: {sum(gold_by_claim.values())} claims, "
        f"{sum(row_gold.values())} rows)"
    )

    nli_scores: dict[str, dict] = {}
    if "A" in arms or "C" in arms or "D" in arms:
        with Timer() as t_nli:
            nli = CachedJudge(NLIJudge(), out_dir / "nli_cache.jsonl")
            all_scores = nli.score_batch([(c["context"], c["claim"]) for c in claims], batch_size)
            nli_scores = dict(zip((c["id"] for c in claims), all_scores, strict=True))
        print(f"NLI scoring done in {Timer.fmt(t_nli.seconds)}")
    else:
        t_nli = None

    verdicts: dict[str, dict[str, str]] = {}

    if "A" in arms:
        verdicts["A"] = {cid: nli_verdict(p)[0] for cid, p in nli_scores.items()}
    with Timer() as t_b:
        if "B" in arms:
            for rep in range(repeat):
                key = "B" if rep == 0 else f"B_run{rep + 1}"
                # per-repeat cache: repeat runs must NOT hit each other's cache,
                # else the repeatability measurement is definitionally 1.0
                llm = OpenRouterJudge(
                    llm_model,
                    cost_log=cost_log,
                    cache_path=out_dir / f"llm_cache_{key}.jsonl",
                )
                verdicts[key] = {}
                executor = ThreadPoolExecutor(max_workers=max_workers)
                futures = {
                    executor.submit(llm.verdict, c["context"], c["claim"]): c["id"] for c in claims
                }
                for i, fut in enumerate(as_completed(futures)):
                    verdicts[key][futures[fut]] = fut.result()
                    if (i + 1) % 200 == 0:
                        print(f"  {key}: {i + 1}/{len(claims)} (${cost_log.usd:.3f})", flush=True)
                executor.shutdown()

    with Timer() as t_c:
        if "C" in arms:
            # arbitration shares B run-1's cache: same model, same claims
            llm = OpenRouterJudge(
                llm_model, cost_log=cost_log, cache_path=out_dir / "llm_cache_B.jsonl"
            )
            hybrid = ArmC(nli, llm, threshold)  # type: ignore[possibly-undefined]
            verdicts["C"] = {
                c["id"]: hybrid.verdict(nli_scores[c["id"]], c["context"], c["claim"])
                for c in claims
            }
            print(f"Arm C proxy-ratio: {hybrid.proxy_ratio:.3f}")

    row_verdicts_d: dict[str, str] = {}
    with Timer() as t_d:
        if "D" in arms:
            d_scores = nli.score_batch([(r["context"], r["output"]) for r in rows], batch_size)  # type: ignore[possibly-undefined]
            row_verdicts_d = dict(
                zip((r["id"] for r in rows), (nli_verdict(s)[0] for s in d_scores), strict=True)
            )

    summary, timings = (
        [],
        {
            "A_nli_scoring": Timer.fmt(t_nli.seconds) if t_nli else "-",
            "B_llm": Timer.fmt(t_b.seconds),
            "C_hybrid": Timer.fmt(t_c.seconds),
            "D_baseline": Timer.fmt(t_d.seconds),
        },
    )
    for arm, vs in verdicts.items():
        (out_dir / f"arm_{arm}.jsonl").write_text(
            "".join(
                json.dumps({"id": cid, "verdict": v, "gold_hallucinated": gold_by_claim[cid]})
                + "\n"
                for cid, v in vs.items()
            )
        )
        ordered = [vs[c["id"]] for c in claims]
        for tag, neutral_pos in (("neutral_neg", False), ("neutral_pos", True)):
            m = binary_metrics(
                [gold_by_claim[c["id"]] for c in claims],
                verdicts_to_binary(ordered, neutral_pos),
            )
            summary.append({"arm": arm, "metric_set": tag, **m})
        # response-level aggregation (fair comparison vs arm D)
        row_pred = {rid: False for rid in row_gold}
        for c in claims:
            if vs[c["id"]] == "unfaithful":
                row_pred[c["row_id"]] = True
        m = binary_metrics(list(row_gold.values()), list(row_pred.values()))
        summary.append({"arm": arm, "metric_set": "response_level", **m})

    if "D" in arms:
        m = binary_metrics(
            list(row_gold.values()),
            verdicts_to_binary([row_verdicts_d[rid] for rid in row_gold], False),
        )
        summary.append({"arm": "D", "metric_set": "response_level", **m})

    # no-alignment control: gold randomly reshuffled -> metrics should sit at chance
    rng = random.Random(42)
    shuffled = list(gold_by_claim.values())
    rng.shuffle(shuffled)
    m = binary_metrics(
        shuffled,
        verdicts_to_binary(
            [verdicts.get("A", {c["id"]: "faithful" for c in claims})[c["id"]] for c in claims],
            False,
        ),
    )
    summary.append({"arm": "CONTROL", "metric_set": "neutral_neg", **m})

    # calibration from arm A NLI confidence
    if "A" in arms:
        confs = [max(nli_scores[c["id"]].values()) for c in claims]
        correct = [
            (verdicts["A"][c["id"]] == "unfaithful") == c["gold_hallucinated"] for c in claims
        ]
        cal = calibration_bins(confs, correct)
        (out_dir / "calibration.json").write_text(json.dumps(cal, indent=2))

    if sum(1 for k in verdicts if k.startswith("B")) > 1:
        b_keys = sorted(k for k in verdicts if k.startswith("B"))
        base, other = verdicts[b_keys[0]], verdicts[b_keys[1]]
        same = sum(base[cid] == other[cid] for cid in base)
        print(f"repeatability: {same}/{len(base)} identical ({same / len(base):.3f})")
        summary.append(
            {
                "arm": "B_REPEAT",
                "metric_set": "agreement",
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "accuracy": round(same / len(base), 4),
            }
        )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arm",
                "metric_set",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ],
        )
        w.writeheader()
        w.writerows(summary)
    (out_dir / "cost.json").write_text(
        json.dumps(
            {
                "calls": cost_log.calls,
                "in_tokens": cost_log.in_tokens,
                "out_tokens": cost_log.out_tokens,
                "usd": round(cost_log.usd, 4),
                "by_model": cost_log.by_model,
                "timings": timings,
            },
            indent=2,
        )
    )
    print(f"done. cost ${cost_log.usd:.4f}, wrote {out_dir}/")
    return {"summary": summary, "cost_usd": cost_log.usd}
