"""Task 1 / Exp5: judge model sweep.

Same 4,000-claim stratified balanced sample for every candidate model
(2,000 hallucinated + 2,000 faithful, fixed seed, ids joinable to Exp1/4).
Balance oversamples positives (natural rate ~6.7%) for tight recall estimates;
F1 is reported both raw (on the balanced sample) and prevalence-corrected to
the natural rate.

Budget guard: models run sequentially, cost prints per model.
"""

import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .decompose import iter_claims
from .exp1 import Timer, binary_metrics, verdicts_to_binary
from .llm_judge import CostLog, OpenRouterJudge
from .ragtruth import load_ragtruth

MODELS = [
    "inclusionai/ling-3.0-flash",
    "qwen/qwen3.7-flash",
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-4.7-flash",
    "openai/gpt-4.1-nano",
    "bytedance-seed/seed-1.6-flash",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
]
BASELINE = "z-ai/glm-5.3-flash"
NATURAL_RATE = 1259 / 18875  # hallucinated share of Exp4 claim pool
PREMISE_MODE = "query"  # Exp4 protocol: labeled QUESTION+PASSAGES


def build_sample(
    total: int = 4000, seed: int = 0, out_dir: Path = Path("results/exp5")
) -> list[dict]:
    # pool has only ~1,259 positives; take as many as possible (~all), rest negatives
    from .exp1 import premise_of

    rows = [r for r in load_ragtruth("test") if r["quality"] == "good"]
    claims = [
        {**c, "premise": premise_of(c, PREMISE_MODE)}
        for r in rows
        for c in iter_claims(r, 0.2)  # match Exp1/Exp4 alignment threshold
    ]
    pos = [c for c in claims if c["gold_hallucinated"]]
    neg = [c for c in claims if not c["gold_hallucinated"]]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    sample = pos + neg[: max(0, total - len(pos))]
    rng.shuffle(sample)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    k: c[k]
                    for k in ("id", "row_id", "query", "premise", "claim", "gold_hallucinated")
                },
                ensure_ascii=False,
            )
            + "\n"
            for c in sample
        )
    )
    return sample


def load_sample(out_dir: Path = Path("results/exp5")) -> list[dict]:
    p = out_dir / "sample.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def prev_corrected_f1(metrics: dict, rate: float = NATURAL_RATE) -> dict:
    """Prevalence-corrected precision/F1 for the balanced sample back to the
    natural hallucination rate."""
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    prec = recall * rate / max(1e-12, recall * rate + fpr * (1 - rate))
    f1 = 2 * prec * recall / max(1e-12, prec + recall)
    out = dict(metrics)
    out["precision_natural"] = round(prec, 4)
    out["f1_natural"] = round(f1, 4)
    return out


def run_sweep(
    models: list[str] | None = None,
    repeat: int = 2,
    max_workers: int = 8,
    out_dir: Path = Path("results/exp5"),
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = (
        load_sample(out_dir)
        if (out_dir / "sample.jsonl").exists()
        else build_sample(out_dir=out_dir)
    )
    print(
        f"sample: {len(sample)} claims "
        f"({sum(c['gold_hallucinated'] for c in sample)} hallucinated)",
        flush=True,
    )

    glm_ref: dict[str, str] = {}
    ref_path = Path("results/exp4/arm_B.jsonl")
    if ref_path.exists():
        for line in ref_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                glm_ref[r["id"]] = r["verdict"]

    summary: list[dict] = []
    for model in models or MODELS:
        for rep in range(1, repeat + 1):
            log = CostLog()
            judge = OpenRouterJudge(
                model,
                cost_log=log,
                cache_path=out_dir / f"cache_{model.replace('/', '_')}_run{rep}.jsonl",
            )
            verdicts: dict[str, str] = {}
            with Timer() as t:
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {
                        ex.submit(judge.verdict, c["premise"], c["claim"]): c["id"] for c in sample
                    }
                    for i, fut in enumerate(as_completed(futs)):
                        verdicts[futs[fut]] = fut.result()
                        if (i + 1) % 500 == 0:
                            print(
                                f"  {model} run{rep}: {i + 1}/{len(sample)} (${log.usd:.3f})",
                                flush=True,
                            )
            golds = [c["gold_hallucinated"] for c in sample]
            ordered = [verdicts[c["id"]] for c in sample]
            m = prev_corrected_f1(binary_metrics(golds, verdicts_to_binary(ordered, False)))
            agree = (
                sum(verdicts[c["id"]] == glm_ref[c["id"]] for c in sample) / len(sample)
                if glm_ref
                else None
            )
            row = {
                "model": model,
                "run": rep,
                "cost_usd": round(log.usd, 4),
                "calls": log.calls,
                "parse_errors": log.parse_errors,
                "secs": round(t.seconds, 1),
                "agree_glm": agree,
                **m,
            }
            summary.append(row)
            (out_dir / f"verdicts_{model.replace('/', '_')}_run{rep}.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "id": c["id"],
                            "verdict": verdicts[c["id"]],
                            "gold_hallucinated": c["gold_hallucinated"],
                        }
                    )
                    + "\n"
                    for c in sample
                )
            )
            print(
                f"{model} run{rep}: F1={m['f1']} F1nat={m['f1_natural']} "
                f"recall={m['recall']} ${log.usd:.3f}",
                flush=True,
            )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"sweep done -> {out_dir}/summary.csv")
    return summary


if __name__ == "__main__":
    run_sweep()
