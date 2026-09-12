"""Threshold recalibration sweep for the hybrid judge (Arm C), $0.

Recombines cached NLI probs (nli_cache.jsonl) with cached GLM verdicts
(llm_cache_B.jsonl) across an escalation-threshold grid. No API, no GPU:
decomposition is deterministic, caches are joined by deterministic sha256 keys.

The NLI model is miscalibrated (Exp1 ECE ~0.11), so the escalation threshold
is a free hyperparameter — pick it by grid sweep on metrics, not by raw
confidence. This module produces that table, for both Exp1 (plain premise)
and Exp4 (query-labeled premise, the production protocol).

Sanity check (fails the run if caches/code drifted):
  t=0.0  -> all claims escalated -> must match recorded Arm B metrics
  t>=1.0 -> nothing escalated    -> must match recorded Arm A metrics
"""

import csv
import json
from pathlib import Path

from .decompose import iter_claims
from .exp1 import binary_metrics, nli_verdict, premise_of, verdicts_to_binary
from .judge import DEFAULT_CHECKPOINT, _cache_key
from .llm_judge import GLM_FLASH, _verdict_key
from .ragtruth import load_ragtruth

# coarse 0..0.95, fine 0.95..1.0 — conf mass concentrates near 1.0
GRID = [round(0.05 * i, 2) for i in range(20)] + [0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _claims_for(exp_dir: Path) -> tuple[list[dict], str]:
    """Rebuild claims exactly as run_exp1 did; return (claims, premise_mode)."""
    query = exp_dir.name == "exp4"
    all_rows = load_ragtruth("test")
    rows = [r for r in all_rows if r["quality"] == "good"] if query else all_rows
    mode = "query" if query else "plain"
    claims = [
        {**c, "premise": premise_of(c, mode)}
        for r in rows
        for c in iter_claims(r, 0.2)
    ]
    return claims, mode


def load_caches(
    claims: list[dict], exp_dir: Path, llm_dir: Path | None, llm_model: str
) -> tuple[dict, dict, int]:
    """Join claims to NLI probs (exp_dir) and LLM verdicts (llm_dir)."""
    nli_path = exp_dir / "nli_cache.jsonl"
    llm_dir = llm_dir or exp_dir
    nli_raw = {r["key"]: r["probs"] for r in _load_jsonl(nli_path)}
    llm_raw = {r["key"]: r["verdict"] for r in _load_jsonl(llm_dir / "llm_cache_B.jsonl")}
    nli, llm, missing = {}, {}, 0
    for c in claims:
        nk = _cache_key(DEFAULT_CHECKPOINT, c["premise"], c["claim"])
        lk = _verdict_key(llm_model, c["premise"], c["claim"])
        if nk in nli_raw and lk in llm_raw:
            nli[c["id"]] = nli_raw[nk]
            llm[c["id"]] = llm_raw[lk]
        else:
            missing += 1
    return nli, llm, missing


def sweep(
    exp_dir: Path,
    out_dir: Path,
    llm_dir: Path | None = None,
    llm_model: str = GLM_FLASH,
    tag: str | None = None,
) -> list[dict]:
    claims, _ = _claims_for(exp_dir)
    llm_dir = llm_dir or exp_dir
    nli, llm, missing = load_caches(claims, exp_dir, llm_dir, llm_model)
    full = len(claims)
    claims = [c for c in claims if c["id"] in nli]
    golds = [c["gold_hallucinated"] for c in claims]
    row_gold = sorted({c["row_id"] for c in claims})
    # row gold: a row is hallucinated iff any aligned claim is gold-hallucinated
    rg = {
        rid: any(c["gold_hallucinated"] for c in claims if c["row_id"] == rid)
        for rid in row_gold
    }

    summaries = _load_jsonl(exp_dir / "arm_A.jsonl") if (exp_dir / "arm_A.jsonl").exists() else None
    arm_b = _load_jsonl(llm_dir / "arm_B.jsonl")

    out_rows: list[dict] = []
    for t in GRID:
        verdicts = {}
        escalated = 0
        for c in claims:
            v, conf = nli_verdict(nli[c["id"]])
            if conf < t:
                v = llm[c["id"]]
                escalated += 1
            verdicts[c["id"]] = v
        ordered = [verdicts[c["id"]] for c in claims]
        m = binary_metrics(golds, verdicts_to_binary(ordered, False))
        row_pred = {rid: False for rid in rg}
        for c in claims:
            if verdicts[c["id"]] == "unfaithful":
                row_pred[c["row_id"]] = True
        mr = binary_metrics(list(rg.values()), list(row_pred.values()))
        out_rows.append(
            {
                "threshold": t,
                "llm_share": round(escalated / len(claims), 4),
                "claim_f1_nn": m["f1"],
                "claim_recall_nn": m["recall"],
                "claim_precision_nn": m["precision"],
                "claim_acc_nn": m["accuracy"],
                "row_f1": mr["f1"],
                "row_recall": mr["recall"],
                "row_precision": mr["precision"],
            }
        )

    # sanity: endpoints must reproduce recorded arms A (t=0 -> all NLI) and
    # B (all GLM from cache -> recorded arm_B)
    rec_b = {r["id"]: r["verdict"] for r in arm_b}
    cache_b = binary_metrics(golds, verdicts_to_binary([llm[c["id"]] for c in claims], False))
    rec_b_m = binary_metrics(golds, verdicts_to_binary([rec_b[c["id"]] for c in claims], False))
    assert abs(cache_b["f1"] - rec_b_m["f1"]) < 1e-3, (
        f"{exp_dir}: cached GLM verdicts do not reproduce arm_B (join broke?) "
        f"— {missing} missing of {full}"
    )
    if summaries:
        a_end = next(r for r in out_rows if r["threshold"] == 0.0)
        rec = {r["id"]: r["verdict"] for r in _load_jsonl(exp_dir / "arm_A.jsonl")}
        rec_m = binary_metrics(golds, verdicts_to_binary([rec[c["id"]] for c in claims], False))
        assert abs(rec_m["f1"] - a_end["claim_f1_nn"]) < 1e-3, \
            f"{exp_dir}: t=0.0 endpoint fails to reproduce Arm A"

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{tag or exp_dir.name}_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def best_by_f1(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r["claim_f1_nn"])


def sweep_sample_judge(
    verdicts_path: Path,
    label: str,
    out_dir: Path,
    sample_path: Path = Path("results/exp5/sample.jsonl"),
) -> list[dict]:
    """Hybrid sweep on the Exp5 4k sample for any judge with cached verdicts.

    Joins exp5 sample claims to exp4 NLI probs; reports prevalence-corrected
    F1 (same protocol as Exp5) so numbers compare 1:1 with the model sweep.
    """
    from .exp5 import NATURAL_RATE, prev_corrected_f1

    sample = _load_jsonl(sample_path)
    nli_raw = {r["key"]: r["probs"] for r in _load_jsonl(Path("results/exp4/nli_cache.jsonl"))}
    judge_raw = {
        r["id"]: r["verdict"]
        for r in _load_jsonl(verdicts_path)
    }
    nli, jv = {}, {}
    for c in sample:
        nk = _cache_key(DEFAULT_CHECKPOINT, c["premise"], c["claim"])
        if nk in nli_raw and c["id"] in judge_raw:
            nli[c["id"]] = nli_raw[nk]
            jv[c["id"]] = judge_raw[c["id"]]
    claims = [c for c in sample if c["id"] in nli]
    golds = [c["gold_hallucinated"] for c in claims]

    out_rows: list[dict] = []
    for t in GRID:
        verdicts, escalated = {}, 0
        for c in claims:
            v, conf = nli_verdict(nli[c["id"]])
            if conf < t:
                v = jv[c["id"]]
                escalated += 1
            verdicts[c["id"]] = v
        bin_m = binary_metrics(
            golds, verdicts_to_binary([verdicts[c["id"]] for c in claims], False)
        )
        m = prev_corrected_f1(bin_m, NATURAL_RATE)
        out_rows.append(
            {
                "threshold": t,
                "llm_share": round(escalated / len(claims), 4),
                "claim_f1": m["f1"],
                "f1_natural": m["f1_natural"],
                "recall": m["recall"],
                "precision_natural": m["precision_natural"],
            }
        )
    # sanity: t=1 endpoint (all-judge) must reproduce the recorded pure-judge F1
    pure = binary_metrics(golds, verdicts_to_binary([jv[c["id"]] for c in claims], False))
    end = out_rows[-1]
    assert abs(pure["f1"] - end["claim_f1"]) < 1e-3, f"{label}: t=1.0 endpoint mismatch"

    with (out_dir / f"sample_hybrid_{label}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    return out_rows


def main(out_dir: Path = Path("results/threshold_sweep")) -> dict:
    report = {"missing_join": {}}
    for name in ("exp1", "exp4"):
        rows = sweep(Path("results") / name, out_dir)
        best = best_by_f1(rows)
        report[name] = {"best": best, "sweep": rows}
        pure_nli = rows[0]  # t=0: nothing escalated
        pure_llm = rows[-1]  # t=1: ~everything escalated
        print(f"[{name}] best t={best['threshold']} F1={best['claim_f1_nn']} "
              f"(llm_share={best['llm_share']})")
        print(f"[{name}] pure GLM F1={pure_llm['claim_f1_nn']} cost-share=1.0")
        print(f"[{name}] pure NLI F1={pure_nli['claim_f1_nn']} cost-share=0.0")

    # full-set hybrid@ling (post-T6): exp4 NLI probs + exp4_ling verdicts
    ling_dir = Path("results/exp4_ling")
    if (ling_dir / "llm_cache_B.jsonl").exists():
        rows = sweep(
            Path("results/exp4"),
            out_dir,
            llm_dir=ling_dir,
            llm_model="inclusionai/ling-3.0-flash",
            tag="exp4_hybrid_ling",
        )
        best = best_by_f1(rows)
        report["exp4_hybrid_ling"] = {"best": best, "sweep": rows}
        print(f"[exp4 hybrid@ling] best t={best['threshold']} F1={best['claim_f1_nn']} "
              f"(llm_share={best['llm_share']})")
        print(f"[exp4 hybrid@ling] pure ling F1={rows[-1]['claim_f1_nn']} cost-share=1.0")

    for label, path in (
        ("ling", "results/exp5/verdicts_inclusionai_ling-3.0-flash_run1.jsonl"),
        ("glm", "results/exp5/verdicts_z-ai_glm-5.3-flash_run1.jsonl"),
    ):
        p = Path(path)
        if not p.exists():
            print(f"[{label}] missing {p}, skipped")
            continue
        rows = sweep_sample_judge(p, label, out_dir)
        best = max(rows, key=lambda r: r["f1_natural"])
        report[f"sample_hybrid_{label}"] = {"best": best, "sweep": rows}
        print(f"[hybrid@{label}] best t={best['threshold']} F1nat={best['f1_natural']} "
              f"llm_share={best['llm_share']}")
        print(f"[hybrid@{label}] pure F1nat={rows[-1]['f1_natural']} cost-share=1.0")

    (out_dir / "threshold_sweep.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
