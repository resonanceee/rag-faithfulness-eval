"""Noise-adjusted metrics (Task 3): recompute all arms with Exp3 gold-noise flags.

Exp3 sampled judge-vs-gold conflicts per (lang, arm); reviewers flagged the
share where the GOLD was wrong (any_noise). On a flagged case the conflict
disappears: the judge was right. Flip-corrected counts per arm, claim level:

    fp' = fp * (1 - r)      fn' = fn * (1 - r)
    tp' = tp + fn * r       tn' = tn + fp * r        (r = group any_noise rate)

Sampling conflict def = "3-way verdict != binary gold", which is exactly
fp + fn under BOTH neutral conventions, so one correction serves both.

Recorded sanity: r=0 must reproduce the repro check numbers (fails loud else).

CAVEAT (provisional): rates come from pre-adjudication reviewer flags on
71 sampled conflicts. After `rfe exp3-adjudicate` lands real kappa + final
labels, rerun this module for publication numbers.
"""

import csv
import json
from pathlib import Path

from .exp1 import binary_metrics, verdicts_to_binary


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def adjudicated_rates(
    ann_dir: Path = Path("data/annotation"),
    adj_path: Path = Path("results/adjudication_final.jsonl"),
) -> tuple[dict, float, bool]:
    """Final per (lang, arm) gold-noise rates: adjudication decisions for the
    40 disputes, reviewer consensus otherwise. Returns (rates, pooled, final).
    Falls back to pre-adjudication any-flag rates if adjudication incomplete.
    """
    adj: dict = {}
    if adj_path.exists():
        for line in adj_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                adj[r["id"]] = r["gold_ok"]
    groups: dict[tuple, list[list]] = {}
    for lang in ("en", "de", "it"):
        try:
            r1 = {r["id"]: r for r in _load_jsonl(ann_dir / f"task2_{lang}_reviewer1.jsonl")}
            r2 = {r["id"]: r for r in _load_jsonl(ann_dir / f"task2_{lang}_reviewer2.jsonl")}
        except FileNotFoundError:
            continue
        for cid in set(r1) & set(r2):
            arm = r1[cid].get("arm", "?")
            if cid in adj:
                noise = not adj[cid]
            else:
                noise = not (r1[cid].get("gold_ok", True) and r2[cid].get("gold_ok", True))
            groups.setdefault((lang, arm), []).append([1 if noise else 0])
    final = len(adj) >= 40  # full dispute queue decided
    if not final:
        return None, None, False  # type: ignore[return-value]
    rates = {k: sum(x[0] for x in v) / len(v) for k, v in groups.items()}
    all_n = sum(len(v) for v in groups.values())
    pooled = sum(x[0] for v in groups.values() for x in v) / all_n
    return rates, pooled, True


def noise_rates(
    analysis_path: Path = Path("results/exp3_analysis.json"),
) -> tuple[dict, float, bool]:
    """Per (lang, arm) noise rate + pooled fallback. Prefers adjudicated labels;
    falls back to pre-adjudication any_noise rates (47/71) + final=False."""
    rates, pooled, final = adjudicated_rates()
    if final:
        return rates, pooled, True
    a = json.loads(analysis_path.read_text())
    rates = {}
    for key, v in a["noise"].items():
        lang, arm = key.split("/")
        rates[(lang, arm)] = v["any_noise"] / v["n"] if v["n"] else 0.0
    pooled = a["noise_all"]["any"] / a["noise_all"]["n"]
    return rates, pooled, False


def _confusion(rows: list[dict], neutral_pos: bool) -> dict:
    golds = [r["gold_hallucinated"] for r in rows]
    preds = verdicts_to_binary([r["verdict"] for r in rows], neutral_pos)
    return binary_metrics(golds, preds)


def corrected(m: dict, r: float) -> dict:
    tp = m["tp"] + m["fn"] * r
    fn = m["fn"] * (1 - r)
    fp = m["fp"] * (1 - r)
    tn = m["tn"] + m["fp"] * r
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4),
    }


def adjust_experiment(
    exp_dir: Path,
    lang_of_row,
    rates: dict,
    pooled: float,
    headline_tag: dict[str, str] | str,
) -> list[dict]:
    """Adjust one experiment's arm files. headline_tag: fixed tag or {lang: tag}."""
    out: list[dict] = []
    for arm_file in sorted(exp_dir.glob("arm_*.jsonl")):
        arm = arm_file.stem.replace("arm_", "")
        rows = _load_jsonl(arm_file)
        langs = sorted({lang_of_row(r) for r in rows})
        for lang in langs + (["all"] if len(langs) > 1 else []):
            sel = [r for r in rows if lang == "all" or lang_of_row(r) == lang]
            if not sel:
                continue
            for tag in ("neutral_neg", "neutral_pos"):
                m = _confusion(sel, tag == "neutral_pos")
                rate = rates.get((lang if lang != "all" else next(iter(langs)), arm), pooled)
                headline = (
                    headline_tag
                    if isinstance(headline_tag, str)
                    else headline_tag.get(lang, "neutral_pos")
                )
                for r_applied, label in ((0.0, "recorded"), (rate, "noise_corrected")):
                    mm = dict(m) if r_applied == 0 else corrected(m, r_applied)
                    row = {
                        "exp": exp_dir.name,
                        "arm": arm,
                        "lang": lang,
                        "metric_set": tag,
                        "variant": label,
                        "noise_rate": round(r_applied, 4),
                        **mm,
                        "headline": tag == headline,
                    }
                    out.append(row)
    return out


def main(out_path: Path = Path("results/noise_adjusted.csv")) -> list[dict]:
    from .repro import diff_summary, recompute_exp1, recompute_exp2

    # sanity: recomputed recorded numbers must match frozen summaries (0 = truth)
    problems = diff_summary(Path("results/exp1/summary.csv"), recompute_exp1())
    problems += diff_summary(Path("results/exp2/summary.csv"), recompute_exp2())
    assert not problems, f"recorded metrics drifted: {problems}"

    rates, pooled, final = noise_rates()
    rows: list[dict] = []
    for name in ("exp1", "exp4"):
        rows += adjust_experiment(
            Path("results") / name, lambda r: "en", rates, pooled, "neutral_neg"
        )
    rows += adjust_experiment(
        Path("results/exp2"),
        lambda r: r["lang"],
        rates,
        pooled,
        {"de": "neutral_pos", "it": "neutral_pos"},
    )
    for r in rows:
        r["labels"] = "adjudicated" if final else "pre-adjudication-any-flag"
    mode = "ADJUDICATED (final)" if final else "pre-adjudication flags (provisional)"
    print(f"noise labels: {mode}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["headline"] and r["variant"] == "noise_corrected":
            print(
                f"{r['exp']}/{r['arm']}/{r['lang']}: F1 {r['f1']} "
                f"(r={r['noise_rate']}) prec={r['precision']} rec={r['recall']}"
            )
    return rows


if __name__ == "__main__":
    main()
