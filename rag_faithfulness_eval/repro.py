"""T5 reproducibility: recompute metric rows from caches and diff vs recorded summary.

Cheap check: all model scores come from caches (no API spend, no GPU), so any
drift = nondeterminism or code change, not infra.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from .exp1 import binary_metrics, verdicts_to_binary


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _arm_row(arm: str, metric_set: str, rows: list[dict]) -> dict:
    golds = [r["gold_hallucinated"] for r in rows]
    preds = [r["verdict"] for r in rows]
    m = binary_metrics(golds, verdicts_to_binary(preds, metric_set == "neutral_pos"))
    return {"arm": arm, "metric_set": metric_set, **m}


def recompute_exp1(exp1_dir: Path = Path("results/exp1")) -> dict[str, dict]:
    out = {}
    for arm_file in sorted(exp1_dir.glob("arm_*.jsonl")):
        arm = arm_file.stem.replace("arm_", "")
        rows = _load_jsonl(arm_file)
        for tag in ("neutral_neg", "neutral_pos"):
            out[f"{arm}/{tag}"] = _arm_row(arm, tag, rows)
    return out


def recompute_exp2(exp2_dir: Path = Path("results/exp2")) -> dict[str, dict]:
    out = {}
    for arm_file in sorted(exp2_dir.glob("arm_*.jsonl")):
        arm = arm_file.stem.replace("arm_", "")
        rows = _load_jsonl(arm_file)
        for lang in ("de", "it", "all"):
            sel = [r for r in rows if lang == "all" or r["lang"] == lang]
            for tag in ("neutral_neg", "neutral_pos"):
                m = _arm_row(arm, tag, sel)
                out[f"{arm}/{lang}/{tag}"] = m
    return out


def diff_summary(recorded_csv: Path, recomputed: dict, keys=("f1", "accuracy")) -> list[str]:
    """Return mismatch lines (empty = reproducible)."""
    problems = []
    rec: dict[str, dict] = defaultdict(dict)
    for row in csv.DictReader(recorded_csv.read_text().splitlines()):
        arm = row["arm"]
        lang = row.get("lang")
        tag = row["metric_set"]
        key = f"{arm}/{lang}/{tag}" if lang else f"{arm}/{tag}"
        if key not in recomputed:
            continue  # derived rows (response_level, CONTROL, repeats) not recomputable
        rec[key].update({k: float(row[k]) for k in keys})
    for key, want in rec.items():
        if key not in recomputed:
            problems.append(f"{key}: MISSING from recompute")
            continue
        for k in keys:
            if abs(recomputed[key][k] - want[k]) > 1e-3:
                problems.append(f"{key}: {k} {recomputed[key][k]} vs recorded {want[k]}")
    return problems
