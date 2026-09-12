"""Figures for the README: consolidated plots across all experiments.

Reads results/exp1|exp2|exp4 summary.csv + cost.json + exp3_analysis.json,
writes docs/figures/*.png. (docs/final_report.md was retired: README.md is
the document of record — this module only refreshes figures.)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("docs/figures")
ARM_LABELS = {"A": "NLI", "B": "LLM (glm-5.3-flash)", "C": "hybrid", "D": "no-decomp"}


def _rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fig_exp1_arms(out=FIG / "exp1_arms.png"):
    rows = [
        r
        for r in _rows(Path("results/exp1/summary.csv"))
        if r["arm"] in "ABCD" and r["metric_set"] in ("neutral_neg", "response_level")
    ]
    claim = {r["arm"]: float(r["f1"]) for r in rows if r["metric_set"] == "neutral_neg"}
    resp = {r["arm"]: float(r["f1"]) for r in rows if r["metric_set"] == "response_level"}
    x = range(len(claim))
    arms = list(claim)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [claim[a] for a in arms], w, label="claim-level F1")
    ax.bar([i + w / 2 for i in x], [resp.get(a, 0) for a in arms], w, label="response-level F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{a}\n{ARM_LABELS[a]}" for a in arms])
    ax.set_ylabel("F1 (hallucinated)")
    ax.set_title("Exp1 RAGTruth: LLM judge beats NLI; decomposition is load-bearing")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_calibration(out=FIG / "exp1_calibration.png"):
    cal = json.load(open("results/exp1/calibration.json"))
    bins = [b for b in cal["bins"] if b["n"]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.4, label="perfect")
    ax.errorbar(
        [b["mean_conf"] for b in bins],
        [b["accuracy"] for b in bins],
        marker="o",
        label=f"NLI (ECE={cal['ece']})",
    )
    ax.set_xlabel("NLI top-prob (confidence)")
    ax.set_ylabel("accuracy")
    ax.set_title("NLI calibration: confident and wrong (0.8-0.9 bin ≈ 0.85)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_exp2_langs(out=FIG / "exp2_langs.png"):
    rows = [
        r
        for r in _rows(Path("results/exp2/summary.csv"))
        if r["metric_set"] == "neutral_pos" and r["lang"] in ("de", "it")
    ]
    data = {}
    for r in rows:
        data.setdefault(r["arm"], {})[r["lang"]] = float(r["f1"])
    arms = sorted(data)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    x = range(len(arms))
    for j, lang in enumerate(("de", "it")):
        ax.bar(
            [i + (j - 0.5) * w for i in x],
            [data[a].get(lang, 0) for a in arms],
            w,
            label=lang.upper(),
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms)
    ax.set_ylabel("F1 (neutral_pos)")
    ax.set_title("Exp2 cross-lingual (v2 data): DE lags IT; hybrid best")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_exp4_delta(out=FIG / "exp4_query_delta.png"):
    a = {
        r["arm"]: float(r["precision"])
        for r in _rows(Path("results/exp1/summary.csv"))
        if r["metric_set"] == "neutral_neg" and r["arm"] in "ABC"
    }
    b = {
        r["arm"]: float(r["precision"])
        for r in _rows(Path("results/exp4/summary.csv"))
        if r["metric_set"] == "neutral_neg" and r["arm"] in "ABC"
    }
    ra = {
        r["arm"]: float(r["recall"])
        for r in _rows(Path("results/exp1/summary.csv"))
        if r["metric_set"] == "neutral_neg" and r["arm"] in "ABC"
    }
    rb = {
        r["arm"]: float(r["recall"])
        for r in _rows(Path("results/exp4/summary.csv"))
        if r["metric_set"] == "neutral_neg" and r["arm"] in "ABC"
    }
    arms = sorted(a)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([a[arm] for arm in arms], [ra[arm] for arm in arms], label="Exp1 (no query)")
    ax.scatter(
        [b[arm] for arm in arms], [rb[arm] for arm in arms], marker="s", label="Exp4 (query)"
    )
    for arm in arms:
        ax.annotate(f" {arm}", (b[arm], rb[arm]))
        ax.plot([a[arm], b[arm]], [ra[arm], rb[arm]], "k:", alpha=0.4)
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.set_title("Query-in-context: B recall +5.7pts, aggregates stable")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_threshold_pareto(out=FIG / "exp_threshold_pareto.png"):
    """Hybrid escalation curve (Exp4 protocol, full set): quality vs LLM cost share."""
    rows = _rows(Path("results/threshold_sweep/exp4_sweep.csv"))
    share = [float(r["llm_share"]) for r in rows]
    f1 = [float(r["claim_f1_nn"]) for r in rows]
    t = [float(r["threshold"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sc = ax.scatter(share, f1, c=t, cmap="viridis", s=28)
    ax.plot(share, f1, alpha=0.3)
    fig.colorbar(sc, label="NLI escalation threshold")
    for q in (0.85, 0.95, 0.99, 1.0):
        r = min(rows, key=lambda r: abs(float(r["threshold"]) - q))
        ax.annotate(f" t={r['threshold']}", (float(r["llm_share"]), float(r["claim_f1_nn"])))
    ax.set_xlabel("share of claims sent to LLM (≈ cost share)")
    ax.set_ylabel("claim F1 (neutral_neg)")
    ax.set_title("Hybrid never dominates pure GLM; 68% cost buys only 84% F1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_exp3_noise(out=FIG / "exp3_noise.png"):
    a3 = json.load(open("results/exp3_analysis.json"))
    labels, confirmed, flagged = [], [], []
    for key, v in sorted(a3["noise"].items()):
        labels.append(key)
        confirmed.append(v["both_noise"] / v["n"])
        flagged.append(v["any_noise"] / v["n"])
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - w / 2 for i in x], flagged, w, label="flagged by either reviewer")
    ax.bar([i + w / 2 for i in x], confirmed, w, label="confirmed by both")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("share of judge-vs-gold conflicts")
    ax.set_title("Exp3: most judge 'errors' are gold annotation noise")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_exp1_arms()
    fig_calibration()
    fig_exp2_langs()
    fig_exp4_delta()
    fig_exp3_noise()
    if Path("results/threshold_sweep/exp4_sweep.csv").exists():
        fig_threshold_pareto()
