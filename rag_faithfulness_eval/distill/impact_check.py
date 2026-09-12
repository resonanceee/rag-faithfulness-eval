"""Impact check: distilled student vs teacher on the same benchmark slices.

Compares: per-language agreement + accuracy on DE/IT synthetic set, latency,
model size. Used by T6 merge decision.
"""

import json
import time
from pathlib import Path

from ..exp1 import binary_metrics, nli_verdict, verdicts_to_binary
from ..exp2 import load_deit_samples
from ..judge import NLIJudge


def _score_all(judge: NLIJudge, samples: list[dict]) -> tuple[dict, float]:
    t0 = time.perf_counter()
    scores = judge.score_batch([(s["context"], s["claim"]) for s in samples])
    return dict(zip((s["id"] for s in samples), scores, strict=True)), (time.perf_counter() - t0)


def impact_report(
    student_dir: Path,
    samples_path: Path = Path("data/samples.jsonl"),
    teacher_ckpt: str | None = None,
) -> dict:
    samples = load_deit_samples(samples_path)
    kwargs = {} if teacher_ckpt is None else {"checkpoint": teacher_ckpt}
    teacher = NLIJudge(**kwargs)
    student = NLIJudge(checkpoint=str(student_dir))

    t_scores, t_secs = _score_all(teacher, samples)
    s_scores, s_secs = _score_all(student, samples)

    report: dict = {
        "teacher_secs": round(t_secs, 1),
        "student_secs": round(s_secs, 1),
        "speedup": round(t_secs / max(s_secs, 1e-9), 2),
    }
    for lang in ("de", "it", "all"):
        sel = [s for s in samples if lang == "all" or s["lang"] == lang]
        agree = sum(
            nli_verdict(t_scores[s["id"]])[0] == nli_verdict(s_scores[s["id"]])[0] for s in sel
        ) / len(sel)
        row = {"teacher_student_agreement": round(agree, 4)}
        for name, scores in (("teacher", t_scores), ("student", s_scores)):
            m = binary_metrics(
                [s["gold_hallucinated"] for s in sel],
                verdicts_to_binary([nli_verdict(scores[s["id"]])[0] for s in sel], False),
            )
            row[f"{name}_f1"] = m["f1"]
        report[lang] = row
    n_params = sum(p.numel() for p in student.model.parameters())
    t_params = sum(p.numel() for p in teacher.model.parameters())
    report["student_params"] = n_params
    report["teacher_params"] = t_params
    report["filter"] = {
        lang: {
            name: filter_report({s["id"]: scores[s["id"]] for s in samples}, samples, lang)
            for name, scores in (("student", s_scores), ("teacher", t_scores))
        }
        for lang in ("de", "it", "all")
    }
    return report


def filter_report(scores: dict, samples: list[dict], lang: str) -> dict:
    """High-recall filter framing (T5 retry goal): judge as cheap first stage —
    pass-through claims with p(entailment) high, escalate the rest to an LLM.

    Sweep: at each retention share, recall of gold-hallucinated claims among
    the ESCALATED pool. Filter is useful iff most hallucinations land in a
    small escalated share. Reports recall@shares + min share for recall>=0.95.
    """
    sel = [s for s in samples if lang == "all" or s["lang"] == lang]
    ranked = sorted(sel, key=lambda s: scores[s["id"]]["entailment"])  # least entailed first
    n_pos = sum(s["gold_hallucinated"] for s in sel)
    curve = {}
    for share in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        k = max(1, round(len(ranked) * share))
        escalated = ranked[:k]
        rec = sum(s["gold_hallucinated"] for s in escalated) / max(1, n_pos)
        curve[str(share)] = {"recall": round(rec, 4), "escalated": k}
    best = None
    for k in range(1, len(ranked) + 1):
        rec = sum(s["gold_hallucinated"] for s in ranked[:k]) / max(1, n_pos)
        if rec >= 0.95:
            best = {"share": round(k / len(ranked), 4), "recall": round(rec, 4)}
            break
    return {"curve": curve, "share_for_recall_0.95": best}


if __name__ == "__main__":
    import sys

    print(json.dumps(impact_report(Path(sys.argv[1])), indent=2))
