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
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(impact_report(Path(sys.argv[1])), indent=2))
