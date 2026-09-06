"""Impact check: does distillation answer the project's question?

Compares student vs teacher on the Exp2 benchmark and reports the story:
latency-vs-accuracy tradeoff for a trilingual NLI judge.
"""

import json
from pathlib import Path

from .impact_check import impact_report  # noqa: F401  (re-export)


def write_report(student_dir: Path, out_path: Path = Path("results/distill/impact.json")) -> dict:
    report = impact_report(student_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    s = report
    verdict = (
        f"student is {s['speedup']}x faster than teacher; "
        f"F1 delta de={s['de']['student_f1'] - s['de']['teacher_f1']:+.4f}, "
        f"it={s['it']['student_f1'] - s['it']['teacher_f1']:+.4f}"
    )
    print(verdict)
    return report


if __name__ == "__main__":
    import sys

    write_report(Path(sys.argv[1]))
