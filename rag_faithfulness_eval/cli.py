"""CLI: rfe build-data | validate | score"""

import argparse
import json
from pathlib import Path

from .judge import DEFAULT_CHECKPOINT, MULTILINGUAL_CHECKPOINT, CachedJudge, NLIJudge, is_faithful
from .pipeline import build_samples, read_jsonl, report, write_jsonl
from .schema import validate_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rfe")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data", help="build balanced EN/DE/IT dataset JSONL")
    b.add_argument("--out", type=Path, default=Path("data/samples.jsonl"))
    b.add_argument("--n-per-lang", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)

    v = sub.add_parser("validate", help="schema-check a dataset JSONL")
    v.add_argument("--input", type=Path, required=True)

    s = sub.add_parser("score", help="score samples with an NLI judge")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument(
        "--judge-checkpoint",
        default=DEFAULT_CHECKPOINT,
        choices=[DEFAULT_CHECKPOINT, MULTILINGUAL_CHECKPOINT],
    )
    s.add_argument("--cache", type=Path, default=Path("results/judge_cache.jsonl"))
    s.add_argument("--device", default=None)

    e = sub.add_parser("exp1", help="Experiment 1: judge calibration on RAGTruth")
    e.add_argument("--split", default="test")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--arms", default="ABCD")
    e.add_argument("--repeat", type=int, default=1, help="LLM judge repeat runs (repeatability)")
    e.add_argument("--llm-model", default="z-ai/glm-5.3-flash")
    e.add_argument("--threshold", type=float, default=0.85)
    e.add_argument("--align-threshold", type=float, default=0.2)
    e.add_argument("--out", type=Path, default=Path("results/exp1"))

    x = sub.add_parser("exp2", help="Experiment 2: cross-lingual transfer DE/IT")
    x.add_argument("--arms", default="ABCD")
    x.add_argument("--n-mix", type=int, default=200)
    x.add_argument("--threshold", type=float, default=0.85)
    x.add_argument("--llm-model", default="z-ai/glm-5.3-flash")
    x.add_argument("--out", type=Path, default=Path("results/exp2"))

    s3 = sub.add_parser("exp3-sample", help="sample disagreement cases for annotation")
    s3.add_argument("--n-exp1", type=int, default=60)
    s3.add_argument("--n-exp2", type=int, default=30)
    s3.add_argument("--out", type=Path, default=Path("data/annotation"))

    k3 = sub.add_parser("exp3-kappa", help="Cohen's kappa between two reviewer files")
    k3.add_argument("file_a", type=Path)
    k3.add_argument("file_b", type=Path)
    k3.add_argument("--field", default="hallucination_type")

    rc = sub.add_parser("exp3-recheck", help="Exp3: build shuffled 10% recheck files")
    rc.add_argument("--ann-dir", type=Path, default=Path("data/annotation"))
    rc.add_argument("--fraction", type=float, default=0.1)

    sa = sub.add_parser("exp3-self-agreement", help="Exp3: self-agreement main vs recheck")
    sa.add_argument("main_file", type=Path)
    sa.add_argument("recheck_file", type=Path)
    sa.add_argument("--field", default="hallucination_type")

    an = sub.add_parser("exp3-annotate", help="Exp3: interactive one-by-one annotation")
    an.add_argument("--reviewer", type=int, required=True, choices=[1, 2])
    an.add_argument("file", type=Path, help="task file, e.g. data/annotation/task2_de.jsonl")

    sub.add_parser("repro", help="T5: recompute metrics from caches and diff recorded results")

    args = p.parse_args(argv)

    if args.cmd == "build-data":
        samples = build_samples(n_per_lang=args.n_per_lang, seed=args.seed)
        write_jsonl(samples, args.out)
        print(json.dumps(report(samples), indent=2, ensure_ascii=False))
        print(f"wrote {len(samples)} samples -> {args.out}")
        return 0

    if args.cmd == "validate":
        rows = read_jsonl(args.input)
        errors = validate_rows(rows)
        for e in errors:
            print(e)
        print(f"{len(rows) - len(errors)}/{len(rows)} rows valid")
        return 1 if errors else 0

    if args.cmd == "score":
        rows = read_jsonl(args.input)
        judge = CachedJudge(NLIJudge(args.judge_checkpoint, device=args.device), args.cache)
        probs = judge.score_batch([(r["context"], r["claim"]) for r in rows])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for r, pr in zip(rows, probs, strict=True):
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "checkpoint": judge.checkpoint,
                            "probs": pr,
                            "verdict_faithful": is_faithful(pr),
                            "gold_faithful": r["faithful"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        agree = sum(is_faithful(pr) == r["faithful"] for r, pr in zip(rows, probs, strict=True))
        print(f"scored {len(rows)} samples, verdict agreement {agree}/{len(rows)}")
        return 0

    if args.cmd == "exp1":
        from .exp1 import run_exp1

        run_exp1(
            split=args.split,
            limit=args.limit,
            arms=args.arms,
            repeat=args.repeat,
            llm_model=args.llm_model,
            threshold=args.threshold,
            out_dir=args.out,
            align_threshold=args.align_threshold,
        )
        return 0

    if args.cmd == "exp2":
        from .exp2 import run_exp2

        run_exp2(
            arms=args.arms,
            n_mix=args.n_mix,
            threshold=args.threshold,
            llm_model=args.llm_model,
            out_dir=args.out,
        )
        return 0

    if args.cmd == "exp3-sample":
        import json as _json

        from .exp3 import sample_disagreements

        print(_json.dumps(sample_disagreements(out_dir=args.out), indent=2))
        return 0

    if args.cmd == "exp3-kappa":
        import json as _json

        from .exp3 import cohens_kappa

        print(_json.dumps(cohens_kappa(args.file_a, args.file_b, args.field), indent=2))
        return 0

    if args.cmd == "exp3-recheck":
        import json as _json

        from .exp3 import build_recheck

        print(_json.dumps(build_recheck(args.ann_dir, args.fraction), indent=2))
        return 0

    if args.cmd == "exp3-self-agreement":
        import json as _json

        from .exp3 import self_agreement

        print(_json.dumps(self_agreement(args.main_file, args.recheck_file, args.field)))
        return 0

    if args.cmd == "exp3-annotate":
        from .exp3 import annotate

        result = annotate(args.reviewer, args.file)
        print(
            f"\n{result['annotated_now']} new, {result['total_done']} done, "
            f"{result['remaining']} remaining -> {result['out']}"
        )
        return 0

    if args.cmd == "repro":
        from .repro import diff_summary, recompute_exp1, recompute_exp2

        problems = diff_summary(Path("results/exp1/summary.csv"), recompute_exp1())
        problems += diff_summary(Path("results/exp2/summary.csv"), recompute_exp2())
        for p in problems:
            print(p)
        print(f"reproducibility: {'FAIL' if problems else 'OK'} ({len(problems)} mismatches)")
        return 1 if problems else 0

    return 2
