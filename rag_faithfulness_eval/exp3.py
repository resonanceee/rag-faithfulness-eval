"""Experiment 3: disagreement sampling + reviewer agreement tooling.

rfe exp3-sample : pulls disagreement cases (judge vs gold) from Exp1/Exp2 outputs
                  into annotation files per docs/annotation_instructions.md
rfe exp3-kappa  : Cohen's kappa between two reviewer files
"""

import json
import random
from collections import Counter
from pathlib import Path


def _load_arm(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_disagreements(
    exp1_dir: Path = Path("results/exp1"),
    exp2_dir: Path = Path("results/exp2"),
    out_dir: Path = Path("data/annotation"),
    n_exp1: int = 60,
    n_exp2: int = 30,
    seed: int = 0,
) -> dict:
    """Sample ~n_exp1 EN (RAGTruth) + n_exp2 DE/IT disagreement cases, proportional
    across arms, split for two reviewers, 20% held out for fix validation."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool: list[dict] = []
    # exp1: row context/claim need original data -> join via ragtruth claims
    claim_map = _claim_map()
    for arm_file in sorted(exp1_dir.glob("arm_*.jsonl")):
        arm = arm_file.stem.replace("arm_", "")
        if arm.startswith("B_run"):
            continue
        for r in _load_arm(arm_file):
            if r["verdict"] != ("unfaithful" if r["gold_hallucinated"] else "faithful"):
                c = claim_map.get(r["id"], {})
                pool.append(
                    {
                        "id": r["id"],
                        "source": "exp1",
                        "lang": "en",
                        "arm": arm,
                        "context": c.get("context", ""),
                        "claim": c.get("claim", ""),
                        "gold": "unfaithful" if r["gold_hallucinated"] else "faithful",
                        "judge": r["verdict"],
                    }
                )
    samples_map = {
        s["id"]: s
        for s in (
            json.loads(line)
            for line in Path("data/samples.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    for arm_file in sorted(exp2_dir.glob("arm_*.jsonl")):
        arm = arm_file.stem.replace("arm_", "")
        if arm == "B":
            continue
        for r in _load_arm(arm_file):
            if r["verdict"] != ("unfaithful" if r["gold_hallucinated"] else "faithful"):
                s = samples_map.get(r["id"], {})
                pool.append(
                    {
                        "id": r["id"],
                        "source": "exp2",
                        "lang": r["lang"],
                        "arm": arm,
                        "context": s.get("context", ""),
                        "claim": s.get("claim", ""),
                        "gold": "unfaithful" if r["gold_hallucinated"] else "faithful",
                        "judge": r["verdict"],
                    }
                )

    en = [p for p in pool if p["lang"] == "en"]
    deit = [p for p in pool if p["lang"] != "en"]
    rng.shuffle(en)
    rng.shuffle(deit)
    picked = en[:n_exp1] + deit[:n_exp2]
    rng.shuffle(picked)
    heldout = picked[: max(1, len(picked) // 5)]
    main = picked[len(heldout) :]
    counts = Counter((p["lang"], p["arm"]) for p in main)

    for name, items in (("task2", main), ("task2_heldout", heldout)):
        by_lang: dict[str, list] = {}
        for p in items:
            by_lang.setdefault(p["lang"], []).append(p)
        for lang, rows_l in by_lang.items():
            path = out_dir / f"{name}_{lang}.jsonl"
            with path.open("w") as f:
                for p in rows_l:
                    f.write(
                        json.dumps(
                            {
                                "id": p["id"],
                                "lang": p["lang"],
                                "arm": p["arm"],
                                "context": p["context"],
                                "claim": p["claim"],
                                "gold_label": p["gold"],
                                "judge_verdict": p["judge"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    return {
        "sampled": len(main),
        "held_out": len(heldout),
        "by_lang_arm": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
        "pool_sizes": {"en": len(en), "deit": len(deit)},
    }


def _claim_map() -> dict:
    from .decompose import iter_claims
    from .ragtruth import load_ragtruth

    return {c["id"]: c for r in load_ragtruth("test") for c in iter_claims(r)}


def build_recheck(ann_dir: Path, fraction: float = 0.1, seed: int = 1) -> dict:
    """Emit task2_<lang>_recheck.jsonl: a shuffled `fraction` sample of each
    task2_<lang>.jsonl with ids suffixed '-rc' (so reviewers can't spot dupes).
    Reviewer annotates it like any other file; self_agreement() ties them back."""
    rng = random.Random(seed)
    made = {}
    for f in sorted(ann_dir.glob("task2_*.jsonl")):
        if any(x in f.name for x in ("_recheck", "_heldout", "_reviewer")):
            continue
        rows = _load_arm(f)
        k = max(1, round(len(rows) * fraction))
        picks = rng.sample(rows, k)
        rng.shuffle(picks)
        out = f.with_name(f.stem + "_recheck.jsonl")
        with out.open("w") as fh:
            for r in picks:
                fh.write(json.dumps({**r, "id": r["id"] + "-rc"}, ensure_ascii=False) + "\n")
        made[out.name] = k
    return made


def self_agreement(main_file: Path, recheck_file: Path, field: str) -> dict:
    """% of recheck items where the reviewer's verdict matches their own
    verdict on the original item. Annotation protocol target: >= 90%."""
    main = {r["id"]: r for r in _load_arm(main_file)}
    recheck = _load_arm(recheck_file)
    pairs = [
        (main[r["id"].removesuffix("-rc")][field], r[field])
        for r in recheck
        if r["id"].removesuffix("-rc") in main and field in r
    ]
    agree = sum(a == b for a, b in pairs)
    return {"n": len(pairs), "agreement": round(agree / max(1, len(pairs)), 4)}


HALLUCINATION_TYPES = [
    "entity",
    "numeric",
    "temporal",
    "relation",
    "fabrication",
    "faithful_but_flagged",
]
CAUSES = [
    "judge_world_knowledge",
    "tokenization_edge",
    "bad_decomposition",
    "bad_alignment",
    "judge_limits",
]
FIXES = ["prompt_fix", "alignment_fix", "decomposition_fix", "threshold_fix", "no_fix_noise"]


def _menu(prompt: str, options: list[str], input_fn=input, multi: bool = False) -> list[str]:
    """Interactive numbered pick. Returns list of chosen option strings."""
    print(prompt)
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        raw = input_fn("> ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        picks = raw.replace(",", " ").split() if multi else [raw]
        try:
            idxs = [int(x) for x in picks]
            if idxs and all(1 <= i <= len(options) for i in idxs):
                return [options[i - 1] for i in idxs]
        except ValueError:
            pass
        print("  invalid - enter number(s) '1' or '1 3' (or q to save+quit)")


def annotate(reviewer: int, in_file: Path, input_fn=input, print_fn=print) -> dict:
    """One-at-a-time annotation loop. Writes `<stem>_reviewer<N>.jsonl`,
    resumable (skips already-annotated ids). input_fn/print_fn injectable
    for tests."""
    out_file = in_file.with_name(f"{in_file.stem}_reviewer{reviewer}.jsonl")
    done = {}
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    rows = _load_arm(in_file)
    n_new = 0
    try:
        for i, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            print_fn(f"\n=== item {i}/{len(rows)}  [{r['id']}]  (q=save+quit) ===")
            print_fn(f"CONTEXT: {r.get('context', '')}")
            print_fn(f"CLAIM:   {r.get('claim', '')}")
            print_fn(f"gold: {r.get('gold_label')}   judge: {r.get('judge_verdict')}")
            while True:
                ok = input_fn("gold label correct? [y/n]> ").strip().lower()
                if ok == "q":
                    raise KeyboardInterrupt
                if ok in ("y", "n"):
                    break
                print_fn("  answer y or n (q to save+quit)")
            if ok == "n":
                done[r["id"]] = {**r, "gold_ok": False, "cause": ["annotation_noise"]}
            else:
                htype = _menu("hallucination type:", HALLUCINATION_TYPES, input_fn)[0]
                cause = _menu("cause (multi ok, e.g. '1 3'):", CAUSES, input_fn, multi=True)
                fix = _menu("requested fix:", FIXES, input_fn)[0]
                done[r["id"]] = {
                    **r,
                    "gold_ok": True,
                    "hallucination_type": htype,
                    "cause": cause,
                    "fix": fix,
                }
            n_new += 1
            with out_file.open("w") as f:  # save progress after every item
                for row in done.values():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        pass
    return {
        "annotated_now": n_new,
        "total_done": len(done),
        "out": str(out_file),
        "remaining": len(rows) - len(done),
    }


def adjudicate(
    ann_dir: Path = Path("data/annotation"),
    out_file: Path = Path("results/adjudication_final.jsonl"),
    input_fn=input,
    print_fn=print,
) -> dict:
    """Interactive joint adjudication of reviewer disagreements.

    Shows item + both reviewers' labels; asks final gold_ok and (if yes) final
    type. Resumable; decisions append to out_file.
    """
    decided: dict[str, dict] = {}
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                decided[row["id"]] = row
    queue = []
    for lang in ("en", "de", "it"):
        r1 = {r["id"]: r for r in _load_arm(ann_dir / f"task2_{lang}_reviewer1.jsonl")}
        r2 = {r["id"]: r for r in _load_arm(ann_dir / f"task2_{lang}_reviewer2.jsonl")}
        for cid in sorted(set(r1) & set(r2)):
            t1 = r1[cid].get("hallucination_type", "annotation_noise")
            t2 = r2[cid].get("hallucination_type", "annotation_noise")
            g1, g2 = r1[cid].get("gold_ok", True), r2[cid].get("gold_ok", True)
            if g1 != g2 or t1 != t2:
                queue.append(
                    {"id": cid, "lang": lang, "row": r1[cid], "r1": (g1, t1), "r2": (g2, t2)}
                )
    print_fn(f"{len(queue)} disagreements, {len(decided)} already adjudicated")
    n_new = 0
    try:
        for i, item in enumerate(queue, 1):
            if item["id"] in decided:
                continue
            r = item["row"]
            print_fn(f"\n=== {i}/{len(queue)} [{item['id']}] ({item['lang']})  q=save+quit")
            print_fn(f"CONTEXT: {r.get('context', '')[:500]}")
            print_fn(f"CLAIM:   {r.get('claim', '')}")
            print_fn(f"gold: {r.get('gold_label')}  judge: {r.get('judge_verdict')}")
            print_fn(f"R1: gold_ok={item['r1'][0]} type={item['r1'][1]}")
            print_fn(f"R2: gold_ok={item['r2'][0]} type={item['r2'][1]}")
            while True:
                ok = input_fn("final gold_ok (was the gold label right)? [y/n]> ").strip().lower()
                if ok == "q":
                    raise KeyboardInterrupt
                if ok in ("y", "n"):
                    break
                print_fn("  y or n")
            final = {"id": item["id"], "lang": item["lang"], "gold_ok": ok == "y"}
            if ok == "y":
                options = HALLUCINATION_TYPES + ["annotation_noise"]
                final["hallucination_type"] = _menu("final type:", options, input_fn)[0]
            else:
                final["hallucination_type"] = "annotation_noise"
            decided[item["id"]] = final
            n_new += 1
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w") as f:
                for v in decided.values():
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        pass
    return {"adjudicated_now": n_new, "total": len(decided), "remaining": len(queue) - len(decided)}


def cohens_kappa(file_a: Path, file_b: Path, field: str) -> dict:
    """Kappa on one annotation field between two reviewers (matched by id)."""
    a = {r["id"]: r for r in _load_arm(file_a)}
    b = {r["id"]: r for r in _load_arm(file_b)}
    common = sorted(set(a) & set(b))
    la = [a[i][field] for i in common]
    lb = [b[i][field] for i in common]
    labels = sorted(set(la) | set(lb))
    n = len(common)
    agree = sum(x == y for x, y in zip(la, lb, strict=True)) / max(1, n)
    pe = sum((la.count(lbl) / n) * (lb.count(lbl) / n) for lbl in labels) if n else 0
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": round(agree, 4), "kappa": round(kappa, 4)}
