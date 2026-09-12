"""Task 6 / Exp6: organic DE/IT hallucination set from real QA corpora via GLM answers.

Corpora: DE = XQuAD (xquad.de), IT = SQuAD-it test (crux82) — XQuAD has no
Italian split.

Synthetic set (Exp2) uses noun-swap injections — neatly labeled but sterile.
Here GLM answers real XQuAD questions (300/lang) with the passage as context;
whatever hallucinations it makes are organic. Answers are sentence-decomposed
(split_claims), provisionally judged by GLM (Exp4 protocol), and exported as
annotation task files (data/annotation/task3_<lang>.jsonl) in the exp3-annotate
format: `gold_label` carries the provisional judge verdict; reviewers confirm
or correct it. Final organic hallucination distribution waits on that pass.

Cost: ~600 answers + ~1.5-2.5k judge calls on glm-5.3-flash (~$0.15).
All stages cached/resumable; reruns are $0.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .decompose import split_claims
from .llm_judge import GLM_FLASH, CostLog, OpenRouterJudge

OUT_DIR = Path("results/exp6")
ANN_DIR = Path("data/annotation")

ANSWER_SYSTEM = (
    "You are a retrieval-augmented QA assistant. Answer the question based on "
    "the PASSAGE in 2-4 full sentences, in the passage's language. Include the "
    "direct answer plus the relevant supporting details from the passage."
)


def _load_xquad(lang: str) -> list[dict]:
    """QA pairs per language.

    DE: google/xquad xquad.de validation (HF; parquet fallback for 429s).
    IT: XQuAD has no Italian split -> crux82/squad-it test set (SQuAD v1 IT
    translation, SQuAD_it-test.json.gz from GitHub). Provenance difference is
    noted in the report; both are passage-grounded extractive QA.
    """
    if lang == "it":
        import gzip
        import urllib.request

        url = "https://raw.githubusercontent.com/crux82/squad-it/master/SQuAD_it-test.json.gz"
        d = json.loads(gzip.decompress(urllib.request.urlopen(url, timeout=120).read()))
        rows = []
        for doc in d["data"]:
            for p in doc["paragraphs"]:
                for qa in p["qas"]:
                    rows.append(
                        {
                            "id": f"squadit-test-{len(rows)}",
                            "question": qa["question"],
                            "passage": p["context"],
                            "gold_answer": (qa.get("answers") or [{}])[0].get("text", ""),
                        }
                    )
        # questions share passages in SQuAD; sample questions, not passages
        return rows

    from datasets import load_dataset

    try:
        ds = load_dataset("google/xquad", f"xquad.{lang}", split="validation")
    except Exception:  # 429 or offline: direct parquet URL (mirrors ragtruth.py)
        import pandas as pd

        url = f"https://huggingface.co/datasets/google/xquad/resolve/main/xquad.{lang}/validation-00000-of-00001.parquet"
        ds = pd.read_parquet(url).to_dict("records")
    return [
        {
            "id": f"xquad-{lang}-{i}",
            "question": r["question"],
            "passage": r["context"],
            "gold_answer": (r.get("answers") or {}).get("text", [""])[0]
            if isinstance(r.get("answers"), dict)
            else "",
        }
        for i, r in enumerate(ds)
    ]


def build_questions(n_per_lang: int = 300, seed: int = 0) -> list[dict]:
    import random

    rng = random.Random(seed)
    out = []
    for lang in ("de", "it"):
        rows = _load_xquad(lang)
        rng.shuffle(rows)
        out += [{**r, "lang": lang} for r in rows[:n_per_lang]]
    return out


def generate_answers(questions: list[dict], model: str = GLM_FLASH) -> tuple[dict, CostLog]:
    """GLM answer per question, cached by sha256(model, question)."""
    cache_path = OUT_DIR / "cache_answers.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row["answer"]
    log = CostLog()
    judge = OpenRouterJudge(model, cost_log=log)
    keys = {
        q["id"]: hashlib.sha256(
            f"{model}\x00{q['question']}\x00{q['passage']}".encode()
        ).hexdigest()
        for q in questions
    }
    todo = [q for q in questions if keys[q["id"]] not in cache]
    print(f"{len(questions)} questions, {len(todo)} to answer")

    def _answer(q: dict) -> tuple[str, str]:
        msg = f"PASSAGE:\n{q['passage']}\n\nQUESTION:\n{q['question']}"
        resp = judge._call(msg, system=ANSWER_SYSTEM, max_tokens=512)
        judge._account(resp)
        return q["id"], (resp["choices"][0]["message"].get("content") or "").strip()

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_answer, q): q for q in todo}
        for i, fut in enumerate(as_completed(futs)):
            qid, text = fut.result()
            cache[keys[qid]] = text
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a") as f:
                f.write(json.dumps({"key": keys[qid], "answer": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  answers: {i + 1}/{len(todo)} (${log.usd:.4f})", flush=True)
    return {q["id"]: cache[keys[q["id"]]] for q in questions}, log


def decompose_answers(questions: list[dict], answers: dict) -> list[dict]:
    claims = []
    for q in questions:
        for j, (_, _, sent) in enumerate(split_claims(answers[q["id"]])):
            claims.append(
                {
                    "id": f"{q['id']}-c{j}",
                    "lang": q["lang"],
                    "question": q["question"],
                    "passage": q["passage"],
                    "answer": answers[q["id"]],
                    "claim": sent.strip(),
                }
            )
    return claims


def judge_claims(claims: list[dict], model: str = GLM_FLASH) -> tuple[dict, CostLog]:
    """Provisional GLM verdicts, Exp4 protocol (QUESTION+PASSAGES premise)."""
    log = CostLog()
    judge = OpenRouterJudge(model, cost_log=log, cache_path=OUT_DIR / "cache_judge.jsonl")
    out: dict = {}

    def _v(c: dict) -> tuple[str, str]:
        premise = f"QUESTION: {c['question']}\nPASSAGES: {c['passage']}"
        return c["id"], judge.verdict(premise, c["claim"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_v, c): c for c in claims}
        for i, fut in enumerate(as_completed(futs)):
            cid, v = fut.result()
            out[cid] = v
            if (i + 1) % 200 == 0:
                print(f"  judged: {i + 1}/{len(claims)} (${log.usd:.4f})", flush=True)
    return out, log


def export_annotation_tasks(claims: list[dict], verdicts: dict) -> dict:
    """exp3-annotate compatible files; gold_label = provisional judge verdict."""
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for lang in ("de", "it"):
        rows = [c for c in claims if c["lang"] == lang]
        path = ANN_DIR / f"task3_{lang}.jsonl"
        with path.open("w") as f:
            for c in rows:
                f.write(
                    json.dumps(
                        {
                            "id": c["id"],
                            "lang": lang,
                            "arm": "xquad_glm",
                            "context": f"QUESTION: {c['question']}\nPASSAGES: {c['passage']}",
                            "claim": c["claim"],
                            "gold_label": verdicts[c["id"]],
                            "judge_verdict": verdicts[c["id"]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        counts[lang] = len(rows)
    return counts


def main(n_per_lang: int = 300) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = build_questions(n_per_lang)
    answers, a_log = generate_answers(questions)
    claims = decompose_answers(questions, answers)
    verdicts, j_log = judge_claims(claims)
    counts = export_annotation_tasks(claims, verdicts)
    dist = {}
    for lang in ("de", "it"):
        vs = [verdicts[c["id"]] for c in claims if c["lang"] == lang]
        dist[lang] = {
            "claims": len(vs),
            "unfaithful": vs.count("unfaithful"),
            "unverifiable": vs.count("unverifiable"),
            "rate": round((vs.count("unfaithful") + vs.count("unverifiable")) / max(1, len(vs)), 4),
        }
    report = {
        "n_questions": len(questions),
        "n_claims": len(claims),
        "annotation_files": counts,
        "provisional_hallucination_rate": dist,
        "cost": {"answers_usd": round(a_log.usd, 4), "judge_usd": round(j_log.usd, 4)},
        "note": "provisional = GLM self-judged; human annotation pass was "
        "descoped 2026-09-12 (task3 files deletable/rebuildable on demand)",
    }
    (OUT_DIR / "organic_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
