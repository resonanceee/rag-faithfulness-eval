"""Experiment 2: cross-lingual transfer.

Arm A: multilingual NLI direct zero-shot on DE/IT (Phase 1 synthetic gold).
Arm B: cross-lingual-mix pairs (EN context <-> DE/IT claim, both directions),
       3-way NLI accuracy vs gold label (no injections - genuine pairs).
Arm C: translate DE/IT to EN (opus-mt), then English judge (deberta-mnli-fever-anli).
Arm D: hybrid, NLI >=0.85 + LLM arbitration (threshold frozen from Exp1).

Thresholds + verdict mapping are FROZEN from Exp1 calibration: no tuning on DE/IT.
"""

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .exp1 import ArmC, Timer, binary_metrics, nli_verdict, verdicts_to_binary
from .judge import CachedJudge, NLIJudge
from .llm_judge import GLM_FLASH, CostLog, OpenRouterJudge
from .pipeline import read_jsonl

ENGLISH_JUDGE = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"
TRANSLATORS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}


def load_deit_samples(path: Path = Path("data/samples.jsonl")) -> list[dict]:
    rows = [r for r in read_jsonl(path) if r["lang"] in ("de", "it")]
    return [
        {
            "id": r["id"],
            "lang": r["lang"],
            "context": r["context"],
            "claim": r["claim"],
            "gold_hallucinated": not r["faithful"],
            "injection": r["injection"],
        }
        for r in rows
    ]


def build_mix_pairs(n: int = 200, seed: int = 0) -> list[dict]:
    """Arm B cross-lingual-mix pairs, both directions, from genuine NLI data."""
    import random

    from datasets import load_dataset

    pairs: list[dict] = []
    xnli = load_dataset("facebook/xnli", "all_languages", split="validation")
    idx = list(range(len(xnli)))
    random.Random(seed).shuffle(idx)
    for i in idx[:n]:
        r = xnli[i]
        j = r["hypothesis"]["language"].index("de")
        pairs.append(
            {
                "id": f"mix-de-en-{i}",
                "mix": "de",
                "context": r["premise"]["en"],
                "claim": r["hypothesis"]["translation"][j],
                "gold3": r["label"],
                "dir": "en_ctx-de_claim",
            }
        )
        pairs.append(
            {
                "id": f"mix-de-de-{i}",
                "mix": "de",
                "context": r["premise"]["de"],
                "claim": r["hypothesis"]["translation"][r["hypothesis"]["language"].index("en")],
                "gold3": r["label"],
                "dir": "de_ctx-en_claim",
            }
        )
    it = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", split="it_mnli")
    idx = list(range(len(it)))
    random.Random(seed + 1).shuffle(idx)
    for i in idx[:n]:
        r = it[i]
        pairs.append(
            {
                "id": f"mix-it-en-{i}",
                "mix": "it",
                "context": r["premise_original"],
                "claim": r["hypothesis"],
                "gold3": int(r["label"]),
                "dir": "en_ctx-it_claim",
            }
        )
        pairs.append(
            {
                "id": f"mix-it-it-{i}",
                "mix": "it",
                "context": r["premise"],
                "claim": r["hypothesis_original"],
                "gold3": int(r["label"]),
                "dir": "it_ctx-en_claim",
            }
        )
    return pairs


class Translator:
    """opus-mt xx->en with JSONL translation cache (raw auto-model API; the
    'translation' pipeline task was removed in transformers v5)."""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self._cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["out"]

    @staticmethod
    def _translate_batch(model_name: str, texts: list[str]) -> list[str]:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        out = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i : i + 16], padding=True, truncation=True, return_tensors="pt").to(
                device
            )
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=256)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def translate(self, texts: list[tuple[str, str]]) -> list[str]:
        """texts: [(lang, text)] -> english strings (batched per language)."""
        missing = {(lg, t) for lg, t in texts if t not in self._cache}
        for lang, model_name in TRANSLATORS.items():
            chunk = sorted(t for lg, t in missing if lg == lang)
            for t, tr in zip(chunk, self._translate_batch(model_name, chunk), strict=True):
                self._cache[t] = tr
        if missing:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                for _lg, t in sorted(missing):
                    f.write(
                        json.dumps({"key": t, "out": self._cache[t]}, ensure_ascii=False) + "\n"
                    )
        return [self._cache[t] for _, t in texts]


GOLD3 = {0: "entailment", 1: "neutral", 2: "contradiction"}


def _metrics_rows(
    arm: str, samples: list[dict], verdicts: dict[str, str], langs: list[str]
) -> list[dict]:
    rows_out = []
    for lang in langs + ["all"]:
        sel = [s for s in samples if lang == "all" or s["lang"] == lang]
        golds = [s["gold_hallucinated"] for s in sel]
        preds = [verdicts[s["id"]] for s in sel]
        for tag, neutral_pos in (("neutral_neg", False), ("neutral_pos", True)):
            m = binary_metrics(golds, verdicts_to_binary(preds, neutral_pos))
            rows_out.append({"arm": arm, "lang": lang, "metric_set": tag, "n": len(sel), **m})
    return rows_out


def run_exp2(
    arms: str = "ABCD",
    out_dir: Path = Path("results/exp2"),
    samples_path: Path = Path("data/samples.jsonl"),
    n_mix: int = 200,
    threshold: float = 0.85,
    llm_model: str = GLM_FLASH,
    max_workers: int = 8,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_log = CostLog()
    summary: list[dict] = []
    timings: dict[str, str] = {}

    samples = load_deit_samples(samples_path)
    print(f"{len(samples)} DE/IT samples ({Counter(s['lang'] for s in samples)})", flush=True)

    with Timer() as t:
        nli = CachedJudge(NLIJudge(), out_dir / "nli_cache.jsonl")
        nli_scores = dict(
            zip(
                (s["id"] for s in samples),
                nli.score_batch([(s["context"], s["claim"]) for s in samples]),
                strict=True,
            )
        )
    timings["nli_deit"] = Timer.fmt(t.seconds)
    print(f"NLI scoring: {timings['nli_deit']}", flush=True)

    if "A" in arms:
        va = {sid: nli_verdict(p)[0] for sid, p in nli_scores.items()}
        summary += _metrics_rows("A", samples, va, ["de", "it"])
        (out_dir / "arm_A.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": s["id"],
                        "verdict": va[s["id"]],
                        "gold_hallucinated": s["gold_hallucinated"],
                        "lang": s["lang"],
                        "injection": s["injection"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for s in samples
            )
        )

    if "B" in arms:
        with Timer() as t_b:
            mix = build_mix_pairs(n_mix)
            mix_scores = nli.score_batch([(m["context"], m["claim"]) for m in mix])
            correct = sum(
                max(sc, key=lambda k: sc[k]) == GOLD3[m["gold3"]]
                for sc, m in zip(mix_scores, mix, strict=True)
            )
        timings["arm_B"] = Timer.fmt(t_b.seconds)
        per_mix = {}
        for tag in ("de", "it"):
            sel = [(sc, m) for sc, m in zip(mix_scores, mix, strict=True) if m["mix"] == tag]
            acc = sum(max(sc, key=lambda k: sc[k]) == GOLD3[m["gold3"]] for sc, m in sel) / max(
                1, len(sel)
            )
            per_mix[tag] = round(acc, 4)
        summary.append(
            {
                "arm": "B",
                "lang": "de",
                "metric_set": "3way_acc",
                "n": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "accuracy": per_mix["de"],
            }
        )
        summary.append(
            {
                "arm": "B",
                "lang": "it",
                "metric_set": "3way_acc",
                "n": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "accuracy": per_mix["it"],
            }
        )
        print(f"Arm B 3-way accuracy: {per_mix} (overall {correct / len(mix):.3f})", flush=True)

    if "C" in arms:
        with Timer() as t_c:
            tr = Translator(out_dir / "translations.jsonl")
            en_contexts = tr.translate([(s["lang"], s["context"]) for s in samples])
            en_claims = tr.translate([(s["lang"], s["claim"]) for s in samples])
            en_judge = CachedJudge(NLIJudge(ENGLISH_JUDGE), out_dir / "en_cache.jsonl")
            en_scores = en_judge.score_batch(list(zip(en_contexts, en_claims, strict=True)))
            vc = {s["id"]: nli_verdict(p)[0] for s, p in zip(samples, en_scores, strict=True)}
        timings["arm_C"] = Timer.fmt(t_c.seconds)
        summary += _metrics_rows("C", samples, vc, ["de", "it"])
        (out_dir / "arm_C.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": s["id"],
                        "verdict": vc[s["id"]],
                        "gold_hallucinated": s["gold_hallucinated"],
                        "lang": s["lang"],
                        "injection": s["injection"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for s in samples
            )
        )
        print(f"Arm C done in {timings['arm_C']}", flush=True)

    if "D" in arms:
        with Timer() as t_d:
            llm = OpenRouterJudge(
                llm_model, cost_log=cost_log, cache_path=out_dir / "llm_cache_D.jsonl"
            )
            hybrid = ArmC(nli, llm, threshold)
            vd = {}
            executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = {
                executor.submit(
                    lambda s=s: hybrid.verdict(nli_scores[s["id"]], s["context"], s["claim"])
                ): s["id"]
                for s in samples
            }
            for fut in as_completed(futures):
                vd[futures[fut]] = fut.result()
            executor.shutdown()
        timings["arm_D"] = Timer.fmt(t_d.seconds)
        print(f"Arm D proxy-ratio: {hybrid.proxy_ratio:.3f}", flush=True)
        summary += _metrics_rows("D", samples, vd, ["de", "it"])
        (out_dir / "arm_D.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": s["id"],
                        "verdict": vd[s["id"]],
                        "gold_hallucinated": s["gold_hallucinated"],
                        "lang": s["lang"],
                        "injection": s["injection"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for s in samples
            )
        )

    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arm",
                "lang",
                "metric_set",
                "n",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
                "accuracy",
            ],
        )
        w.writeheader()
        w.writerows(summary)
    (out_dir / "cost.json").write_text(
        json.dumps(
            {
                "calls": cost_log.calls,
                "usd": round(cost_log.usd, 4),
                "by_model": cost_log.by_model,
                "parse_errors": cost_log.parse_errors,
                "timings": timings,
            },
            indent=2,
        )
    )
    print(f"exp2 done. llm cost ${cost_log.usd:.4f}", flush=True)
    return {"summary": summary}
