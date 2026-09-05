"""Dataset build pipeline: SNLI (EN) + XNLI (DE/IT) -> balanced JSONL.

Per language: n_per_lang entailment pairs, each emitting one faithful sample
(original hypothesis) and one unfaithful sample (injected hallucination).
Injection type rotates entity_swap -> numeric_perturb -> temporal_perturb,
so counts are balanced 1:1 faithful:unfaithful and ~even across injection types.
"""

import json
import random
from collections import Counter
from pathlib import Path

from .inject import INJECTIONS, inject
from .schema import LANGS, Sample

ENTAILMENT = 0  # snli/xnli label convention


def _en_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    from datasets import load_dataset  # lazy: heavy dep, only needed at build time

    # train split: validation is too small once we skip claims with no injection target
    ds = load_dataset("stanfordnlp/snli", split="train")
    rows = [r for r in ds if r["label"] == ENTAILMENT]
    random.Random(seed).shuffle(rows)
    return [(r["premise"], r["hypothesis"]) for r in rows[:n]]


def _de_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("facebook/xnli", "all_languages", split="validation")
    rows = [r for r in ds if r["label"] == ENTAILMENT]
    random.Random(seed + 1).shuffle(rows)
    out = []
    for r in rows[:n]:
        i = r["hypothesis"]["language"].index("de")
        out.append((r["premise"]["de"], r["hypothesis"]["translation"][i]))
    return out


def _it_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    # XNLI has no Italian; it_mnli = MNLI hard-translated to IT (same data family
    # the judge trained on -> note as contamination caveat for IT eval results).
    from datasets import load_dataset

    ds = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", split="it_mnli")
    rows = [r for r in ds if int(r["label"]) == ENTAILMENT]
    random.Random(seed + 2).shuffle(rows)
    return [(r["premise"], r["hypothesis"]) for r in rows[:n]]


def build_samples(n_per_lang: int = 200, seed: int = 0) -> list[Sample]:
    samples: list[Sample] = []
    fetchers = {"en": _en_pairs, "de": _de_pairs, "it": _it_pairs}
    sources = {"en": "snli", "de": "xnli", "it": "it_mnli"}
    # EN hypotheses are mostly lowercase/no-digit -> only ~2.5% injectable; DE/IT ~33%+
    oversample = {"en": 60, "de": 3, "it": 3}
    for lang in LANGS:
        kept = 0
        for i, (context, claim) in enumerate(fetchers[lang](n_per_lang * oversample[lang], seed)):
            if kept >= n_per_lang:
                break
            kinds = [INJECTIONS[(i + j) % len(INJECTIONS)] for j in range(len(INJECTIONS))]
            bad_claim, kind = next(
                ((out, k) for k in kinds if (out := inject(claim, k, lang, seed + i)) is not None),
                (None, None),
            )
            if bad_claim is None or kind is None:
                continue  # no injection applies to this claim
            sid = f"{lang}-{kept:05d}"
            samples.append(
                Sample(
                    id=sid,
                    lang=lang,
                    context=context,
                    claim=claim,
                    faithful=True,
                    source=sources[lang],
                )
            )
            samples.append(
                Sample(
                    id=f"{sid}-{kind}",
                    lang=lang,
                    context=context,
                    claim=bad_claim,
                    faithful=False,
                    injection=kind,
                    source=sources[lang],
                )
            )
            kept += 1
    return samples


def report(samples: list[Sample]) -> dict:
    by_lang = Counter(s.lang for s in samples)
    by_faithful = Counter(s.faithful for s in samples)
    by_injection = Counter(s.injection for s in samples if s.injection)
    return {
        "per_lang": dict(by_lang),
        "faithful": dict(by_faithful),
        "injection_types": dict(by_injection),
        "total": len(samples),
    }


def write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
