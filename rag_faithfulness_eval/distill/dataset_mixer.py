"""Distillation data pool assembly.

Pool: MNLI train EN (n_en) + de_mnli + it_mnli (hard-translated MNLI splits of
the 26lang-2mil7 dataset). XNLI has no train split and no Italian; these two
splits ARE the multilingual MNLI train material for DE/IT.
Output rows: {id, lang, premise, hypothesis, label(int)}
"""

import json
import random
from pathlib import Path

DEFAULT_SIZES = {"en": 100_000, "de": 25_000, "it": 25_000}


def mix_pool(out_path: Path, sizes: dict = DEFAULT_SIZES, seed: int = 0) -> dict:
    from datasets import load_dataset

    rng = random.Random(seed)
    rows = []
    mnli = load_dataset("nyu-mll/multi_nli", split="train")
    idx = list(range(len(mnli)))
    rng.shuffle(idx)
    for i in idx[: sizes["en"]]:
        r = mnli[i]
        rows.append({"id": f"en-mnli-{i}", "lang": "en", "premise": r["premise"],
                     "hypothesis": r["hypothesis"], "label": r["label"]})
    for lang, split in (("de", "de_mnli"), ("it", "it_mnli")):
        ds = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", split=split)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[: sizes[lang]]:
            r = ds[i]
            rows.append({"id": f"{lang}-{split}-{i}", "lang": lang,
                         "premise": r["premise"], "hypothesis": r["hypothesis"],
                         "label": int(r["label"])})
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"total": len(rows), "sizes": sizes}


if __name__ == "__main__":
    print(mix_pool(Path("data/distill/pool.jsonl")))
