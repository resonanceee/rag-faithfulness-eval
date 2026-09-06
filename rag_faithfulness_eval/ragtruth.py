"""RAGTruth loading: wandb/RAGTruth-processed -> rows with parsed gold spans.

Falls back to direct parquet download when the HF API rate-limits (429).
"""

import json
from pathlib import Path

DATASET = "wandb/RAGTruth-processed"
_PARQUET_URL = (
    "https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/main/data/"
    "{split}-00000-of-00001.parquet"
)


def _load_frame(split: str, cache_dir: Path):
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"{split}.parquet"
    if not local.exists():
        try:
            from datasets import load_dataset

            ds = load_dataset(DATASET, split=split)
            ds.to_parquet(str(local))
        except Exception:  # 429 or offline: direct parquet URL
            pd.read_parquet(_PARQUET_URL.format(split=split)).to_parquet(local)
    return pd.read_parquet(local)


def load_ragtruth(split: str = "test", cache_dir: Path = Path("data/ragtruth")) -> list[dict]:
    """Rows: {id, query, context, output, task_type, model, spans: [...]}."""
    df = _load_frame(split, cache_dir)
    rows = []
    for r in df.to_dict("records"):
        spans = r["hallucination_labels"]
        spans = json.loads(spans) if isinstance(spans, str) else list(spans)
        rows.append(
            {
                "id": f"rt-{split}-{r['id']}",
                "query": r["query"],
                "context": r["context"],
                "output": r["output"],
                "task_type": r["task_type"],
                "model": r["model"],
                "quality": r["quality"],
                "spans": [
                    {"start": int(s["start"]), "end": int(s["end"]), "text": s["text"]}
                    for s in spans
                ],
            }
        )
    return rows
