"""Teacher labeling: frozen 2mil7 teacher -> soft-logit labels for the pool.

Appends to output JSONL (resumable): {id, logits: [e, n, c]}
"""

import json
import math
from pathlib import Path

from ..judge import NLIJudge

# teacher label order: email-neu-contra per checkpoint config (verified config:
# 0=entailment, 1=neutral, 2=contradiction)
ORDER = ("entailment", "neutral", "contradiction")


def label_pool(
    pool_path: Path,
    out_path: Path,
    checkpoint: str | None = None,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict:
    rows = [json.loads(line) for line in pool_path.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    done = set()
    if out_path.exists():
        done = {
            json.loads(line)["id"] for line in out_path.read_text().splitlines() if line.strip()
        }
    todo = [r for r in rows if r["id"] not in done]
    print(f"pool {len(rows)}, done {len(done)}, todo {len(todo)}")
    if not todo:
        return {"labeled": 0, "total": len(done)}

    kwargs = {} if checkpoint is None else {"checkpoint": checkpoint}
    judge = NLIJudge(**kwargs)
    labeled = 0
    for i in range(0, len(todo), 1000):
        chunk = todo[i : i + 1000]
        probs = judge.score_batch([(r["premise"], r["hypothesis"]) for r in chunk], batch_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as f:
            for r, p in zip(chunk, probs, strict=True):
                logits = [math.log(max(p[k], 1e-12)) for k in ORDER]
                f.write(
                    json.dumps({"id": r["id"], "label": p and r["label"], "logits": logits}) + "\n"
                )
        labeled += len(chunk)
        print(f"labeled {labeled}/{len(todo)}", flush=True)
    return {"labeled": labeled}


if __name__ == "__main__":
    print(label_pool(Path("data/distill/pool.jsonl"), Path("data/distill/teacher_labels.jsonl")))
