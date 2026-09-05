"""NLI judge: checkpoint-pluggable entailment scoring with caching.

Default checkpoint: MoritzLaurer/mDeBERTa-v3-base-xnli-2mil7 (10-language,
covers EN/DE/IT natively). Multilingual variant behind the checkpoint flag.
"""

import hashlib
import json
from pathlib import Path
from typing import Protocol

# NOTE: planned 10-language "xnli-2mil7" subset checkpoint does not exist on HF
# (verified via Hub API). The 27-language 2mil7 is the only 2mil7 release and
# serves as default judge AND distillation teacher (Phase 6).
DEFAULT_CHECKPOINT = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MULTILINGUAL_CHECKPOINT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

Probs = dict  # {"entailment": float, "neutral": float, "contradiction": float}


class Judge(Protocol):
    checkpoint: str

    def score(self, premise: str, hypothesis: str) -> Probs: ...

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[Probs]: ...


def is_faithful(probs: Probs) -> bool:
    """Verdict mapping: faithful iff entailment >= contradiction."""
    return probs["entailment"] >= probs["contradiction"]


def _cache_key(checkpoint: str, premise: str, hypothesis: str) -> str:
    return hashlib.sha256(f"{checkpoint}\x00{premise}\x00{hypothesis}".encode()).hexdigest()


class NLIJudge:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        import torch
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(device).eval()
        self.device = device
        id2label = {v: k.lower() for k, v in self.model.config.label2id.items()}
        self.label_of_id = {
            i: next(k for k in ("entailment", "neutral", "contradiction") if k in label)
            for i, label in id2label.items()
        }

    def score(self, premise: str, hypothesis: str) -> Probs:
        return self.score_batch([(premise, hypothesis)])[0]

    def score_batch(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[Probs]:
        import torch

        out: list[Probs] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            premises, hyps = zip(*chunk, strict=True)
            enc = self.tokenizer(
                list(premises), list(hyps), padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            for row in logits.softmax(-1).tolist():
                out.append({self.label_of_id[j]: p for j, p in enumerate(row)})
        return out


class CachedJudge:
    """JSONL-backed score cache keyed by (checkpoint, premise, hypothesis)."""

    def __init__(self, judge: Judge, cache_path: Path):
        self.judge = judge
        self.cache_path = Path(cache_path)
        self._cache: dict[str, Probs] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["probs"]

    @property
    def checkpoint(self) -> str:
        return self.judge.checkpoint

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[Probs]:
        keys = [_cache_key(self.checkpoint, p, h) for p, h in pairs]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            fresh = self.judge.score_batch([pairs[i] for i in missing])
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                for i, probs in zip(missing, fresh, strict=True):
                    self._cache[keys[i]] = probs
                    f.write(json.dumps({"key": keys[i], "probs": probs}) + "\n")
        return [self._cache[k] for k in keys]

    def score(self, premise: str, hypothesis: str) -> Probs:
        return self.score_batch([(premise, hypothesis)])[0]
