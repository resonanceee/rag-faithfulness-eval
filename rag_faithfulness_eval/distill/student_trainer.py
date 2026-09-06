"""Student training: MiniLM-class student mimics frozen teacher.

Student: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (~118M),
with a 3-way classification head trained by:
  loss = alpha * KL(student || teacher, T) + (1-alpha) * CE(hard labels)
Default alpha=0.7 (70% soft, 30% hard) per spec 70-30 weighting.
"""

import json
import random
from pathlib import Path

STUDENT_CHECKPOINT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


class DistillationTrainer:
    def __init__(
        self,
        out_dir: Path,
        student_ckpt: str = STUDENT_CHECKPOINT,
        temperature: float = 2.0,
        alpha: float = 0.7,
        lr: float = 2e-5,
        device: str | None = None,
    ):
        import torch

        self.torch = torch
        self.T = temperature
        self.alpha = alpha
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(student_ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            student_ckpt,
            num_labels=3,
            id2label={v: k for k, v in LABEL2ID.items()},
            label2id=LABEL2ID,
        ).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.out_dir = Path(out_dir)

    @staticmethod
    def load_rows(labels_path: Path, pool_path: Path) -> list[dict]:
        labels = {
            json.loads(line)["id"]: json.loads(line)
            for line in labels_path.read_text().splitlines()
            if line.strip()
        }
        rows = []
        for line in pool_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in labels:
                rows.append({**r, "teacher_logits": labels[r["id"]]["logits"]})
        return rows

    def _batch_loss(self, batch: list[dict]):
        torch = self.torch
        enc = self.tok(
            [b["premise"] for b in batch],
            [b["hypothesis"] for b in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        teacher = torch.tensor([b["teacher_logits"] for b in batch], device=self.device)
        soft = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits / self.T, dim=-1),
            torch.nn.functional.softmax(teacher / self.T, dim=-1),
            reduction="batchmean",
        ) * (self.T**2)
        hard = torch.nn.functional.cross_entropy(
            logits, torch.tensor([b["label"] for b in batch], device=self.device)
        )
        return self.alpha * soft + (1 - self.alpha) * hard

    def train(
        self,
        labels_path: Path,
        pool_path: Path,
        epochs: int = 1,
        batch_size: int = 32,
        seed: int = 0,
        max_rows: int | None = None,
    ) -> dict:
        rows = self.load_rows(labels_path, pool_path)
        if max_rows:
            rows = rows[:max_rows]
        rng = random.Random(seed)
        self.model.train()
        step, losses = 0, []
        for epoch in range(epochs):
            rng.shuffle(rows)
            for i in range(0, len(rows), batch_size):
                loss = self._batch_loss(rows[i : i + batch_size])
                loss.backward()
                if (step + 1) % 4 == 0:
                    self.opt.step()
                    self.opt.zero_grad()
                losses.append(float(loss))
                step += 1
                if step % 200 == 0:
                    print(
                        f"epoch {epoch} step {step} loss {sum(losses[-200:]) / 200:.4f}", flush=True
                    )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.out_dir)
        self.tok.save_pretrained(self.out_dir)
        return {"steps": step, "final_loss": losses[-1], "n_rows": len(rows)}


if __name__ == "__main__":
    t = DistillationTrainer(Path("models/student"))
    print(t.train(Path("data/distill/teacher_labels.jsonl"), Path("data/distill/pool.jsonl")))
