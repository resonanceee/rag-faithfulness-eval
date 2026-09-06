"""Vocabulary slimming: keep only tokens observed in EN/DE/IT corpora.

Builds a presence list from the training pool + eval data, then prunes the
student embedding matrix. Measure speed/accuracy impact before adopting:
run impact_check with and without slimming.
"""

import json
from collections.abc import Iterable
from pathlib import Path


def build_presence_list(texts: Iterable[str], tokenizer_name: str) -> set[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    used: set[int] = set()
    for t in texts:
        used.update(tok.encode(t, add_special_tokens=True))
    # always keep special tokens
    used.update(i for i in tok.all_special_ids if i is not None)
    return used


def corpus_texts(
    paths: list[Path], fields: tuple[str, ...] = ("premise", "hypothesis", "context", "claim")
):
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for f in fields:
                if row.get(f):
                    yield row[f]


def slim_model(model_dir: Path, out_dir: Path, keep_ids: set[int]) -> dict:
    """Rewrite model embeddings keeping only keep_ids rows. Saves to out_dir."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    old_emb = model.get_input_embeddings().weight.data
    keep = sorted(keep_ids)
    index = torch.full((old_emb.size(0),), -1, dtype=torch.long)
    for new_id, old_id in enumerate(keep):
        index[old_id] = new_id
    model.resize_token_embeddings(len(keep))
    with torch.no_grad():
        model.get_input_embeddings().weight.data = old_emb[keep, :]
    # remap tokenizer: build new vocab
    new_vocab = {tok.convert_ids_to_tokens(old): new for new, old in enumerate(keep)}
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "vocab_map.json").write_text(
        json.dumps(
            {
                "keep_ids_order": [
                    old_vocab_inv for old_vocab_inv in [tok.convert_ids_to_tokens(o) for o in keep]
                ],
                "new_vocab": new_vocab,
            }
        )
    )
    n_params = sum(p.numel() for p in model.parameters())
    return {"old_vocab": old_emb.size(0), "new_vocab": len(keep), "params_after": n_params}
