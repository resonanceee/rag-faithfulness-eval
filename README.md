# rag-faithfulness-eval

Measuring RAG faithfulness across EN/DE/IT using NLI judges.

Default judge: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (also the
Phase 6 distillation teacher). Alternative behind `--judge-checkpoint`:
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.

> Checkpoint note: the originally planned 10-language `xnli-2mil7` subset model
> does not exist on the HF Hub. The 27-language 2mil7 is the only 2mil7 release
> and covers EN/DE/IT among its languages.

## Phases

0. Scaffold
1. Data pipeline (datasets + hallucination injection)
2. Judge harness
3-4. Experiments (to be defined)
5. Benchmarks + writeup
6. Distillation (separate branch, promoted to default only if it beats benchmarks)

Each phase is followed by a testing phase (T0-T6). See GitHub milestones/issues.

## Usage

```sh
pip install -e '.[dev,models]'
rfe build-data --n-per-lang 200            # -> data/samples.jsonl + balance report
rfe validate --input data/samples.jsonl    # schema check (exit 1 on violations)
rfe score --input data/samples.jsonl --out results/scores.jsonl \
    --judge-checkpoint MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
```

## Data sources + licenses

| Lang | Source | License |
|------|--------|---------|
| EN | [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) (train) | CC BY-SA 4.0 |
| DE | [XNLI](https://huggingface.co/datasets/facebook/xnli) validation | OANC (via MNLI) |
| IT | [multilingual-NLI-26lang-2mil7](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7) `it_mnli` | CC BY-NC 4.0 |

IT caveat: XNLI contains no Italian, and `it_mnli` is MNLI machine-translated to IT
— the same data family the judge trained on, so IT results carry a contamination
caveat. Revisit IT source when experiments are defined.

Per premise, the pipeline emits one faithful sample (gold entailment hypothesis)
and one unfaithful sample (deterministic injection: entity swap / numeric /
temporal), 1:1 balanced. Injection type balance is entity-heavy (see build report).

## Development

```sh
pip install -e '.[dev]'
pytest            # unit tests
ruff check .
```

Real-model checks (both checkpoints, ~1.1GB download each): `pytest -m smoke`
(nightly CI + manual dispatch).

Human spot-check protocol before experiments: `docs/human_agreement_protocol.md`.
