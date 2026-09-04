# rag-faithfulness-eval

Measuring RAG faithfulness across EN/DE/IT using NLI judges.

Default judge: `MoritzLaurer/mDeBERTa-v3-base-xnli-2mil7` (10-language, covers EN/DE/IT natively).
Multilingual variant selectable via judge checkpoint flag.

## Phases

0. Scaffold
1. Data pipeline (datasets + hallucination injection)
2. Judge harness
3-4. Experiments (to be defined)
5. Benchmarks + writeup
6. Distillation (separate branch, promoted to default only if it beats benchmarks)

Each phase is followed by a testing phase (T0-T6).

## Development

```sh
pip install -e '.[dev]'
pytest            # unit tests
ruff check .
```

Real-model smoke checks (needs `.[models]`): `pytest -m smoke`
