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

## Results

### Experiment 1 — Judge calibration on RAGTruth (test, 2700 rows → 18,903 claims)

Claim-level, `neutral_neg` mapping (LLM = glm-5.3-flash):

| Arm | Judge | Precision | Recall | F1 | Response-level F1 |
|-----|-------|-----------|--------|-----|-------------------|
| A | multilingual NLI (2mil7) | 0.104 | 0.229 | 0.143 | 0.446 |
| B | LLM (glm-5.3-flash) | 0.377 | 0.641 | **0.475** | **0.753** |
| C | hybrid (NLI ≥0.85 → LLM) | 0.210 | 0.354 | 0.263 | 0.572 |
| D | no-decomposition baseline | — | — | — | 0.107 |
| | random-alignment control | 0.063 | 0.138 | 0.087 | — |

- NLI ECE 0.113; hybrid proxy-ratio 0.345; LLM repeatability 91.3% (2 full runs)
- Judge B cost: ~$4 for 2×18.9k claims via OpenRouter. Judge A/D cost: $0 (local).
- Direct NLI is weak on real RAG data despite looking strong on synthetic
  injections: confident-wrong predictions are why hybrid arbitration underperforms.
- Decomposition is load-bearing (D recall 0.066 at response level).

### Experiment 2 — Cross-lingual DE/IT (Phase 1 synthetic gold, 400+400)

Claim-level, `neutral_pos` mapping shown (arms B rows use 3-way accuracy):

| Arm | Judge | DE F1 | IT F1 |
|-----|-------|-------|-------|
| A | multilingual NLI direct | 0.714 | 0.774 |
| B | cross-lingual-mix (3-way acc) | 1.000 | 0.875 |
| C | translate → English NLI judge | **0.795** | 0.786 |
| D | hybrid + LLM arbitration | **0.868** | **0.860** |

**Quotable**: translate-then-English-judge beats zero-shot multilingual judging
on German hallucination recall (0.845 vs 0.555). Hybrid LLM arbitration beats
both — for ~$0.03. DE lags IT substantially on direct multilingual judging.

### Experiment 3 — Error taxonomy

`rfe exp3-sample` → `data/annotation/` (72 main + 18 held-out disagreements,
two-reviewer protocol in `docs/annotation_instructions.md`).
Awaiting human annotation; then `rfe exp3-kappa` + taxonomy aggregation.

### Reproducibility

`rfe repro` recomputes all core metrics from caches → 0 mismatches (T5).
Pinned revisions + seeds: `config/benchmark.json`.

## Development

```sh
pip install -e '.[dev]'
pytest            # unit tests
ruff check .
```

Real-model checks (both checkpoints, ~1.1GB download each): `pytest -m smoke`
(nightly CI + manual dispatch).

Human spot-check protocol before experiments: `docs/human_agreement_protocol.md`.
