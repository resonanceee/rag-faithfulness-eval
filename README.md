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

![Exp1 arms](docs/figures/exp1_arms.png)

- NLI ECE 0.113; hybrid proxy-ratio 0.345; LLM repeatability 91.3% (2 full runs)

![Calibration](docs/figures/exp1_calibration.png)

- Judge B cost: ~$4 for 2×18.9k claims via OpenRouter. Judge A/D cost: $0 (local).
- Direct NLI is weak on real RAG data despite looking strong on synthetic
  injections: confident-wrong predictions are why hybrid arbitration underperforms.
- Decomposition is load-bearing (D recall 0.066 at response level).

### Experiment 2 — Cross-lingual DE/IT (Phase 1 synthetic gold, 400+400)

Claim-level, `neutral_pos` mapping shown (arms B rows use 3-way accuracy):

| Arm | Judge | DE F1 | IT F1 |
|-----|-------|-------|-------|
| A | multilingual NLI direct | 0.556 | 0.774 |
| B | cross-lingual-mix (3-way acc) | 1.000 | 0.875 |
| C | translate → English NLI judge | 0.726 | 0.786 |
| D | hybrid + LLM arbitration | **0.678** | **0.862** |

![Exp2 langs](docs/figures/exp2_langs.png)

**Quotable**: translate-then-English-judge beats zero-shot multilingual judging
on German hallucination recall (0.73 vs 0.39). Hybrid LLM arbitration beats
both for Italian and overall. DE lags IT substantially on direct judging.

> **Data-quality note (v2)**: initial DE samples had an anglocentric bug —
> `entity_swap` targeted "first capitalized token", which in German is any
> noun, producing mangled claims ("Es gibt mehr Siemens..."). v1 (archived in
> `results/exp2_v1_artifact/`) inflated DE F1 by up to 0.20; ordering D > C > A
> unchanged, and the German translate-vs-direct gap *widened* after the fix.
> Dataset of record uses corpus-noun swapping for DE (fluent, category-
> preserving, unsupported claims).

### Experiment 4 — Query-in-context variant (vs Exp 1)

Same 18,875 claims (25 incorrect_refusal/truncated rows excluded), premise =
`QUESTION: q + PASSAGES: p`. Verdict-level agreement with Exp 1: B 90.2%,
A 80.5%, C 79.6% — the query materially flips individual verdicts, but
aggregates are stable (B claim-F1 0.481 vs 0.475; only B recall gains +5.7pts).
Judge ranking unchanged. Response-level exclusion of non-good rows is now the
protocol for final benchmarks.

![Exp4 delta](docs/figures/exp4_query_delta.png)

### Experiment 3 — Error taxonomy (two human reviewers, adjudicated)

71 disagreement cases annotated independently by two humans (interactive CLI,
`docs/annotation_instructions.md`), kappa computed, disagreements adjudicated.

![Exp3 noise](docs/figures/exp3_noise.png)

- **Headline**: 66% of judge-vs-gold conflicts were flagged as gold-side
  annotation noise by at least one reviewer (20% confirmed by both). A large
  share of apparent judge error is RAGTruth span→claim alignment noise, not
  judge failure. This revises raw Exp 1 metrics *upward* for all arms.
- **Real error profile**: dominant type `faithful_but_flagged` (judges
  over-flag faithful claims; precision-side, not recall-side).
- **Inter-rater**: kappa varies by language/field (gold_ok ~0.5-0.73 on EN/DE;
  type labels noisier — documented honestly in `results/adjudication_queue.csv`).
- **Top-3 fixes**: threshold calibration (16 votes), data-noise cleanup (14),
  decomposition improvements (8) — covering 84% of held-out cases.

### Final ratings (after annotation-adjusted interpretation)

| Question | Answer |
|---|---|
| Best judge overall | **LLM (glm-5.3-flash)** — F1 0.48 claim / 0.75 response, ≈$4/run for 19k claims |
| Best $/quality | **Hybrid NLI→LLM** — within ~10-20% of LLM quality at ~6x lower cost |
| NLI alone | Weak as pinpoint detector (recall 0.23), decent high-precision filter |
| Cross-lingual | Translate-then-English beats direct multilingual on DE (recall 0.73 vs 0.39); IT near-tied |
| Query in context | Materially changes ~10-20% of verdicts, aggregates stable |

### Distillation (Phase 6, branch `distillation`) — negative result

Teacher 2mil7 (279M) → student paraphrase-multilingual-MiniLM-L12-v2 (118M),
1 epoch on 150k pairs (100k MNLI-EN + 25k de_mnli + 25k it_mnli), KL+CE 70/30, T=2.
Student is 6.35x faster, 2.4x smaller — but failed parity: F1 0.358 vs teacher
0.503 on the DE/IT benchmark, with the IT drop catastrophic (0.310 vs 0.662)
and teacher-student agreement only ~58%. **Not promoted** (merge criterion:
parity within noise AND faster). Plausible fixes for a retry: multi-epoch,
IT-heavier mix, alpha annealing. Branch kept; teacher remains default judge.

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
