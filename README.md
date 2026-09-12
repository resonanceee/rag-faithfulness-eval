# rag-faithfulness-eval

What does it cost to check that a RAG answer is faithful to its sources, and
how good a check can you buy at each price point? This repo benchmarks the real
options end-to-end — an NLI model, LLM judges (9-model sweep + 2 frontier
flagships), hybrids, translate-pipelines for DE/IT, and a distilled student —
on RAGTruth (EN, 18.9k claims) plus synthetic DE/IT (400+400), with two human
reviewers auditing the judge-vs-gold disagreements.

**TL;DR.** `z-ai/glm-5.3-flash` is the best judge we measured — nothing beat
it, including frontier flagships at 22× the cost. `inclusionai/ling-3.0-flash`
is the price/perf king (~95–97% of GLM quality at ~23% of billed cost). NLI
alone is not a detector (recall 0.23), and NLI-confidence hybrid escalation is
a dead lever — the frontier is dominated by the cheap-LLM path everywhere we
looked. Final adjudicated audit: 42% of judge-vs-gold conflicts were gold
annotation noise, so raw metrics *understate* every judge.

## Headline results

| Judge | Claim F1 (EN, Exp4 protocol) | Noise-adjusted F1 | Cost (full 18.9k set) |
|---|---|---|---|
| glm-5.3-flash | **0.481** | **0.693** | ~$1.92 |
| ling-3.0-flash | 0.457 (95% of GLM); response F1 0.742 (98%) | — | $0.45 |
| claude-sonnet-5 (1k frontier check) | 0.702 raw / F1nat 0.403 vs GLM's **0.485** on same ids | — | $2.64 per 1k |
| NLI (mDeBERTa-xnli-2mil7) | 0.138 | 0.369 | $0 (local) |
| Distilled MiniLM student | not promotable (2× negative result) | — | $0 (local) |

## Experiments

### Exp1 — Judge arms on English RAGTruth (18.9k claims)

| Arm | Judge | Claim F1 | Response F1 | Note |
|---|---|---|---|---|
| A | multilingual NLI | 0.143 | 0.446 | fails on recall (0.23), ECE 0.113 |
| B | LLM (glm-5.3-flash) | **0.475** | **0.753** | repeatability 91.3% |
| C | hybrid @0.85 | 0.263 | 0.517 | inherits confident-wrong NLI misses |
| D | no decomposition | — | 0.69 | decomposition is load-bearing |

![Exp1 arms](docs/figures/exp1_arms.png)
![NLI calibration](docs/figures/exp1_calibration.png)

### Threshold recalibration — hybrid does not pay ($0, cache sweep)

The hybrid threshold was set by calibration intuition. A full sweep
(recombined from cached NLI probs + cached GLM verdicts, sanity-checked at
both endpoints) says the lever is dead:

| Escalation threshold | Share sent to LLM | Claim F1 |
|---|---|---|
| 0.50 | 3% | 0.153 |
| 0.85 (old default) | 32% | 0.263 |
| 0.95 | 48% | 0.325 |
| 0.99 | 69% | 0.403 |
| 1.0 (= pure GLM) | 100% | 0.482 |

![Threshold pareto](docs/figures/exp_threshold_pareto.png)

The NLI is too miscalibrated (ECE 0.113) for its confidence to carry
information: 69% of GLM's cost buys only 84% of GLM's F1, and 97% of F1
still costs 93%. hybrid@ling (recomputed after the ling full-set run) shows
the same shape. **Recommendation: drop the hybrid, buy the cheap judge.**

### Exp2 — Cross-lingual DE/IT (synthetic gold, 400+400, v2 data)

| Arm | Judge | DE F1 | IT F1 |
|---|---|---|---|
| A | multilingual NLI direct | 0.556 | 0.774 |
| C | translate → English NLI judge | **0.726** | **0.786** |
| D | hybrid NLI + LLM arbitration | 0.678 | **0.862** |

![Exp2 langs](docs/figures/exp2_langs.png)

Translate-then-English-judge beats direct multilingual on German
hallucination recall (**0.73 vs 0.39**); IT is near-tied; hybrid wins
overall. DE is systematically weaker than IT across arms — repeatable, and it
survived the v2 data fix.

> **Data-quality note (v2)**: initial DE samples had an anglocentric bug —
> `entity_swap` targeted "first capitalized token", which in German is any
> noun, producing mangled claims. v1 (archived in `results/exp2_v1_artifact/`)
> inflated DE F1 by up to 0.20; rankings unchanged, and the translate-vs-
> direct gap *widened* after the fix. Do not quote DE numbers older than
> 2026-09-06.

### Exp4 — Query-in-context (standing protocol)

Adding the labeled question to the premise (`QUESTION: q + PASSAGES: p`)
flips 10–20% of individual verdicts but leaves aggregates stable; LLM recall
gains +5.7pts. Non-good rows excluded. **This is the protocol all headline
numbers use.**

![Exp4 delta](docs/figures/exp4_query_delta.png)

### Exp5 — Judge model sweep (9 models × 2 runs, 4k balanced sample)

![Exp5 pareto](docs/figures/exp5_pareto.png)

| Model | F1nat | Verdict |
|---|---|---|
| glm-5.3-flash (baseline) | 0.475* | quality king |
| **ling-3.0-flash** | **0.463 / 0.469** | **price/perf king** (84.5% agreement w/ GLM) |
| gemini-2.5-flash | 0.424 | fine, not cheapest |
| deepseek-v4-flash, seed-1.6-flash | ~0.42 | mid |
| gpt-4o-mini | 0.372 | underperforms its price |
| gemini-2.5-flash-lite | 0.312 | cheap-and-weak |
| glm-4.7-flash, qwen3.7-flash, gpt-4.1-nano | ≤0.25 | losers at any price |

\* GLM row quoted at full-set rate (18.9k claims).

### Frontier check (1k subset, $4.27)

The open question after the sweep – does a true frontier flagship beat a
purpose-bought cheap judge? – answered on identical claims:

| Model | Claim F1 | F1nat | Recall | Agree w/ GLM | Cost per 1k |
|---|---|---|---|---|---|
| glm-5.3-flash | 0.739 | **0.485** | 0.692 | — | ~$0.12 |
| ling-3.0-flash | 0.658 | 0.444 | 0.565 | — | ~$0.04 |
| claude-sonnet-5 | 0.702 | 0.403 | 0.686 | 0.845 | $2.64 |
| deepseek-v4-pro | 0.677 | 0.417 | 0.619 | 0.796 | $1.63 |

**The frontier did not win** — sonnet-5 lost by 8 F1nat points at ~22×
GLM-flash's billed cost; deepseek-v4-pro lost to ling at ~40× the cost (plus
135 parse errors / +42% retry overhead; strict-JSON compliance is part of
the job at scale). Caveats: n=1,000, single pass, CI ≈ ±0.04–0.05 — the
sonnet-vs-GLM gap is likely real, sonnet-vs-ling is not separable. The
burden of proof is now on the frontier.

### Exp3 — Human audit + adjudication (two reviewers, 71 conflicts, 40 adjudicated)

- **Final, adjudicated: 30/71 conflicts (42%) are confirmed gold-side
  noise** (optimistic any-flag upper bound was 66%). Raw metrics understate
  every judge; the ranking does not change.
- Inter-rater kappa on gold_ok: EN −0.05, DE 0.47, IT 0.09 — weak, hence
  adjudication. Adjudicator sided R1 16× / R2 19× / neither 5×.
- Dispute taxonomy: annotation_noise 16, faithful_but_flagged 12, relation 6,
  fabrication 5, entity 1. Dominant *judge* error: over-flagging faithful
  claims (precision-side).
- Top-3 fixes (threshold / data-noise / decomposition) cover 84% of held-out
  cases.

![Exp3 noise](docs/figures/exp3_noise.png)

Noise-adjusted final ratings (adjudicated labels, `rfe noise-adjust`):

| Arm | Reported claim F1 | Noise-corrected F1 |
|---|---|---|
| exp4 B (GLM) | 0.475 | **0.693** |
| exp4 A (NLI) | 0.138 | 0.369 |
| exp4 C (hybrid@0.85) | 0.263 | 0.493 |
| exp2 C · DE | 0.726 | 0.931 |
| exp2 C · IT | 0.786 | 0.831 |

Per-(lang, arm) cells are small (n=1…19): corrected F1 1.0 cells in exp2
(A/it, D/it) come from 1–2-case cells — read as direction, not decimals. All
rows (recorded + corrected, label provenance per row) in
`results/noise_adjusted.csv`.

### ling-3.0-flash full-set confirmation (18.9k claims, $0.4461)

| Metric | GLM-5.3-flash | ling-3.0-flash | ling as % of GLM |
|---|---|---|---|
| Claim F1 (neutral_neg) | 0.4811 | 0.4568 | 94.9% |
| Response F1 | 0.7603 | 0.7420 | 97.6% |
| Billed cost | $1.92 | $0.4461 | **23%** |

Honest footnote: "~7% of GLM's cost" is the per-token list-price ratio; the
*billed* ratio on this workload is 23% (long contexts dominate input
tokens). The price/perf claim holds at production scale either way.

### Organic DE/IT hallucination distribution (600 answers, $0.46)

Complementing synthetic injections: GLM answered 300 real questions per
language (XQuAD-DE + SQuAD-IT; XQuAD has no Italian split), 1,833 claims,
provisionally judged (Exp4 protocol): hard fabrication **<1%** of claims;
unverifiable drift DE 9.7% / IT 7.5% — the load-bearing organic failure is
answers wandering off-passage, not inventing facts. (The human-adjudication
pass over the organic claims was descoped; results in `results/exp6/` remain
self-judged/provisional.)

### Distillation (branch `distillation`) — negative result, twice

Retry fixed the IT collapse (F1 0.310 → 0.538 vs teacher 0.662) with an
IT-heavy mix (60k EN / 30k DE / 50k IT), 3 epochs, alpha annealed 0.7→0.3 —
the EN-heavy mix, not student capacity, was the bug. But the reframed goal
(high-recall pre-filter) fails on its own terms: catching 95% of
hallucinations requires escalating **78%** of claims, thinner than the ling
path (23% of cost for 95% of quality). **Not promoted.** Teacher remains
the default local judge. (`results/distill/impact_retry.json`.)

## Setup & usage

```sh
pip install -e '.[dev,models]'
rfe build-data --n-per-lang 200   # synthetic DE/IT set -> data/samples.jsonl
rfe validate --input data/samples.jsonl
```

Default local judge: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
(also the distillation teacher). Alternative behind `--judge-checkpoint`.
The planned 10-language 2mil7 subset checkpoint does not exist on HF; the
27-language 2mil7 is the only 2mil7 release.

LLM judging goes through OpenRouter; key in `.env`
(`OPENROUTER_API_KEY`, never committed).

**Full experiment-by-experiment reproduction: [docs/reproduce.md](docs/reproduce.md).**
Every measured number in this README can be recomputed from the checked-in
caches for $0 (`rfe repro`, `rfe threshold-sweep`, `rfe noise-adjust`,
`rfe exp5`).

## Data sources + licenses

| Lang | Source | License |
|------|--------|---------|
| EN | [RAGTruth](https://wandb.ai/wandb/ragtruth_processed_4_benchmarks) processed test | research use |
| DE/IT synthetic | [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) / XNLI / [2mil7](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7) `it_mnli` | CC BY-SA 4.0 / OANC / CC BY-NC 4.0 |
| DE organic | google/xquad `xquad.de` | CC BY-SA 4.0 |
| IT organic | [crux82/squad-it](https://github.com/crux82/squad-it) test | CC BY-SA 4.0 |

IT caveat: XNLI contains no Italian, and `it_mnli` is MNLI machine-translated
to IT — the same data family the NLI judge trained on, so IT synthetic
results carry a contamination caveat.

## Development

```sh
pip install -e '.[dev]'
pytest            # unit tests (35)
ruff check .
```

Real-model checks (both NLI checkpoints, ~1.1GB download each):
`pytest -m smoke` (nightly CI + manual dispatch). Human annotation tooling
(iterative CLI, resumable): `docs/annotation_instructions.md`.
