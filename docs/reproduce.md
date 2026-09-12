# Reproducing every experiment

Prereqs: `pip install -e '.[dev,models]'` from repo root. API-dependent runs
need `OPENROUTER_API_KEY` in `.env` (user-controlled cap; key never
committed). Python 3.13+, torch MPS/CPU automatically selected.

**Golden rule: caches are truth.** `results/exp*/{nli_cache,llm_cache*,cache_*}.jsonl`
make every analysis-layer rerun free. Never delete them. Anything that hits
an API prints live `$` spend and is resumable (interrupt + rerun = cache hit,
$0 for completed items).

## One-command sanity

```sh
rfe repro            # recompute exp1+exp2 metrics from caches; "OK (0 mismatches)"
```

## Per experiment

| Exp | What | Command | Cost first run / replay |
|---|---|---|---|
| Data | Balanced EN/DE/IT synthetic set | `rfe build-data --n-per-lang 200` | $0 / $0 |
| Exp1 | Judge arms on RAGTruth EN (plain premise) | `rfe exp1 --arms ABCD --repeat 2` | ~$4 / $0 |
| Exp2 | Cross-lingual DE/IT arms (translate, hybrid) | `rfe exp2 --arms ABCD` | ~$0.06 / $0 |
| Exp4 | Query-in-context (standing protocol) | `rfe exp4 --arms ABCD` | ~$1.92 / $0 |
| Exp5 | 9-model judge sweep, 4k sample | `rfe exp5 --repeat 2` | ~$9.1 / $0 |
| Exp5 frontier | sonnet-5 + deepseek-v4-pro, 1k subset | `rfe exp5 --models anthropic/claude-sonnet-5 deepseek/deepseek-v4-pro --repeat 1 --out results/exp5_frontier` | $4.27 / $0 |
| Exp3 sample | Disagreement sampling for annotation | `rfe exp3-sample` | $0 |
| Exp3 annotate | Human annotation pass (resumable) | `rfe exp3-annotate --reviewer N data/annotation/task2_<lang>.jsonl` | human time |
| Exp3 kappa | Inter-rater kappa | `rfe exp3-kappa <file_a> <file_b> --field gold_ok` | $0 |
| Exp3 adjudicate | Joint decision on 40 disputes | `rfe exp3-adjudicate` | human time |
| Task 2 | Hybrid threshold recalibration sweep | `rfe threshold-sweep` | $0 / $0 |
| Task 3 | Noise-adjusted metrics (auto-adjudicated) | `rfe noise-adjust` | $0 / $0 |
| Task 6 | Organic DE/IT set (XQuAD-DE + SQuAD-IT) | `rfe xquad --n-per-lang 300` | ~$0.46 / $0 |
| Ling full set | ling-3.0-flash on full 18.9k claims | `rfe exp4 --arms B --repeat 1 --llm-model inclusionai/ling-3.0-flash --out results/exp4_ling` | $0.45 / $0 |
| Figures | Refresh README figures | `python -m rag_faithfulness_eval.report` | $0 |
| Smoke | Real-model checks (nightly CI) | `pytest -m smoke` | $0 (local NLI) |

## Distillation (branch `distillation`, negative result)

```sh
git switch distillation
python -m rag_faithfulness_eval.distill.dataset_mixer    # 110k IT-heavy pool
python -m rag_faithfulness_eval.distill.teacher_labeler  # ~2h on MPS
python -m rag_faithfulness_eval.distill.student_trainer  # 3 epochs, alpha 0.7->0.3
python - <<'EOF'   # impact + high-recall filter eval
from pathlib import Path
from rag_faithfulness_eval.distill.impact_check import impact_report
import json
print(json.dumps(impact_report(Path("models/student"),
      samples_path=Path("data/samples.jsonl")), indent=2))
EOF
```

Ops notes: train at `batch_size=16` on MPS (bs=32 ballooned the MPS footprint
to 25.5G and stalled); per-epoch checkpoints land at `models/student_e{N}`.

## Protocol invariants (do not drift)

- RAGTruth rows: exclude `quality != good` (25 rows). Premise: labeled
  `QUESTION: q\nPASSAGES: p` (Exp4 protocol). `align_threshold = 0.2`.
- Claim gold = phrase-level ≥20% overlap with RAGTruth spans (verbatim
  substrings — offset alignment, no embeddings).
- Headline metrics: `neutral_neg` for EN, `neutral_pos` for DE/IT (both
  always computed and reported).
- DE synthetic set: v2 only (corpus-noun swaps). v1 in
  `results/exp2_v1_artifact/` must not be quoted.
- Reasoning judges must pass `reasoning: {exclude: true}` (hidden tokens are
  still billed — expect ~100–250 per call on top of visible output).
- exp5-style comparisons report prevalence-corrected F1nat back to the
  natural 6.7% hallucination rate.

## Total recorded spend

≈ **$19.4** all-in: $15.2 through exp5 + $0.4461 ling full set + $0.46
organic set + $4.27 frontier check. Reruns from caches: $0.
