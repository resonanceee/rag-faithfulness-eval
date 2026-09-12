# HANDOFF — rag-faithfulness-eval

Updated: 2026-09-12 (evening), after NEXT-TASK batch T1/T2/T5/T6 executed.
Repo: github.com/resonanceee/rag-faithfulness-eval (public, MIT). All 58 issues closed.

## What this project measured
RAG faithfulness judges (EN/RAGTruth 18.9k claims + DE/IT synthetic 400+400):
NLI vs LLM vs hybrid vs translate-pipelines vs distilled student.

## Headline findings
- **Best judge: z-ai/glm-5.3-flash.** Claim F1 0.475 / response F1 0.753 on
  RAGTruth (vs NLI 0.143/0.446). Sweep winner among 9 models.
- **Best price/perf judge: inclusionai/ling-3.0-flash.** ~97% of GLM quality
  (F1nat 0.466 vs 0.475, 84.5% verdict agreement) at ~7% of its cost.
- Cross-lingual: translate-then-English-judge beats direct multilingual on DE
  (recall 0.73 vs 0.39). IT roughly tied. DE systematically weaker than IT.
- 66% of judge-vs-gold conflicts = gold annotation noise (Exp3, 2 humans).
- Query-in-context: flips 10-20% of verdicts, aggregates stable (B recall +5.7).
- Distillation (MiniLM student): 6.35x faster but IT F1 collapse. Not promoted.
- Sweep losers at any price: glm-4.7-flash, qwen3.7-flash, gpt-4.1-nano.

## Spend
Total ≈ $15.2 (through Exp5) + $0.46 (T6 organic set) + $0.4461 (ling
full-set) + $4.2735 (frontier check: sonnet-5 $2.6434 + deepseek-v4-pro
$1.6301) ≈ **$19.4**. Key lives in `.env` (OPENROUTER_API_KEY), user
controls cap. NEVER commit `.env`.

## Data/protocol state (important, non-obvious)
- RAGTruth rows: exclude quality != good (25 rows). Premise = labeled
  "QUESTION: q\nPASSAGES: p" (Exp4 protocol). align_threshold = 0.2.
- Claim gold = phrase-level ≥20% overlap with RAGTruth spans (claims are
  verbatim substrings => offset alignment, no embeddings).
- Neutral→faithful vs →hallucinated reported BOTH ways; headline = neutral_neg
  for EN, neutral_pos for DE/IT.
- DE dataset = v2 (corpus-noun swaps). v1 archived at results/exp2_v1_artifact/.
  Never trust any DE numbers from before 2026-09-06.
- Judging caches: results/exp*/{nli_cache,llm_cache_*,cache_*}.jsonl —
  reruns of analysis are FREE; never delete.
- Judge sanity smoke (real models): `pytest -m smoke` (nightly CI job).

## Key commands
.venv/bin/activate then `rfe ...`
- repro: `rfe repro` (0 mismatches = truth)
- replay sweep: `rfe exp5 --repeat 2` (fully cached now, $0)
- annotate: `rfe exp3-annotate --reviewer N data/annotation/task2_<lang>.jsonl`
- kappa/self-agreement: `rfe exp3-kappa`, `rfe exp3-self-agreement`
- adjudication: `rfe exp3-adjudicate`
- figures/report: `python -m rag_faithfulness_eval.report`

## Open pitfalls
- `exp3-adjudicate` used to truncate CONTEXT display to 500 chars; judge
  verdicts were computed on FULL context (cache-key verified). Any
  adjudications made before the fix are void — see
  results/adjudication_final.truncated_view.bak.jsonl.
- Reasoning models (GLM 5.3): must pass `reasoning: {exclude: true}` or results
  are null/econ-burned. Already handled in llm_judge.py.
- HF Hub 429s: loader falls back to direct parquet URLs (ragtruth.py).
- `data/`, `results/`, `models/`, MID_EXPERIMENT_FINDINGS.md are gitignored.
- distillation branch holds Phase 6 code; main has docs-only negative result.

## NEXT TASKS — status after 2026-09-12 evening batch
1. Task 2 — DONE ($0): `rfe threshold-sweep` → results/threshold_sweep/.
   Hybrid escalation never dominates: 69% of GLM cost buys only 84% of F1
   (t=0.99). Cheap-judge (ling) dominates the frontier; new figure
   docs/figures/exp_threshold_pareto.png.
2. Task 3 — DONE-FINAL: `rfe noise-adjust` auto-switches to adjudicated
   labels (adjudication_final.jsonl 40/40) → adjudicated gold noise 30/71
   (42%, vs 66% any-flag upper bound). exp4 B 0.475→0.693 corrected.
   results/noise_adjusted.csv rows carry "labels" provenance column.
3. Task 4 (human) — DONE 2026-09-12: 40/40 adjudicated. Real kappas (gold_ok):
   EN -0.05, DE 0.47, IT 0.09; adjudicator sided R1 16 / R2 19 / neither 5.
   Dispute taxonomy: annotation_noise 16, faithful_but_flagged 12, relation 6,
   fabrication 5, entity 1. Two truncated-view decisions (pre-display-fix) were
   re-adjudicated correctly; bak file kept for audit.
4. Task 5 — DONE (negative result 2x): IT F1 fixed 0.310→0.538 (mix was the
   bug, not capacity), but as 95%-recall filter student escalates 78% of
   claims — not promotable. results/distill/impact_retry.json (also in
   worktree). Training op notes: bs=16 + per-epoch checkpoints
   (models/student_e{N}) — bs=32 blew MPS to 25.5G and stalled overnight.
5. Task 6 — DONE, annotation descoped 2026-09-12: `rfe xquad` → results/exp6/
   (600 answers, 1833 claims, $0.46). Organic rate stays PROVISIONAL
   (self-judged): DE 10.5% / IT 7.9%; hard fabrication <1%. task3 annotation
   files deleted per user call; rebuilt on demand via `rfe xquad` export.
   IT source = SQuAD-it test (crux82) — XQuAD has NO Italian split.
6. Optional ling run — DONE ($0.4461, under $0.50): results/exp4_ling/.
   Full-set ling vs GLM: claim F1 0.4568 vs 0.4811 (95%), response F1 0.742
   vs 0.760 (98%), billed cost 23% (NOT the 7% list-price ratio).
   hybrid@ling swept $0 → same dead escalation shape (t=0.999: 97% F1 @93% cost).
7. Writeup — DONE, restructured 2026-09-12: docs/final_report.md and
   docs/human_agreement_protocol.md DELETED per user call. README.md is the
   document of record (all findings + diagrams); docs/reproduce.md holds the
   experiment-by-experiment reproduction table. report.py now only refreshes
   figures (render_report removed).
8. Frontier check — DONE ($4.2735 of $5.19 budget): results/exp5_frontier/
   (1k stratified subset, seed 0). claude-sonnet-5 F1nat 0.4032 vs glm-flash
   0.4846 on same ids at 22x cost; deepseek-v4-pro 0.4173 vs ling 0.4440,
   +135 parse errors/retries. GLM-flash remains best judge; ling keeps
   price/perf crown. Single-pass, n=1k — sonnet-vs-GLM gap likely real,
   sonnet-vs-ling not separable.

Pending human actions: Task 4 adjudication (30min), task3 annotation pass
(1833 organic claims — maybe sample), then `rfe noise-adjust` rerun +
finalize final_report §7 + §8.

Git: main working tree has uncommitted changes (T1/T2/T5 code + report +
figures). Distillation branch worktree ../rag-faithfulness-eval-distill has
uncommitted retry changes (mixer sizes, trainer anneal+checkpoints, filter
eval). Nothing committed/pushed this batch — awaiting user go-ahead.

## NEXT TASKS (original list, preserved)
(see status above; original numbering kept for traceability)
