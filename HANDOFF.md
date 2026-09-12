# HANDOFF — ragfaith (formerly rag-faithfulness-eval)

Updated: 2026-09-12, post-next-task batch. Repo renamed: github.com/resonanceee/ragfaith.
Benchmark package dir keeps the historical name: `rag_faithfulness_eval/`
(the measured stuff stays under that module; CLI is `rfe`).

## Project state

All benchmark work complete. README.md is the document of record (warm case
study + numbers); docs/reproduce.md has per-experiment reproduction; CI green.
Spend ≈ $19.4 total. Key in `.env` (OPENROUTER_API_KEY), never committed.

## Protocol invariants (these make the numbers mean anything)

- RAGTruth rows: exclude `quality != good` (25 rows). Premise = labeled
  `QUESTION: q\nPASSAGES: p` (Exp4 protocol). align_threshold = 0.2.
- Claim gold = phrase-level ≥20% overlap with RAGTruth spans; align via
  offsets, no embeddings.
- Headline metrics: neutral_neg EN, neutral_pos DE/IT; both always computed.
- DE synthetic = v2 (corpus-noun swaps). v1 archived at
  results/exp2_v1_artifact/; never quote DE numbers from before 2026-09-06.
- Judging caches under `results/exp*/` make all analysis reruns FREE; never
  delete them.
- Adjudication done: results/adjudication_final.jsonl (40/40). noise-adjust
  auto-prefers adjudicated labels; exp3 noise-adjusted numbers are FINAL.
- Organic set (results/exp6/) is self-judged/provisional; human annotation
  pass was descoped (task3 files deleted, regenerate via `rfe xquad`).
- distillation branch holds the retry (2x negative result); main is default.

## Open pitfalls

- exp3-adjudicate previously truncated context display at 500 chars (fixed);
  results/adjudication_final.truncated_view.bak.jsonl kept for audit only.
- Reasoning models need `reasoning: {exclude: true}` (still bills hidden
  reasoning tokens, ~100-250/call extra).
- DeepSeek-family models fail strict-JSON ~13% of calls (135/1000 retries on
  the frontier run, +42% retry cost). Budget retries for any DS judge.
- `python -m rag_faithfulness_eval.report` only refreshes figures now; it
  no longer writes any markdown report.
- Smoke tests (real NLI checkpoints): `pytest -m smoke` (nightly CI).

## Key commands

CLI: `rfe` after `pip install -e '.[dev,models]'`
- repro truth check: `rfe repro` (0 mismatches = caches consistent)
- $0 analyses: `rfe threshold-sweep`, `rfe noise-adjust`
- replay sweep: `rfe exp5 --repeat 2` (cached, $0)
- annotation: `rfe exp3-annotate --reviewer N <file>`, `rfe exp3-adjudicate`
- figures: `python -m rag_faithfulness_eval.report`
- synthetic provider (synthetic.new) model list:
  `GET https://api.synthetic.new/v1/models` with the synthetic key

## NEXT WORK — integrations (agent instructions)

Two new deliverables, each on its own feature branch off main. Do NOT merge
until the merge criteria below pass. Share judge code between them (copy the
small judge module, don't add cross-path imports).

### Shared judge system (both integrations)

1. Detect the currently-active chat model of the session/host tool.
2. Judge selection:
   - active model != GLM-5.3-Flash -> judge = GLM-5.3-Flash
     (`z-ai/glm-5.3-flash` / synthetic `hf:zai-org/GLM-5.3-Flash`)
   - active model == GLM-5.3-Flash -> judge = DeepSeek-v4.1-Flash
     (`hf:deepseek-ai/DeepSeek-V4.1-Flash` on synthetic) — never let a model
     judge its own output
3. Behavior per assistant reply:
   - decompose reply into claims (port `split_claims` logic)
   - judge each claim against premises = content actually pulled this
     session (tool results, fetched docs/URLs, not the model's memory)
   - `faithful` verdict: SILENT PASS (no annotation, no log noise beyond
     metrics)
   - `unfaithful`/`unverifiable` verdict: inject a follow-up prompt into the
     model conversation nudging it to re-check its sources and reconcile
     the claim against what was actually pulled (do NOT auto-correct the
     claim yourself; nudge, then record the outcome)
4. Defaults: warnings/nudges only (never block replies), temperature 0,
   `reasoning: {exclude: true}`, sha256-cached verdicts, live cost logging.

### Deliverable 1: opencode plugin -> `integrations/opencode/`

- AGENTS.md in userspace: treat as opencode-native context (plugin hook
  surface: tool.execute.before/after, experimental.system/text transforms).
- Implement layers: doc-pull process check (free, deterministic — dependency
  actively used in a tool call without a prior docs fetch in session) +
  faithfulness cascade above.
- Runs against opencode's provider stack (synthetic); detect active model
  from request params.

### Deliverable 2: universal proxy -> `integrations/proxy/`

- OpenAI-compatible sidecar: host tool points at proxy endpoint, proxy
  forwards to the real provider (synthetic), passes SSE through, harvests
  premises from conversation history, runs the cascade, appends nudge
  prompts as injected follow-up user messages.
- FRAME THIS AS A UNIVERSAL ADAPTER for any tool without a plugin system —
  FlowDown (Apple AI chat client) is the reference example in the README of
  this subdir, not the sole target.
- Must handle: SSE streaming (stream-through; append verdict/nudge as
  follow-up turn, do NOT delay first token), tool-call result capture,
  model detection per request, per-host decorator config (which host tools
  get which nudge template).

### Merge criteria (gate before merging either branch to main)

1. Unit tests green (CI) + `ruff check` clean.
2. Smoke test on 50 cached RAGTruth claims through the synthetic endpoint:
   GLM judge reproduces cache verdicts within noise; deepseek judge emits
   parseable verdicts >=95% of claims (else add parse-retry logic).
3. End-to-end dry run in the real host (opencode plugin in an opencode
   session; proxy in front of ANY OpenAI client): verdicts appended/nudged
   correctly, faithful claims silent, no deadlocks on streaming.
4. Spend logged and within cap ($1 total across both smokes is plenty).
5. README subdir docs explain config; main README untouched.

Human gate: user reviews branch PRs; agent does not self-merge to main.
