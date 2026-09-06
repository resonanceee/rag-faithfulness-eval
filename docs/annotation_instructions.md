# Human Annotation Instructions — Faithfulness Labels & Error Review

Two reviewers, independent, no communication during annotation. Same instructions
for both (these first sections are the shared protocol).

## Task 1: Faithfulness labels (Experiment 2, Option B only)

You are given `(context, claim)` pairs in German or Italian. The context is from
Wikipedia. The claim was machine-generated.

**Label exactly one of:**

- `faithful` — every fact in the claim is fully supported by the context alone.
  Do NOT use outside knowledge. If the claim is true in the world but not in the
  context, it is NOT faithful.
- `unfaithful` — at least one fact in the claim contradicts or is unsupported by
  the context (wrong entity, wrong number, wrong date, fabricated detail).
- `unverifiable` — the claim cannot be checked against the context (context
  irrelevant to claim, claim is pure opinion, or claim is empty/garbled).

**Rules:**

1. Context only. World knowledge never overrides the context.
2. One bad fact makes the whole claim `unfaithful`.
3. Paraphrase is fine: same meaning, different words = `faithful`.
4. Numbers and dates must match exactly (units too: 3 km ≠ 3 miles).
5. When torn between `unfaithful` and `unverifiable`: if you can name the specific
   unsupported fact, label `unfaithful`; if the context simply doesn't address
   the claim at all, label `unverifiable`.
6. Self-agreement (10% double-pass) is mechanical, you do NOT shuffle anything:
   - Files named `task2_<lang>_recheck.jsonl` contain a shuffled ~10% sample of
     your main file with disguised ids (you won't recognize the duplicates).
   - After finishing your main file, annotate the recheck file exactly the same
     way, same rules, without looking back.
   - Self-agreement is computed after the fact:
     `rfe exp3-self-agreement <main_reviewer_file> <recheck_reviewer_file> --field hallucination_type`
   Target: ≥ 0.90. Below that: pause, re-read rules, redo the last block.

**Pace:** budget 1.5-2 min per item. Report time-per-25-items blocks.

## Task 2: Error taxonomy review (Experiment 3)

You are given `(context, claim, gold_label, judge_verdict)` where at least one
judge disagreed with the gold label. For each case:

1. **Was the gold label right?** If the gold is wrong, mark `annotation_noise`
   and stop — no taxonomy needed.
2. **Hallucination type** (pick one):
   - `entity` — wrong person, place, org, name
   - `numeric` — wrong quantity, measurement, percentage
   - `temporal` — wrong date, year, ordering, duration
   - `relation` — wrong who-did-what-to-whom
   - `fabrication` — content absent from context entirely
   - `faithful_but_flagged` — judge wrongly flagged a faithful claim
3. **Suspected cause** (pick any that apply):
   - `judge_world_knowledge` — judge trusted its own knowledge over the context
   - `tokenization_edge` — German compounds / Italian or German verb agreement
   - `bad_decomposition` — claim was split badly, lost meaning
   - `bad_alignment` — claim matched to wrong context passage
   - `judge_limits` — none of the above; judge just missed semantics
4. **Requested fix** (pick one): `prompt_fix`, `alignment_fix`,
   `decomposition_fix`, `threshold_fix`, `no_fix_noise`.

**Independence:** reviewers annotate in separate files, no peeking. After both
finish: compute Cohen's kappa per task; adjudicate disagreements together, log
final label + which reviewer yielded.

## Files (how annotating actually works)

Each line is one JSON object with `id`, `lang`, `context`, `claim`,
`gold_label`, `judge_verdict`. You annotate by **copying the file to
`..._reviewer{N}.jsonl` and adding your fields to each line** (in any text
editor, one line at a time, top to bottom):

- `task2_<lang>.jsonl` — your main items (annotate all)
- `task2_<lang>_recheck.jsonl` — the disguised ~10% re-annotation (annotate after
  the main file, same fields)
- `task2_heldout_<lang>.jsonl` — DO NOT annotate; reserved for fix validation
- Never edit the other reviewer's file (`_reviewer1` vs `_reviewer2`)
- Fields to add per line (Task 2): `gold_ok` (yes/no), `hallucination_type`,
  `cause`, `fix` — per the rules above

Reviewer 2 gets the same `task2_*.jsonl` files; independence = you never see
each other's `_reviewerN` files until adjudication.
