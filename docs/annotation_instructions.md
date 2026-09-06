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

## How to annotate (interactive CLI, no file editing)

### First-time setup (fresh clone, macOS)

The `rfe` command has no external dependencies, so nothing beyond the package
itself needs installing. Requires Python ≥ 3.11 (check: `python3 --version`;
if older, `brew install python@3.13` and substitute `python3.13` below).

```sh
cd /path/to/rag-faithfulness-eval     # repo root after cloning
python3 -m venv .venv                 # one-time
source .venv/bin/activate
pip install -e .                      # one-time; creates the `rfe` command
```

**Setup (once per terminal session after that):**

```sh
cd /Users/res/Code/rag-faithfulness-eval
source .venv/bin/activate        # makes `rfe` available
```

(or prefix every command with `.venv/bin/rfe` / use `.venv/bin/python -m rag_faithfulness_eval`)

When finished annotating: `deactivate` exits the venv.

You are reviewer 1 or 2. For each language file, run:

```sh
rfe exp3-annotate --reviewer 1 data/annotation/task2_en.jsonl
rfe exp3-annotate --reviewer 1 data/annotation/task2_de.jsonl
rfe exp3-annotate --reviewer 1 data/annotation/task2_it.jsonl
```

Reviewer 2 uses `--reviewer 2` on the same input files (your outputs land in
separate `..._reviewer2.jsonl` files — never look at the other reviewer's
files until adjudication).

Per item the tool shows CONTEXT, CLAIM, gold label, judge verdict, then asks:

1. **gold label correct?** `y`/`n` — if `n`, item is annotation noise, done.
2. **hallucination type** — pick one number from the menu.
3. **cause** — one or more numbers, space-separated (e.g. `1 3`).
4. **requested fix** — one number.

- Your answer is saved **after every item**; `q` at any prompt saves and quits.
- Re-run the same command to resume — already-annotated items are skipped.
- `task2_<lang>_heldout.jsonl`: DO NOT annotate (reserved for fix validation).
- The fields correspond to the taxonomy rules in Task 2 below.

### Self-agreement pass (after your main files)

Annotate the small recheck files exactly the same way, without re-reading
your main answers (ids are disguised, you won't recognize the duplicates):

```sh
rfe exp3-annotate --reviewer 1 data/annotation/task2_de_recheck.jsonl
rfe exp3-self-agreement data/annotation/task2_de_reviewer1.jsonl \
    data/annotation/task2_de_recheck_reviewer1.jsonl --field hallucination_type
```

Target: agreement ≥ 0.90. Below that: pause, re-read the rules, redo the
last block.
