# Human agreement spot-check protocol

Pre-experiment calibration: verify the NLI judge agrees with human judgment
before trusting experiment numbers.

## Sample

Per experiment run, draw **30 random scored samples** (10 per language EN/DE/IT,
balanced faithful/unfaithful) from the judge's output.

## Judging

1. Human reads `(context, claim)` without seeing the judge's verdict.
2. Human labels: faithful / unfaithful.
3. Unclear cases count as disagreement (no skip).

## Pass criterion

Agreement >= **26/30 (87%)** per model checkpoint, and >= 8/10 within each
language. Below threshold: stop, inspect failure modes, adjust verdict mapping
or checkpoint before running experiments.

## Logging

Append results to `results/agreement_log.jsonl`:
`{"date", "checkpoint", "n", "agree", "per_lang": {...}, "notes"}`.
