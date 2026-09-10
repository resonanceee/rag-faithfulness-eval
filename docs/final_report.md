# Final report

Figures: docs/figures/ (exp1_arms, exp1_calibration, exp2_langs, exp4_query_delta, exp3_noise)

## Exp 3 highlights

- Judge-vs-gold conflicts that are gold-side noise: confirmed by both reviewers 14/71 (20%), flagged by either 47/71 (66%).
- Dominant real error type: faithful_but_flagged (judge over-flags).
- Top-3 fixes: threshold_fix (16), no_fix_noise (14), decomposition_fix (8). Cover 84% of held-out cases.
