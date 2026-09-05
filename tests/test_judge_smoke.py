"""Real-checkpoint sanity checks. Heavy: downloads ~1.1GB per checkpoint.

Run via: pytest -m smoke  (nightly CI job or manual). Judge sanity threshold:
100% agreement on these hand-checked known-entailed/contradicted pairs.
"""

import pytest

from rag_faithfulness_eval.judge import (
    DEFAULT_CHECKPOINT,
    MULTILINGUAL_CHECKPOINT,
    NLIJudge,
    is_faithful,
)

# (premise, hypothesis, gold_faithful) — hand-checked
KNOWN_PAIRS = [
    ("A man is sleeping on a bench.", "A person is resting.", True),
    ("A man is sleeping on a bench.", "A woman is running a marathon.", False),
    ("Ein Mann schläft auf einer Bank.", "Eine Person ruht sich aus.", True),
    ("Ein Mann schläft auf einer Bank.", "Eine Frau läuft einen Marathon.", False),
    ("Un uomo dorme su una panchina.", "Una persona si sta riposando.", True),
    ("Un uomo dorme su una panchina.", "Una donna corre una maratona.", False),
]


@pytest.mark.smoke
@pytest.mark.parametrize("checkpoint", [DEFAULT_CHECKPOINT, MULTILINGUAL_CHECKPOINT])
def test_judge_agrees_on_known_pairs(checkpoint):
    judge = NLIJudge(checkpoint)
    probs = judge.score_batch([(p, h) for p, h, _ in KNOWN_PAIRS])
    verdicts = [is_faithful(pr) for pr in probs]
    agreement = sum(v == gold for v, (_, _, gold) in zip(verdicts, KNOWN_PAIRS, strict=True))
    assert agreement == len(KNOWN_PAIRS), f"{checkpoint}: {agreement}/{len(KNOWN_PAIRS)}"
