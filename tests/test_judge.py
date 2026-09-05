import pytest

from rag_faithfulness_eval.cli import main
from rag_faithfulness_eval.judge import (
    DEFAULT_CHECKPOINT,
    MULTILINGUAL_CHECKPOINT,
    CachedJudge,
    is_faithful,
)


class FakeJudge:
    checkpoint = "fake"

    def __init__(self):
        self.calls = 0

    def score_batch(self, pairs):
        self.calls += len(pairs)
        return [{"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} for _ in pairs]

    def score(self, premise, hypothesis):
        return self.score_batch([(premise, hypothesis)])[0]


def test_verdict_mapping():
    assert is_faithful({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05})
    assert not is_faithful({"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7})
    assert is_faithful({"entailment": 0.4, "neutral": 0.2, "contradiction": 0.4})  # tie -> faithful


def test_cache_hits_avoid_judge_calls(tmp_path):
    judge = FakeJudge()
    cached = CachedJudge(judge, tmp_path / "cache.jsonl")
    pairs = [("p1", "h1"), ("p2", "h2")]
    cached.score_batch(pairs)
    assert judge.calls == 2
    CachedJudge(judge, tmp_path / "cache.jsonl").score_batch(pairs)  # fresh instance
    assert judge.calls == 2  # served from disk cache


@pytest.mark.parametrize("checkpoint", [DEFAULT_CHECKPOINT, MULTILINGUAL_CHECKPOINT])
def test_cli_accepts_both_checkpoints(checkpoint, tmp_path, monkeypatch):
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        '{"id":"e1","lang":"en","context":"A cat sleeps.","claim":"An animal sleeps.",'
        '"faithful":true,"injection":null,"source":"t"}\n'
    )
    monkeypatch.setattr("rag_faithfulness_eval.cli.NLIJudge", lambda ck, device=None: FakeJudge())
    rc = main(
        [
            "score",
            "--input",
            str(inp),
            "--out",
            str(tmp_path / "out.jsonl"),
            "--judge-checkpoint",
            checkpoint,
            "--cache",
            str(tmp_path / "c.jsonl"),
        ]
    )
    assert rc == 0


def test_cli_rejects_unknown_checkpoint():
    with pytest.raises(SystemExit):
        main(["score", "--input", "x", "--out", "y", "--judge-checkpoint", "bogus/model"])
