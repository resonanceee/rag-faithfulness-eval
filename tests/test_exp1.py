import json

import pytest

import rag_faithfulness_eval.llm_judge as lj
from rag_faithfulness_eval.decompose import align_claim_gold, overlaps, split_claims
from rag_faithfulness_eval.exp1 import (
    ArmC,
    binary_metrics,
    calibration_bins,
    nli_verdict,
    verdicts_to_binary,
)


# --- decomposition & alignment ---
def test_split_claims_offsets_are_verbatim():
    pytest.importorskip("spacy")  # spacy lives in [models] extra; unit CI skips
    text = "The cat sleeps. A dog barked loudly at noon. It rained."
    claims = split_claims(text)
    assert [text[s:e] for s, e, _ in claims] == [t for _, _, t in claims]
    assert len(claims) == 3


def test_overlap_math():
    assert overlaps(0, 10, 5, 15) == 5
    assert overlaps(0, 10, 12, 20) == 0
    assert overlaps(5, 8, 0, 100) == 3


def test_align_threshold():
    spans = [{"start": 0, "end": 6}]
    assert align_claim_gold(0, 10, spans, 0.5)  # 6/10 covered
    assert not align_claim_gold(0, 20, spans, 0.5)  # 6/20


# --- verdict mapping ---
def test_nli_verdict_argmax():
    assert nli_verdict({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}) == (
        "faithful",
        0.9,
    )
    assert nli_verdict({"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1})[0] == (
        "unverifiable"
    )
    assert nli_verdict({"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8})[0] == (
        "unfaithful"
    )


def test_verdicts_to_binary_neutral_flag():
    vs = ["faithful", "unfaithful", "unverifiable"]
    assert verdicts_to_binary(vs, False) == [False, True, False]
    assert verdicts_to_binary(vs, True) == [False, True, True]


# --- hybrid ---
class FakeLLM:
    def __init__(self):
        self.calls = 0

    def verdict(self, context, claim):
        self.calls += 1
        return "unfaithful"


def test_hybrid_threshold_escalation():
    hybrid = ArmC(None, FakeLLM(), threshold=0.85)
    confident = {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
    shaky = {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2}
    assert hybrid.verdict(confident, "c", "h") == "faithful"  # accepted, no LLM
    assert hybrid.verdict(shaky, "c", "h") == "unfaithful"  # escalated
    assert hybrid.llm.calls == 1
    assert hybrid.proxy_ratio == 0.5


# --- metrics ---
def test_binary_metrics_perfect():
    m = binary_metrics([True, False, True], [True, False, True])
    assert m == {
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "tn": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "accuracy": 1.0,
    }


def test_calibration_bins_empty_bin_safe():
    out = calibration_bins([0.95, 0.92], [True, False])
    assert out["bins"][9]["n"] == 2
    assert out["bins"][0]["n"] == 0
    assert 0 <= out["ece"] <= 1


# --- llm judge (mocked HTTP) ---
def _fake_resp(verdict, tin=100, tout=5):
    return {
        "choices": [{"message": {"content": json.dumps({"verdict": verdict})}}],
        "usage": {"prompt_tokens": tin, "completion_tokens": tout},
    }


def test_llm_judge_accounts_cost(monkeypatch):
    monkeypatch.setattr(lj.OpenRouterJudge, "_call", lambda self, msg, **kw: _fake_resp("faithful"))
    log = lj.CostLog()
    judge = lj.OpenRouterJudge(lj.GLM_FLASH, api_key="k", cost_log=log)
    assert judge.verdict("ctx", "claim") == "faithful"
    assert log.calls == 1 and log.in_tokens == 100
    assert log.usd == pytest.approx((100 * 0.075 + 5 * 0.25) / 1e6)


def test_parse_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        lj._parse_verdict("I think the answer is faithful because...")


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-test\n")
    assert lj.load_api_key(tmp_path / ".env") == "sk-test"
    with pytest.raises(RuntimeError):
        lj.load_api_key(tmp_path / "nope")


def test_premise_of_modes():
    c = {"context": "Some passages.", "query": "What is X?", "claim": "X is Y."}
    from rag_faithfulness_eval.exp1 import premise_of

    assert premise_of(c, "plain") == "Some passages."
    assert premise_of(c, "query") == "QUESTION: What is X?\nPASSAGES: Some passages."
