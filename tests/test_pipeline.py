import rag_faithfulness_eval.pipeline as pipeline
from rag_faithfulness_eval.pipeline import build_samples, report
from rag_faithfulness_eval.schema import validate_rows


def test_build_samples_balanced_and_valid(monkeypatch):
    calls = {}

    def fake_fetch(n, seed, tag):
        calls[tag] = n
        base = [
            ("Berlin hosted expo in 2019.", "The expo was in Berlin in 2019."),  # all kinds apply
            ("A man sleeps.", "A person sleeps."),  # no kind applies -> must be skipped
        ]
        return base * ((n // 2) + 1)

    monkeypatch.setattr(pipeline, "_en_pairs", lambda n, seed: fake_fetch(n, seed, "en"))
    monkeypatch.setattr(pipeline, "_de_pairs", lambda n, seed: fake_fetch(n, seed, "de"))
    monkeypatch.setattr(pipeline, "_it_pairs", lambda n, seed: fake_fetch(n, seed, "it"))

    samples = build_samples(n_per_lang=5, seed=0)
    rows = [s.to_dict() for s in samples]
    assert validate_rows(rows) == []

    rep = report(samples)
    assert rep["per_lang"] == {"en": 10, "de": 10, "it": 10}
    assert rep["faithful"] == {True: 15, False: 15}  # exact 1:1 balance
    assert set(rep["injection_types"]) <= {"entity_swap", "numeric_perturb", "temporal_perturb"}

    # every unfaithful claim differs from its paired faithful claim
    by_pair = {}
    for s in samples:
        pid = s.id.split("-")[0] + "-" + s.id.split("-")[1]
        by_pair.setdefault(pid, []).append(s)
    for pair in by_pair.values():
        assert len(pair) == 2
        assert pair[0].claim != pair[1].claim


def test_build_samples_deterministic(monkeypatch):
    def fake_fetch(n, seed):
        return [("Berlin expo 2019 happened.", "Expo in Berlin in 2019 with 500 guests.")] * n

    for fn in ("_en_pairs", "_de_pairs", "_it_pairs"):
        monkeypatch.setattr(pipeline, fn, fake_fetch)
    a = [s.to_dict() for s in build_samples(n_per_lang=4, seed=7)]
    b = [s.to_dict() for s in build_samples(n_per_lang=4, seed=7)]
    assert a == b
