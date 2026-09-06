from pathlib import Path

from rag_faithfulness_eval.exp3 import (
    cohens_kappa,
    sample_disagreements,
)


def _write(path: Path, rows: list[dict]):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_cohens_kappa_perfect_and_chance(tmp_path):
    _write(
        tmp_path / "a.jsonl",
        [
            {"id": "1", "hallucination_type": "entity"},
            {"id": "2", "hallucination_type": "numeric"},
            {"id": "3", "hallucination_type": "entity"},
        ],
    )
    _write(
        tmp_path / "b.jsonl",
        [
            {"id": "1", "hallucination_type": "entity"},
            {"id": "2", "hallucination_type": "numeric"},
            {"id": "3", "hallucination_type": "entity"},
        ],
    )
    out = cohens_kappa(tmp_path / "a.jsonl", tmp_path / "b.jsonl", "hallucination_type")
    assert out == {"n": 3, "agreement": 1.0, "kappa": 1.0}


def test_cohens_kappa_partial(tmp_path):
    _write(tmp_path / "a.jsonl", [{"id": str(i), "t": "x"} for i in range(4)])
    _write(
        tmp_path / "b.jsonl",
        [
            {"id": "0", "t": "x"},
            {"id": "1", "t": "y"},
            {"id": "2", "t": "x"},
            {"id": "3", "t": "y"},
        ],
    )
    out = cohens_kappa(tmp_path / "a.jsonl", tmp_path / "b.jsonl", "t")
    assert out["n"] == 4
    assert out["agreement"] == 0.5
    assert 0 <= out["kappa"] < 1


def test_disagreement_sampling(tmp_path, monkeypatch):
    # fake exp1 arm files + fake claim map + fake exp2 arm files + samples.jsonl
    exp1 = tmp_path / "exp1"
    exp2 = tmp_path / "exp2"
    exp1.mkdir()
    exp2.mkdir()
    _write(
        exp1 / "arm_A.jsonl",
        [
            {"id": "rt-1-c0", "verdict": "faithful", "gold_hallucinated": True},
            {"id": "rt-2-c0", "verdict": "unfaithful", "gold_hallucinated": False},
            {"id": "rt-3-c0", "verdict": "faithful", "gold_hallucinated": False},
        ],
    )
    _write(
        exp2 / "arm_A.jsonl",
        [
            {
                "id": "de-00001-entity_swap",
                "verdict": "faithful",
                "gold_hallucinated": True,
                "lang": "de",
                "injection": "entity_swap",
            },
        ],
    )
    import json as j

    (tmp_path / "samples.jsonl").write_text(
        j.dumps(
            {
                "id": "de-00001-entity_swap",
                "lang": "de",
                "context": "Kontext.",
                "claim": "Behauptung.",
                "faithful": False,
                "injection": "entity_swap",
                "source": "x",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "rag_faithfulness_eval.exp3._claim_map",
        lambda: {
            "rt-1-c0": {"context": "ctx1", "claim": "cl1"},
            "rt-2-c0": {"context": "ctx2", "claim": "cl2"},
        },
    )
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    (Path("data") / "samples.jsonl").write_text((tmp_path / "samples.jsonl").read_text())

    out = sample_disagreements(
        exp1_dir=exp1, exp2_dir=exp2, out_dir=tmp_path / "ann", n_exp1=10, n_exp2=10
    )
    assert out["sampled"] >= 2  # 2 EN disagreements + 1 DE disagreement, minus ~20% heldout
    files = list((tmp_path / "ann").glob("task2_*.jsonl"))
    assert files
