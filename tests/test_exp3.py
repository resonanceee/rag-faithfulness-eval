from pathlib import Path

from rag_faithfulness_eval.exp3 import (
    build_recheck,
    cohens_kappa,
    sample_disagreements,
    self_agreement,
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


def test_recheck_and_self_agreement(tmp_path):
    import json

    _write(
        tmp_path / "task2_de.jsonl",
        [
            {
                "id": f"d-{i}",
                "lang": "de",
                "context": "c",
                "claim": "x",
                "gold_label": "unfaithful",
                "judge_verdict": "faithful",
            }
            for i in range(20)
        ],
    )
    _write(tmp_path / "task2_heldout_de.jsonl", [{"id": "h-1"}])
    _write(tmp_path / "task2_de_reviewer1.jsonl", [{"id": "r-1"}])
    made = build_recheck(tmp_path, fraction=0.1)
    assert made == {"task2_de_recheck.jsonl": 2}
    rc = [
        json.loads(line) for line in (tmp_path / "task2_de_recheck.jsonl").read_text().splitlines()
    ]
    assert all(r["id"].endswith("-rc") for r in rc)
    assert not (tmp_path / "task2_de_reviewer1_recheck.jsonl").exists()

    _write(
        tmp_path / "task2_de_main_annot.jsonl",
        [
            {"id": "d-0", "hallucination_type": "entity"},
            {"id": "d-1", "hallucination_type": "numeric"},
        ],
    )
    _write(
        tmp_path / "task2_de_rc_annot.jsonl",
        [
            {"id": "d-0-rc", "hallucination_type": "entity"},
            {"id": "d-1-rc", "hallucination_type": "temporal"},
        ],
    )
    out = self_agreement(
        tmp_path / "task2_de_main_annot.jsonl",
        tmp_path / "task2_de_rc_annot.jsonl",
        "hallucination_type",
    )
    assert out == {"n": 2, "agreement": 0.5}


def test_annotate_interactive_flow(tmp_path):
    import json as J

    from rag_faithfulness_eval.exp3 import annotate

    in_file = tmp_path / "task2_de.jsonl"
    _write(
        in_file,
        [
            {
                "id": "a-1",
                "lang": "de",
                "context": "ctx",
                "claim": "c1",
                "gold_label": "unfaithful",
                "judge_verdict": "faithful",
            },
            {
                "id": "a-2",
                "lang": "de",
                "context": "ctx",
                "claim": "c2",
                "gold_label": "faithful",
                "judge_verdict": "unfaithful",
            },
            {
                "id": "a-3",
                "lang": "de",
                "context": "ctx",
                "claim": "c3",
                "gold_label": "unfaithful",
                "judge_verdict": "unfaithful",
            },
        ],
    )
    # a-1: full annotation (y, type=1, causes='1 3', fix=2); a-2: noise (n); a-3: quit
    answers = iter(["y", "1", "1 3", "2", "n", "q"])
    out = annotate(
        1, in_file, input_fn=lambda prompt="": next(answers), print_fn=lambda *a, **k: None
    )
    assert out["annotated_now"] == 2 and out["remaining"] == 1

    out_lines = (tmp_path / "task2_de_reviewer1.jsonl").read_text().splitlines()
    rows = [J.loads(x) for x in out_lines]
    by_id = {r["id"]: r for r in rows}
    assert by_id["a-1"]["gold_ok"] is True
    assert by_id["a-1"]["hallucination_type"] == "entity"
    assert by_id["a-1"]["cause"] == ["judge_world_knowledge", "bad_decomposition"]
    assert by_id["a-1"]["fix"] == "alignment_fix"
    assert by_id["a-2"] == {**by_id["a-2"], "gold_ok": False, "cause": ["annotation_noise"]}

    # resume: a-1, a-2 skipped; answer a-3 then done
    answers2 = iter(["y", "3", "2", "1"])
    out2 = annotate(
        1, in_file, input_fn=lambda prompt="": next(answers2), print_fn=lambda *a, **k: None
    )
    assert out2["annotated_now"] == 1 and out2["remaining"] == 0
