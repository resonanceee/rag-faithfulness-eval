import re

from rag_faithfulness_eval.inject import INJECTIONS, inject


def test_entity_swap_changes_capitalized():
    out = inject("A man talks about Berlin with friends.", "entity_swap", "en", 42)
    assert out is not None
    assert "Berlin" not in out
    assert out.startswith("A man talks about ")


def test_entity_swap_none_when_no_caps():
    assert inject("a lowercase sentence.", "entity_swap", "en", 1) is None


def test_numeric_perturb_changes_one_digit():
    out = inject("There are 3 dogs.", "numeric_perturb", "en", 7)
    assert out is not None
    assert out != "There are 3 dogs."
    assert re.fullmatch(r"There are \d dogs\.", out)


def test_temporal_perturb_shifts_year():
    out = inject("The treaty was signed in 1989.", "temporal_perturb", "en", 3)
    assert out is not None
    m = re.search(r"\b(19|20)\d{2}\b", out)
    assert m is not None
    year = int(m.group(0))
    assert year != 1989
    assert 1 <= abs(year - 1989) <= 20


def test_deterministic_same_seed():
    text = "Berlin hosted events in 2015 with about 5000 guests visiting Germany."
    for kind in INJECTIONS:
        assert inject(text, kind, "de", 99) == inject(text, kind, "de", 99)


def test_determinism_independent_of_call_order():
    text = "Mozart was born in Salzburg in 1756 today."
    a = inject(text, "entity_swap", "de", 5)
    inject(text, "numeric_perturb", "de", 5)  # interleave other call
    assert inject(text, "entity_swap", "de", 5) == a
