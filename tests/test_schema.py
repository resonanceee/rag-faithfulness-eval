import pytest

from rag_faithfulness_eval.schema import Sample, validate_row, validate_rows


def test_valid_sample():
    row = Sample(
        id="en-1", lang="en", context="A cat.", claim="There is an animal.", faithful=True
    ).to_dict()
    validate_row(row)


def test_bad_lang():
    row = Sample(id="x", lang="fr", context="c", claim="c", faithful=True).to_dict()
    with pytest.raises(ValueError, match="lang"):
        validate_row(row)


def test_missing_field():
    with pytest.raises(ValueError, match="missing"):
        validate_row({"id": "x", "lang": "en"})


def test_validate_rows_reports_indices():
    rows = [
        Sample(id="a", lang="en", context="c", claim="c", faithful=True).to_dict(),
        {"id": "b"},
    ]
    errors = validate_rows(rows)
    assert len(errors) == 1
    assert errors[0].startswith("row 1")
