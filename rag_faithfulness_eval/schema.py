"""Unified sample schema: id, lang, context, claim, faithful."""

from dataclasses import asdict, dataclass

LANGS = ("en", "de", "it")
REQUIRED = ("id", "lang", "context", "claim", "faithful")


@dataclass
class Sample:
    id: str
    lang: str
    context: str
    claim: str
    faithful: bool
    injection: str | None = None  # None = original claim, else injection type used
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def validate_row(row: dict) -> None:
    """Raise ValueError if row violates the schema."""
    missing = [f for f in REQUIRED if f not in row]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if row["lang"] not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {row['lang']!r}")
    if not isinstance(row["faithful"], bool):
        raise ValueError("faithful must be bool")
    if not row["context"] or not row["claim"]:
        raise ValueError("context and claim must be non-empty")


def validate_rows(rows: list[dict]) -> list[str]:
    """Return one error string per invalid row (empty = all valid)."""
    errors = []
    for i, row in enumerate(rows):
        try:
            validate_row(row)
        except ValueError as e:
            errors.append(f"row {i}: {e}")
    return errors
