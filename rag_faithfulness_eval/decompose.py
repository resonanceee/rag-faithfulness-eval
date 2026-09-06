"""Rule-based claim decomposition + offset-overlap alignment.

Rule-split sentences are verbatim substrings of the RAGTruth output, so gold
labels align exactly by char-offset overlap with hallucination spans. No
embeddings needed for the rule-based path (embedding alignment only makes sense
for LLM-regenerated claims, behind a future flag).
"""

from collections.abc import Iterator

_nlp = None


def _get_nlp():  # lazy: spacy lives in the [models] extra, unit CI must not need it
    import spacy

    global _nlp
    if _nlp is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _nlp = nlp
    return _nlp


def split_claims(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, sentence)] with char offsets into text."""
    doc = _get_nlp()(text)
    return [(s.start_char, s.end_char, s.text) for s in doc.sents if s.text.strip()]


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def align_claim_gold(
    claim_start: int, claim_end: int, spans: list[dict], threshold: float = 0.5
) -> bool:
    """True if >= threshold of the claim's chars are covered by gold hallucination spans."""
    claim_len = max(1, claim_end - claim_start)
    covered = sum(overlaps(claim_start, claim_end, s["start"], s["end"]) for s in spans)
    return covered / claim_len >= threshold


def iter_claims(row: dict, threshold: float = 0.5) -> Iterator[dict]:
    """Yield one claim dict per decomposed sentence of a RAGTruth row."""
    for i, (start, end, text) in enumerate(split_claims(row["output"])):
        yield {
            "id": f"{row['id']}-c{i}",
            "row_id": row["id"],
            "lang": "en",
            "context": row["context"],
            "query": row.get("query", ""),
            "claim": text,
            "start": start,
            "end": end,
            "gold_hallucinated": align_claim_gold(start, end, row["spans"], threshold),
        }
