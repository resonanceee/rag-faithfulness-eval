"""Deterministic hallucination injections.

All functions are pure + seeded: same (text, seed) always gives same output.
Return None when no injection target exists in the text.
"""

import random
import re

# Lazy, flat name pools per language. Proper nouns the model has seen in NLI corpora.
ENTITY_POOLS = {
    "en": ["Berlin", "Toyota", "Mozart", "Canada", "Napoleon", "Adidas"],
    "de": ["München", "Siemens", "Goethe", "Österreich", "Bismarck", "Bosch"],
    "it": ["Milano", "Ferrari", "Dante", "Spagna", "Garibaldi", "Barilla"],
}

INJECTIONS = ("entity_swap", "numeric_perturb", "temporal_perturb")

_CAPS = re.compile(r"\b[A-ZÀ-ÜÄÖẞ][\wÀ-ÿäöüß-]*")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DIGIT = re.compile(r"\d")


def entity_swap(text: str, lang: str, rng: random.Random) -> str | None:
    m = _CAPS.search(text, 1)  # skip first char to dodge sentence-initial capital
    if not m:
        return None
    pool = [e for e in ENTITY_POOLS[lang] if e.lower() != m.group(0).lower()]
    if not pool:
        return None
    return text[: m.start()] + rng.choice(pool) + text[m.end() :]


def numeric_perturb(text: str, rng: random.Random) -> str | None:
    m = _DIGIT.search(text)
    if not m:
        return None
    old = m.group(0)
    new = old
    while new == old:
        new = str(rng.randint(0, 9))
    return text[: m.start()] + new + text[m.end() :]


def temporal_perturb(text: str, rng: random.Random) -> str | None:
    m = _YEAR.search(text)
    if not m:
        return None
    year = int(m.group(0))
    delta = rng.choice([-1, 1]) * rng.randint(1, 20)
    new_year = min(max(year + delta, 1000), 2099)
    return text[: m.start()] + str(new_year) + text[m.end() :]


def inject(text: str, kind: str, lang: str, seed: int) -> str | None:
    """Apply one injection. Deterministic given seed."""
    rng = random.Random(f"{seed}:{kind}:{text}")  # stable across runs
    fn = {
        "entity_swap": entity_swap,
        "numeric_perturb": numeric_perturb,
        "temporal_perturb": temporal_perturb,
    }[kind]
    if kind == "entity_swap":
        return fn(text, lang, rng)
    return fn(text, rng)
