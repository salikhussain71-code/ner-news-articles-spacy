"""Label schemes and the mapping between spaCy (OntoNotes) and CoNLL-2003 labels.

CoNLL-2003 uses four entity types: PER, ORG, LOC, MISC.
spaCy's English pipelines are trained on OntoNotes 5 (18 types), so to evaluate
them on CoNLL-2003 every OntoNotes label must be mapped (or dropped).
"""

from __future__ import annotations

CONLL_LABELS: tuple[str, ...] = ("PER", "ORG", "LOC", "MISC")

# OntoNotes (spaCy native) -> CoNLL-2003.  ``None`` means "not an entity in CoNLL".
SPACY_TO_CONLL: dict[str, str | None] = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "MISC",  # nationalities, religious / political groups
    "EVENT": "MISC",
    "LANGUAGE": "MISC",
    "WORK_OF_ART": "MISC",
    "PRODUCT": "MISC",
    "LAW": "MISC",
    # Numeric / temporal types do not exist in CoNLL-2003 -> dropped.
    "DATE": None,
    "TIME": None,
    "MONEY": None,
    "PERCENT": None,
    "QUANTITY": None,
    "CARDINAL": None,
    "ORDINAL": None,
}

# Aliases found in other NER corpora / Kaggle variants -> CoNLL labels.
LABEL_ALIASES: dict[str, str | None] = {
    "per": "PER",
    "person": "PER",
    "org": "ORG",
    "loc": "LOC",
    "misc": "MISC",
    "geo": "LOC",
    "gpe": "LOC",
    "art": "MISC",
    "eve": "MISC",
    "nat": "MISC",
    "tim": None,
}

# Colours used for displacy and the charts (colour-blind friendly palette).
LABEL_COLORS: dict[str, str] = {
    "PER": "#fbb4ae",
    "ORG": "#b3cde3",
    "LOC": "#ccebc5",
    "MISC": "#decbe4",
    "PERSON": "#fbb4ae",
    "NORP": "#decbe4",
    "FAC": "#ccebc5",
    "GPE": "#ccebc5",
    "DATE": "#fed9a6",
    "TIME": "#fed9a6",
    "MONEY": "#ffffcc",
    "PERCENT": "#e5d8bd",
    "CARDINAL": "#fddaec",
    "ORDINAL": "#f2f2f2",
    "EVENT": "#decbe4",
    "PRODUCT": "#decbe4",
    "LAW": "#decbe4",
    "LANGUAGE": "#decbe4",
    "WORK_OF_ART": "#decbe4",
    "QUANTITY": "#fddaec",
}


def map_spacy_label(label: str) -> str | None:
    """Map a spaCy/OntoNotes label to CoNLL-2003.

    Labels that already belong to the CoNLL scheme are returned unchanged and
    unknown labels are dropped (``None``) so they never count as false hits.
    """
    if label in CONLL_LABELS:
        return label
    return SPACY_TO_CONLL.get(label)


def normalize_label(label: str) -> str | None:
    """Normalise a raw corpus label (``geo``, ``per`` ...) to CoNLL labels."""
    key = label.strip().lower()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    return label.strip().upper()
