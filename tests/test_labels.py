from ner_news.labels import CONLL_LABELS, map_spacy_label, normalize_label


def test_spacy_mapping():
    assert map_spacy_label("PERSON") == "PER"
    assert map_spacy_label("GPE") == "LOC"
    assert map_spacy_label("FAC") == "LOC"
    assert map_spacy_label("NORP") == "MISC"
    assert map_spacy_label("ORG") == "ORG"


def test_numeric_labels_are_dropped():
    for label in ("DATE", "MONEY", "CARDINAL", "PERCENT", "TIME"):
        assert map_spacy_label(label) is None
    assert map_spacy_label("SOMETHING_NEW") is None


def test_conll_labels_pass_through():
    for label in CONLL_LABELS:
        assert map_spacy_label(label) == label


def test_normalize_label_aliases():
    assert normalize_label("geo") == "LOC"
    assert normalize_label("per") == "PER"
    assert normalize_label("tim") is None
    assert normalize_label("PER") == "PER"
