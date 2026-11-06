import pandas as pd

from ner_news.config import SAMPLE_CONLL
from ner_news.data import (
    bio_to_spans,
    dataset_statistics,
    infer_spaces,
    iob_to_bio,
    load_dataset,
    normalize_tag,
    read_conll,
    spans_to_bio,
)


def test_bio_to_spans_basic():
    tags = ["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "I-ORG", "I-ORG"]
    assert bio_to_spans(tags) == [(0, 2, "PER"), (3, 4, "LOC"), (5, 8, "ORG")]


def test_adjacent_entities_of_same_type_are_split_by_b():
    assert bio_to_spans(["B-LOC", "B-LOC"]) == [(0, 1, "LOC"), (1, 2, "LOC")]


def test_iob1_conversion():
    # original CoNLL-2003 uses I- for the first token unless two same-type entities touch
    iob1 = ["I-ORG", "I-ORG", "O", "I-PER", "B-PER", "I-PER"]
    assert iob_to_bio(iob1) == ["B-ORG", "I-ORG", "O", "B-PER", "B-PER", "I-PER"]


def test_roundtrip_spans_bio():
    spans = [(0, 2, "PER"), (4, 5, "LOC")]
    assert bio_to_spans(spans_to_bio(spans, 6)) == spans


def test_normalize_tag():
    assert normalize_tag("B-geo") == "B-LOC"
    assert normalize_tag("I-per") == "I-PER"
    assert normalize_tag("B-tim") == "O"
    assert normalize_tag("O") == "O"
    assert normalize_tag("PER") == "I-PER"


def test_infer_spaces():
    toks = ["He", "said", ",", "\"", "no", "\"", "."]
    spaces = infer_spaces(toks)
    assert spaces[0] and not spaces[1] and not spaces[-1]  # no space before ',' and after the last token


def test_read_sample_conll():
    sents = read_conll(SAMPLE_CONLL)
    assert len(sents) == 15
    assert {s.doc_id for s in sents} == {1, 2}
    assert sents[0].spans == [(1, 3, "ORG"), (9, 10, "LOC")]
    assert sents[0].text.startswith("The United Nations said")


def test_read_conll_iob1_and_two_columns(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("-DOCSTART- -X- O O\n\nEU NNP I-NP I-ORG\nrejects VBZ I-VP O\nGerman JJ I-NP I-MISC\n\nPeter I-PER\nBlackburn I-PER\n")
    sents = read_conll(f)
    assert sents[0].spans == [(0, 1, "ORG"), (2, 3, "MISC")]
    assert sents[1].spans == [(0, 2, "PER")]


def test_read_csv_kaggle_layout(tmp_path):
    f = tmp_path / "t.csv"
    pd.DataFrame({
        "Sentence #": ["Sentence: 1", None, None, "Sentence: 2", None],
        "Word": ["Thousands", "in", "London", "Obama", "spoke"],
        "Tag": ["O", "O", "B-geo", "B-per", "O"],
    }).to_csv(f, index=False)
    sents = load_dataset(f)
    assert len(sents) == 2
    assert sents[0].spans == [(2, 3, "LOC")]
    assert sents[1].spans == [(0, 1, "PER")]


def test_read_csv_numeric_tags(tmp_path):
    f = tmp_path / "t.csv"
    pd.DataFrame({"sentence_id": [0, 0, 0], "tokens": ["Karachi", "is", "big"], "ner_tags": [5, 0, 0]}).to_csv(f, index=False)
    assert load_dataset(f)[0].spans == [(0, 1, "LOC")]


def test_dataset_statistics():
    stats = dataset_statistics(read_conll(SAMPLE_CONLL)).set_index("label")["entities"]
    assert stats["LOC"] > 0 and stats["ORG"] > 0 and stats["PER"] > 0 and stats["MISC"] > 0


def test_missing_file_message():
    import pytest

    with pytest.raises(FileNotFoundError, match="DATASET"):
        load_dataset("does_not_exist.txt")
