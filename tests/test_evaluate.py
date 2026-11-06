import pytest

from ner_news.data import Sentence
from ner_news.evaluate import error_analysis, evaluate_spans, markdown_table, prf


def test_prf():
    assert prf(2, 0, 0) == (1.0, 1.0, 1.0)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f = prf(1, 1, 2)
    assert (p, r) == (0.5, 1 / 3)
    assert f == pytest.approx(0.4)


def test_perfect_prediction():
    gold = [[(0, 2, "PER"), (3, 4, "LOC")]]
    df = evaluate_spans(gold, gold)
    assert df.loc["micro avg", "f1"] == 1.0
    assert df.loc["PER", "support"] == 1


def test_partial_prediction_and_per_label():
    gold = [[(0, 2, "PER"), (3, 4, "LOC")], [(0, 1, "ORG")]]
    pred = [[(0, 2, "PER"), (3, 4, "ORG")], [(0, 1, "ORG"), (2, 3, "LOC")]]
    df = evaluate_spans(gold, pred)
    assert df.loc["PER", "f1"] == 1.0
    assert df.loc["LOC", "recall"] == 0.0 and df.loc["LOC", "precision"] == 0.0
    assert df.loc["ORG", "tp"] == 1 and df.loc["ORG", "fp"] == 1
    assert df.loc["micro avg", "tp"] == 2
    assert df.loc["micro avg", "fp"] == 2 and df.loc["micro avg", "fn"] == 1


def test_boundary_only_ignores_type():
    gold = [[(0, 2, "PER")]]
    pred = [[(0, 2, "ORG")]]
    assert evaluate_spans(gold, pred).loc["micro avg", "f1"] == 0.0
    assert evaluate_spans(gold, pred, ignore_type=True).loc["micro avg", "f1"] == 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_spans([[]], [[], []])


def test_error_analysis_categories():
    sent = Sentence(
        tokens=["A", "B", "C", "D", "E", "F", "G", "H"],
        tags=["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "I-ORG", "O"],
    )
    pred = [[(0, 2, "PER"),   # correct
             (3, 4, "ORG"),   # type error (gold LOC)
             (5, 6, "ORG"),   # boundary error (gold 5-7)
             (7, 8, "MISC")]]  # spurious
    summary, details = error_analysis([sent], pred)
    counts = summary.set_index("kind")["count"].to_dict()
    assert counts == {"correct": 1, "type_error": 1, "boundary_error": 1, "missed": 0, "spurious": 1}
    # a gold entity with no prediction at all is "missed"
    summary, _ = error_analysis([sent], [[]])
    assert summary.set_index("kind")["count"]["missed"] == 3


def test_markdown_table():
    import pandas as pd

    md = markdown_table(pd.DataFrame({"a": ["x"], "b": [0.12345]}))
    assert md.splitlines()[0] == "| a | b |"
    assert "0.123" in md
