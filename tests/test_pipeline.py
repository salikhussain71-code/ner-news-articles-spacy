import json

import pytest

from ner_news.cli import main
from ner_news.config import SAMPLE_ARTICLES, SAMPLE_CONLL
from ner_news.data import load_dataset
from ner_news.evaluate import run_evaluation
from ner_news.extract import extract_entities, read_articles, split_articles
from ner_news.models import RuleRecognizer, SpacyRecognizer, build_recognizers

from .conftest import requires_sm


def test_split_articles():
    assert split_articles("a b\nc\n\nd e") == ["a b c", "d e"]
    assert len(read_articles(SAMPLE_ARTICLES)) == 3


def test_extract_entities_dataframe():
    df = extract_entities(RuleRecognizer(), read_articles(SAMPLE_ARTICLES))
    assert {"doc_id", "entity", "label", "start_char", "end_char", "approach"} <= set(df.columns)
    assert {"PER", "ORG", "LOC", "MISC"} <= set(df["label"])
    text = read_articles(SAMPLE_ARTICLES)[0]
    row = df[df.doc_id == 0].iloc[0]
    assert text[row.start_char:row.end_char] == row.entity


def test_run_evaluation_writes_all_artifacts(tmp_path):
    sents = load_dataset(SAMPLE_CONLL)
    results = run_evaluation(sents, [RuleRecognizer()], tmp_path, dataset_name="sample")
    assert results[0].report.loc["micro avg", "f1"] > 0.5
    for name in ("metrics_summary.csv", "metrics_per_label.csv", "error_breakdown.csv", "error_details.csv",
                 "evaluation_report.md", "f1_comparison.png", "per_label_f1.png", "error_breakdown.png"):
        assert (tmp_path / name).exists(), name


def test_unknown_approach():
    with pytest.raises(ValueError):
        build_recognizers(["magic"], [])


def test_cli_stats(capsys):
    assert main(["stats"]) == 0
    assert "LOC" in capsys.readouterr().out


def test_cli_missing_data_returns_error(capsys):
    assert main(["stats", "--data", "nope.txt"]) == 2
    assert "CoNLL-2003" in capsys.readouterr().err


def test_cli_extract_rule_only(tmp_path):
    out = tmp_path / "x"
    assert main(["extract", "--approach", "rule", "--output-dir", str(out)]) == 0
    assert (out / "entities.csv").exists() and (out / "entities_displacy.html").exists()
    assert json.loads((out / "entities.json").read_text())


@requires_sm
def test_spacy_model_predicts_conll_labels():
    rec = SpacyRecognizer("en_core_web_sm")
    doc = rec.analyze("Barack Obama visited Microsoft in Seattle on Monday .")
    labels = {e.label_ for e in doc.ents}
    assert labels <= {"PER", "ORG", "LOC", "MISC"}  # DATE etc. are dropped
    assert ("Barack Obama", "PER") in [(e.text, e.label_) for e in doc.ents]


@requires_sm
def test_native_label_scheme_keeps_ontonotes_labels():
    rec = SpacyRecognizer("en_core_web_sm", label_scheme="native")
    labels = {e.label_ for e in rec.analyze("Apple opened an office in Lahore on Monday .").ents}
    assert "GPE" in labels or "DATE" in labels


@requires_sm
def test_hybrid_never_loses_model_entities_when_model_first():
    sents = load_dataset(SAMPLE_CONLL)
    pairs = [(s.tokens, s.spaces) for s in sents]
    model = build_recognizers(["model"], ["en_core_web_sm"])[0].predict_tokens(pairs)
    hybrid = build_recognizers(["hybrid"], ["en_core_web_sm"])[0].predict_tokens(pairs)
    for m, h in zip(model, hybrid):
        assert set(m) <= set(h)


@requires_sm
def test_leading_determiner_is_stripped_by_default():
    text = "Officials met the World Bank in the Hague ."
    stripped = SpacyRecognizer("en_core_web_sm").analyze(text)
    kept = SpacyRecognizer("en_core_web_sm", strip_determiners=False).analyze(text)
    assert all(not e.text.lower().startswith(("the ", "a ", "an ")) for e in stripped.ents)
    assert any(e.text.lower().startswith("the ") for e in kept.ents)
