import pytest

from ner_news.config import GAZETTEER_DIR
from ner_news.models import HybridRecognizer, RuleRecognizer, merge_spans, set_ents
from ner_news.rules import RuleEngine, load_gazetteer


@pytest.fixture(scope="module")
def rec():
    return RuleRecognizer()


def ents(rec, text):
    return [(e.text, e.label_) for e in rec.analyze(text).ents]


def test_gazetteers_load_and_have_no_comments():
    for name in ("countries", "regions", "cities", "organizations", "nationalities", "titles"):
        entries = load_gazetteer(GAZETTEER_DIR / f"{name}.txt")
        assert len(entries) > 10
        assert all(not e.startswith("#") for e in entries)


def test_gazetteer_locations_orgs_misc(rec):
    got = ents(rec, "Pakistani officials in Islamabad met the World Bank in Geneva .")
    assert ("Pakistani", "MISC") in got
    assert ("Islamabad", "LOC") in got
    assert ("World Bank", "ORG") in got
    assert ("Geneva", "LOC") in got


def test_title_is_stripped_from_person(rec):
    assert ents(rec, "Dr. Ayesha Khan arrived .") == [("Ayesha Khan", "PER")]
    assert ents(rec, "President Smith spoke .") == [("Smith", "PER")]


def test_org_suffix_and_prefix_patterns(rec):
    got = ents(rec, "Acme Holdings Ltd bought a stake . The Ministry of Foreign Affairs agreed .")
    assert ("Acme Holdings Ltd", "ORG") in got
    assert ("Ministry of Foreign Affairs", "ORG") in got


def test_longest_match_wins(rec):
    # "University of Oxford" (ORG) must beat the gazetteer city/country inside it
    assert ents(rec, "She studied at University of Oxford .") == [("University of Oxford", "ORG")]
    assert ents(rec, "Floods hit the Indus River .") == [("Indus River", "LOC")]


def test_reporting_verb_heuristic(rec):
    assert ents(rec, "Sarah Johnson said it was fine .") == [("Sarah Johnson", "PER")]


def test_uppercase_headline_variants(rec):
    assert ("ISLAMABAD", "LOC") in ents(rec, "ISLAMABAD , Oct 5 ( Reuters ) - Talks began .")


def test_no_false_positive_on_plain_text(rec):
    assert ents(rec, "the weather was nice and everyone went home early .") == []


def test_engine_works_on_pretokenised_docs():
    engine = RuleEngine()
    doc = engine.make_doc(["Tesla", "opened", "an", "office", "in", "Lahore"])
    assert engine.find(doc) == [(0, 1, "ORG"), (5, 6, "LOC")]


def test_merge_spans_primary_wins():
    primary = [(0, 2, "PER")]
    secondary = [(1, 3, "ORG"), (4, 5, "LOC")]
    assert merge_spans(primary, secondary) == [(0, 2, "PER"), (4, 5, "LOC")]


def test_set_ents_sorted_and_labelled(rec):
    doc = rec.make_doc("Ali visited Lahore")
    set_ents(doc, [(2, 3, "LOC"), (0, 1, "PER")])
    assert [(e.text, e.label_) for e in doc.ents] == [("Ali", "PER"), ("Lahore", "LOC")]


def test_hybrid_priority_validation():
    with pytest.raises(ValueError):
        HybridRecognizer("en_core_web_sm", priority="nonsense")


def test_org_suffix_pattern_does_not_swallow_leading_the(rec):
    assert ents(rec, "The European Union agreed .") == [("European Union", "ORG")]
    assert ents(rec, "Officials of Bayview Corp. agreed .")[0] == ("Bayview Corp.", "ORG")
