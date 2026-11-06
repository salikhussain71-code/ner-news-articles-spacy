"""Recognizers: a common interface over rule-based, spaCy-model and hybrid NER.

Every recognizer returns *token spans* ``(start, end_exclusive, label)`` using
the CoNLL-2003 label scheme (PER/ORG/LOC/MISC) unless ``label_scheme="native"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from .config import DISABLED_COMPONENTS
from .labels import map_spacy_label
from .rules import RuleEngine

TokenSpan = tuple[int, int, str]
LEADING_DETERMINERS = frozenset({"the", "a", "an"})


class ModelNotInstalledError(RuntimeError):
    pass


def load_spacy_model(name: str) -> Language:
    """Load a spaCy pipeline, with a helpful error if it is missing."""
    try:
        return spacy.load(name, exclude=list(DISABLED_COMPONENTS))
    except OSError as exc:
        raise ModelNotInstalledError(
            f"spaCy model '{name}' is not installed. Run:  python -m spacy download {name}"
        ) from exc


def doc_spans(doc: Doc) -> list[TokenSpan]:
    return [(e.start, e.end, e.label_) for e in doc.ents]


def set_ents(doc: Doc, spans: Iterable[TokenSpan]) -> Doc:
    doc.set_ents([Span(doc, s, e, label=lab) for s, e, lab in sorted(spans)], default="outside")
    return doc


def merge_spans(primary: list[TokenSpan], secondary: list[TokenSpan]) -> list[TokenSpan]:
    """Keep every ``primary`` span; add ``secondary`` spans that do not overlap them."""
    merged = list(primary)
    taken = {i for s, e, _ in primary for i in range(s, e)}
    for s, e, lab in sorted(secondary):
        if not any(i in taken for i in range(s, e)):
            merged.append((s, e, lab))
            taken.update(range(s, e))
    return sorted(merged)


class Recognizer(ABC):
    """Base class: build docs, annotate them and return spans."""

    name: str
    nlp: Language

    def make_doc(self, text: str) -> Doc:
        return self.nlp.make_doc(text)

    def doc_from_tokens(self, tokens: list[str], spaces: list[bool] | None = None) -> Doc:
        return Doc(self.nlp.vocab, words=tokens, spaces=spaces)

    @abstractmethod
    def annotate_many(self, docs: Iterable[Doc]) -> Iterator[Doc]:
        """Set ``doc.ents`` on every doc (docs must come from this recognizer's vocab)."""

    def annotate(self, doc: Doc) -> Doc:
        return next(iter(self.annotate_many([doc])))

    def analyze(self, text: str) -> Doc:
        return self.annotate(self.make_doc(text))

    def analyze_many(self, texts: Iterable[str]) -> list[Doc]:
        return list(self.annotate_many(self.make_doc(t) for t in texts))

    def predict_tokens(
        self, sentences: list[tuple[list[str], list[bool] | None]]
    ) -> list[list[TokenSpan]]:
        docs = (self.doc_from_tokens(toks, sp) for toks, sp in sentences)
        return [doc_spans(d) for d in self.annotate_many(docs)]


class RuleRecognizer(Recognizer):
    """Gazetteers + token patterns (no statistical model)."""

    def __init__(self, use_gazetteers: bool = True, use_patterns: bool = True) -> None:
        self.engine = RuleEngine(use_gazetteers=use_gazetteers, use_patterns=use_patterns)
        self.nlp = self.engine.nlp
        self.name = "rule"

    def annotate_many(self, docs: Iterable[Doc]) -> Iterator[Doc]:
        for doc in docs:
            yield set_ents(doc, self.engine.find(doc))


def _pipe(proc, docs: Iterable[Doc], batch_size: int) -> Iterable[Doc]:
    if hasattr(proc, "pipe"):
        return proc.pipe(docs, batch_size=batch_size)
    return (proc(d) for d in docs)


class SpacyRecognizer(Recognizer):
    """A pre-trained spaCy pipeline (``en_core_web_sm`` / ``_md`` / ``_lg`` / ``_trf``)."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        label_scheme: str = "conll",
        batch_size: int = 64,
        strip_determiners: bool = True,
    ) -> None:
        if label_scheme not in {"conll", "native"}:
            raise ValueError("label_scheme must be 'conll' or 'native'")
        self.model_name = model_name
        self.label_scheme = label_scheme
        # OntoNotes-trained models often include a leading "the"/"a"/"an" in the entity
        # ("the World Bank"); CoNLL-2003 never does.  Stripping it aligns the guidelines.
        self.strip_determiners = strip_determiners
        self.batch_size = batch_size
        self.nlp = load_spacy_model(model_name)
        self.name = f"model:{model_name}"

    def _remap(self, doc: Doc) -> Doc:
        if self.label_scheme == "native":
            return doc
        spans = []
        for ent in doc.ents:
            label = map_spacy_label(ent.label_)
            if label is None:
                continue
            start, end = ent.start, ent.end
            if self.strip_determiners:
                while start < end - 1 and doc[start].lower_ in LEADING_DETERMINERS:
                    start += 1
            spans.append((start, end, label))
        return set_ents(doc, spans)

    def annotate_many(self, docs: Iterable[Doc]) -> Iterator[Doc]:
        stream: Iterable[Doc] = docs
        for _, proc in self.nlp.pipeline:
            stream = _pipe(proc, stream, self.batch_size)
        for doc in stream:
            yield self._remap(doc)


class HybridRecognizer(Recognizer):
    """Statistical model completed (or corrected) by high-precision rules.

    ``priority="model"``: model entities are kept, rule entities only fill gaps.
    ``priority="rules"``: rule entities override overlapping model entities.
    """

    def __init__(self, model_name: str = "en_core_web_sm", priority: str = "model", **kwargs) -> None:
        if priority not in {"model", "rules"}:
            raise ValueError("priority must be 'model' or 'rules'")
        self.model = SpacyRecognizer(model_name, label_scheme="conll", **kwargs)
        self.rules = RuleRecognizer()
        self.nlp = self.model.nlp
        self.priority = priority
        suffix = "" if priority == "model" else "-rules-first"
        self.name = f"hybrid{suffix}:{model_name}"

    def annotate_many(self, docs: Iterable[Doc]) -> Iterator[Doc]:
        for doc in self.model.annotate_many(docs):
            model_spans = doc_spans(doc)
            rule_doc = self.rules.doc_from_tokens(
                [t.text for t in doc], [bool(t.whitespace_) for t in doc]
            )
            rule_spans = self.rules.engine.find(rule_doc)
            if self.priority == "model":
                merged = merge_spans(model_spans, rule_spans)
            else:
                merged = merge_spans(rule_spans, model_spans)
            yield set_ents(doc, merged)


def build_recognizers(
    approaches: Iterable[str],
    models: Iterable[str],
    label_scheme: str = "conll",
    strip_determiners: bool = True,
) -> list[Recognizer]:
    """Create recognizers for the requested approaches.

    ``approaches`` may contain ``rule``, ``model``, ``hybrid`` and ``hybrid-rules-first``.
    """
    recs: list[Recognizer] = []
    models = list(models)
    for approach in approaches:
        if approach == "rule":
            recs.append(RuleRecognizer())
        elif approach == "model":
            recs.extend(
                SpacyRecognizer(m, label_scheme=label_scheme, strip_determiners=strip_determiners) for m in models
            )
        elif approach == "hybrid":
            recs.extend(HybridRecognizer(m, priority="model", strip_determiners=strip_determiners) for m in models)
        elif approach == "hybrid-rules-first":
            recs.extend(HybridRecognizer(m, priority="rules", strip_determiners=strip_determiners) for m in models)
        else:
            raise ValueError(f"Unknown approach: {approach!r}")
    return recs
