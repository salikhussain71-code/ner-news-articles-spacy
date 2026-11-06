"""Dataset readers and BIO / span utilities.

Supported inputs (auto-detected from the file extension / header):

* CoNLL-2003 text files (``train.txt`` / ``valid.txt`` / ``test.txt``): one
  token per line, whitespace separated columns, the NER tag is the *last*
  column, blank line between sentences, ``-DOCSTART-`` between documents.
* CSV files with ``Sentence #`` / ``Word`` / ``Tag`` style columns (several
  Kaggle NER uploads use this layout), including numeric Hugging Face tag ids.

Both IOB1 (original CoNLL-2003: ``I-`` starts an entity after ``O``) and IOB2
(``B-`` starts every entity) are handled and normalised to IOB2.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .labels import normalize_label

Span = tuple[int, int, str]  # (start_token, end_token_exclusive, label)

# Order used by the Hugging Face ``conll2003`` dataset for numeric tags.
HF_TAG_IDS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

_NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m", "''", "\u2019s"}
_NO_SPACE_AFTER = {"(", "[", "{", "$", "``"}


@dataclass
class Sentence:
    """One annotated sentence."""

    tokens: list[str]
    tags: list[str]  # IOB2
    doc_id: int = 0
    spaces: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.tags):
            raise ValueError("tokens and tags must have the same length")
        if not self.spaces:
            self.spaces = infer_spaces(self.tokens)

    @property
    def spans(self) -> list[Span]:
        return bio_to_spans(self.tags)

    @property
    def text(self) -> str:
        return "".join(t + (" " if s else "") for t, s in zip(self.tokens, self.spaces)).strip()


# --------------------------------------------------------------------------- #
# Tag helpers
# --------------------------------------------------------------------------- #
def normalize_tag(tag: str) -> str:
    """Normalise a raw tag to ``O`` or ``<B|I>-<LABEL>`` (labels via :func:`normalize_label`)."""
    tag = tag.strip()
    if tag in {"", "O", "o", "_"}:
        return "O"
    if "-" in tag and tag.split("-", 1)[0].upper() in {"B", "I", "E", "S", "L", "U"}:
        prefix, label = tag.split("-", 1)
        prefix = prefix.upper()
        prefix = {"E": "I", "L": "I", "S": "B", "U": "B"}.get(prefix, prefix)
    else:
        prefix, label = "I", tag  # bare label: treat like IOB1
    norm = normalize_label(label)
    if norm is None:
        return "O"
    return f"{prefix}-{norm}"


def iob_to_bio(tags: list[str]) -> list[str]:
    """Convert IOB1 (or mixed) tags to strict IOB2."""
    out: list[str] = []
    prev_label: str | None = None
    for tag in tags:
        if tag == "O":
            out.append("O")
            prev_label = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and prev_label != label:
            prefix = "B"
        out.append(f"{prefix}-{label}")
        prev_label = label
    return out


def bio_to_spans(tags: list[str]) -> list[Span]:
    """Convert BIO/IOB tags to ``(start, end_exclusive, label)`` spans."""
    spans: list[Span] = []
    start: int | None = None
    current: str | None = None
    for i, tag in enumerate([*tags, "O"]):
        if tag == "O":
            prefix, label = "O", None
        else:
            prefix, label = tag.split("-", 1)
        boundary = prefix == "O" or prefix == "B" or (prefix == "I" and label != current)
        if current is not None and boundary:
            spans.append((start, i, current))  # type: ignore[arg-type]
            current = None
        if prefix in {"B", "I"} and current is None:
            start, current = i, label
    return spans


def spans_to_bio(spans: Iterable[Span], length: int) -> list[str]:
    """Convert spans back to IOB2 tags."""
    tags = ["O"] * length
    for start, end, label in sorted(spans):
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
    return tags


def infer_spaces(tokens: list[str]) -> list[bool]:
    """Guess the whitespace after each token (pre-tokenised corpora have none)."""
    spaces = []
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is None:
            spaces.append(False)
        elif tok in _NO_SPACE_AFTER or nxt in _NO_SPACE_BEFORE:
            spaces.append(False)
        else:
            spaces.append(True)
    return spaces


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_conll(path: str | Path) -> list[Sentence]:
    """Read a CoNLL-2003 style text file."""
    sentences: list[Sentence] = []
    tokens: list[str] = []
    tags: list[str] = []
    doc_id = 0

    def flush() -> None:
        nonlocal tokens, tags
        if tokens:
            sentences.append(Sentence(tokens, iob_to_bio(tags), doc_id))
        tokens, tags = [], []

    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                flush()
                continue
            if line.startswith("-DOCSTART-"):
                flush()
                doc_id += 1
                continue
            cols = line.split()
            if len(cols) < 2:
                continue
            tokens.append(cols[0])
            tags.append(normalize_tag(cols[-1]))
    flush()
    return sentences


_TOKEN_COLS = ("word", "token", "tokens", "text")
_TAG_COLS = ("tag", "ner", "ner_tag", "ner_tags", "label", "labels", "entity")
_SENT_COLS = ("sentence #", "sentence_id", "sentence", "sent_id", "sentence#")


def _find_col(columns: dict[str, str], options: tuple[str, ...]) -> str | None:
    for opt in options:
        if opt in columns:
            return columns[opt]
    return None


def read_csv_ner(path: str | Path) -> list[Sentence]:
    """Read a token-per-row CSV (``Sentence #``, ``Word``, ``Tag`` ...)."""
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    cols = {c.strip().lower(): c for c in df.columns}
    tok_col = _find_col(cols, _TOKEN_COLS)
    tag_col = _find_col(cols, _TAG_COLS)
    sent_col = _find_col(cols, _SENT_COLS)
    if tok_col is None or tag_col is None:
        raise ValueError(
            f"Could not find token/tag columns in {path}. Found columns: {list(df.columns)}"
        )
    if sent_col is not None:
        df[sent_col] = df[sent_col].ffill()
        groups = [g for _, g in df.groupby(sent_col, sort=False)]
    else:  # no sentence column: a NaN token row separates sentences
        groups, current = [], []
        for _, row in df.iterrows():
            if pd.isna(row[tok_col]):
                if current:
                    groups.append(pd.DataFrame(current))
                    current = []
            else:
                current.append(row)
        if current:
            groups.append(pd.DataFrame(current))

    sentences: list[Sentence] = []
    for i, group in enumerate(groups):
        tokens = [str(t) for t in group[tok_col].tolist()]
        raw_tags = group[tag_col].tolist()
        tags = []
        for raw in raw_tags:
            if isinstance(raw, (int, float)) and not pd.isna(raw):
                raw = HF_TAG_IDS[int(raw)]
            tags.append(normalize_tag(str(raw)))
        sentences.append(Sentence(tokens, iob_to_bio(tags), i))
    return sentences


def load_dataset(path: str | Path, fmt: str = "auto", limit: int | None = None) -> list[Sentence]:
    """Load a NER dataset. ``fmt`` is ``auto``, ``conll`` or ``csv``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See docs/DATASET.md for how to download CoNLL-2003."
        )
    if fmt == "auto":
        fmt = "csv" if path.suffix.lower() == ".csv" else "conll"
    sentences = read_csv_ner(path) if fmt == "csv" else read_conll(path)
    if limit:
        sentences = sentences[:limit]
    return sentences


def sentences_to_frame(sentences: list[Sentence]) -> pd.DataFrame:
    """Token-level DataFrame (handy for notebooks / EDA)."""
    rows = []
    for si, sent in enumerate(sentences):
        for ti, (tok, tag) in enumerate(zip(sent.tokens, sent.tags)):
            rows.append({"doc_id": sent.doc_id, "sentence_id": si, "token_id": ti, "token": tok, "tag": tag})
    return pd.DataFrame(rows)


def dataset_statistics(sentences: list[Sentence]) -> pd.DataFrame:
    """Entity counts per label (useful for the README / EDA)."""
    counts: dict[str, int] = {}
    for sent in sentences:
        for _, _, label in sent.spans:
            counts[label] = counts.get(label, 0) + 1
    df = pd.DataFrame({"label": list(counts), "entities": list(counts.values())})
    return df.sort_values("entities", ascending=False).reset_index(drop=True)
