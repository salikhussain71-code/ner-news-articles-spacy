"""Extract entities from raw text into a tidy DataFrame (CSV / JSON)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .models import Recognizer

COLUMNS = ["doc_id", "entity", "label", "start_char", "end_char", "start_token", "end_token", "approach"]


def split_articles(text: str) -> list[str]:
    """Split a text file into articles on blank lines."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [" ".join(p.split()) for p in parts if p.strip()]


def read_articles(path: str | Path) -> list[str]:
    return split_articles(Path(path).read_text(encoding="utf-8"))


def extract_entities(rec: Recognizer, texts: list[str]) -> pd.DataFrame:
    """One row per entity mention."""
    rows = []
    for doc_id, doc in enumerate(rec.analyze_many(texts)):
        for ent in doc.ents:
            rows.append({
                "doc_id": doc_id, "entity": ent.text, "label": ent.label_,
                "start_char": ent.start_char, "end_char": ent.end_char,
                "start_token": ent.start, "end_token": ent.end, "approach": rec.name,
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def entity_counts(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Most frequent (entity, label) pairs."""
    if df.empty:
        return pd.DataFrame(columns=["entity", "label", "count"])
    return (df.groupby(["entity", "label"]).size().reset_index(name="count")
            .sort_values(["count", "entity"], ascending=[False, True]).head(top).reset_index(drop=True))
