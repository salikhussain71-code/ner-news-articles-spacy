"""Visualisations: displacy HTML pages and matplotlib charts."""

from __future__ import annotations

import html
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from spacy import displacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

from .labels import CONLL_LABELS, LABEL_COLORS  # noqa: E402
from .models import Recognizer  # noqa: E402


def _options(docs: Iterable[Doc]) -> dict:
    labels = sorted({e.label_ for d in docs for e in d.ents})
    return {"ents": labels, "colors": {lab: LABEL_COLORS.get(lab, "#dddddd") for lab in labels}}


def render_displacy(docs: list[Doc], path: str | Path, title: str = "Named entities") -> Path:
    """Write a stand-alone displacy HTML page for ``docs``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page = displacy.render(docs, style="ent", page=True, options=_options(docs))
    page = page.replace("<title>displaCy</title>", f"<title>{html.escape(title)}</title>")
    path.write_text(page, encoding="utf-8")
    return path


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}
h1{font-size:1.5rem} h2{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
h3{font-size:.95rem;margin:1rem 0 .3rem;color:#555}
.legend span{display:inline-block;padding:.15rem .6rem;margin-right:.4rem;border-radius:.3rem;font-size:.85rem}
.entities{line-height:2.2!important}
"""


def render_comparison_html(texts: list[str], recognizers: list[Recognizer], path: str | Path) -> Path:
    """Side-by-side page: every article annotated by every approach."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    analysed = {rec.name: rec.analyze_many(texts) for rec in recognizers}
    legend = "".join(f'<span style="background:{LABEL_COLORS[lab]}">{lab}</span>' for lab in CONLL_LABELS)
    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>NER comparison</title><style>{_CSS}</style></head><body>",
        "<h1>Named Entity Recognition - approach comparison</h1>",
        f"<p class='legend'>{legend}</p>",
    ]
    for i, text in enumerate(texts):
        parts.append(f"<h2>Article {i + 1}</h2>")
        for rec in recognizers:
            doc = analysed[rec.name][i]
            frag = displacy.render(doc, style="ent", page=False, options=_options([doc]))
            parts.append(f"<h3>{html.escape(rec.name)} - {len(doc.ents)} entities</h3>{frag}")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_entity_counts(df: pd.DataFrame, path: str | Path, title: str = "Entities per type") -> Path:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[LABEL_COLORS.get(lab, "#999") for lab in counts.index], edgecolor="#444")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title(title)
    ax.set_ylabel("mentions")
    return _save(fig, path)


def plot_top_entities(df: pd.DataFrame, path: str | Path, top: int = 15) -> Path:
    from .extract import entity_counts

    top_df = entity_counts(df, top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top_df) + 1)))
    ax.barh([f"{e} ({lab})" for e, lab in zip(top_df["entity"], top_df["label"])], top_df["count"],
            color=[LABEL_COLORS.get(lab, "#999") for lab in top_df["label"]], edgecolor="#444")
    ax.set_title("Most frequent entities")
    ax.set_xlabel("mentions")
    return _save(fig, path)


def plot_f1_comparison(summary: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(summary) + 3), 4.5))
    x = range(len(summary))
    w = 0.27
    ax.bar([i - w for i in x], summary["precision"], w, label="precision")
    ax.bar(list(x), summary["recall"], w, label="recall")
    ax.bar([i + w for i in x], summary["f1"], w, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["approach"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Entity-level scores (exact match, micro avg)")
    ax.legend()
    return _save(fig, path)


def plot_per_label_f1(per_label: pd.DataFrame, path: str | Path) -> Path:
    pivot = per_label.pivot(index="label", columns="approach", values="f1").reindex(list(CONLL_LABELS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax, edgecolor="#444")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("F1 per entity type")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_error_breakdown(errors: pd.DataFrame, path: str | Path) -> Path:
    pivot = errors.pivot(index="approach", columns="kind", values="count").fillna(0)
    order = [c for c in ("correct", "type_error", "boundary_error", "missed", "spurious") if c in pivot.columns]
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(pivot) + 3), 4.5))
    pivot[order].plot.bar(stacked=True, ax=ax, edgecolor="#444")
    ax.set_ylabel("entities")
    ax.set_title("Error breakdown")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, path)
