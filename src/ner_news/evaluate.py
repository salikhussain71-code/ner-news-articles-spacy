"""Entity-level evaluation (CoNLL / seqeval style) and error analysis.

An entity counts as correct only if its **start, end and label** all match the
gold annotation (exact match) - the same definition as ``conlleval``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .data import Sentence, Span
from .labels import CONLL_LABELS
from .models import Recognizer


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_spans(
    gold: list[Iterable[Span]],
    pred: list[Iterable[Span]],
    labels: Iterable[str] = CONLL_LABELS,
    ignore_type: bool = False,
) -> pd.DataFrame:
    """Per-label and aggregate metrics.

    Returns a DataFrame indexed by label plus ``micro avg`` / ``macro avg`` /
    ``weighted avg`` rows with columns precision, recall, f1, support, tp, fp, fn.
    ``ignore_type=True`` scores boundaries only (all labels collapsed to ``ENT``).
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same number of sentences")
    labels = ["ENT"] if ignore_type else list(labels)
    counts = {lab: [0, 0, 0] for lab in labels}  # tp, fp, fn

    def norm(spans: Iterable[Span]) -> set[Span]:
        out = {(s, e, "ENT" if ignore_type else lab) for s, e, lab in spans}
        return {sp for sp in out if sp[2] in counts}

    for g, p in zip(gold, pred):
        gs, ps = norm(g), norm(p)
        for lab in labels:
            g_l = {x for x in gs if x[2] == lab}
            p_l = {x for x in ps if x[2] == lab}
            counts[lab][0] += len(g_l & p_l)
            counts[lab][1] += len(p_l - g_l)
            counts[lab][2] += len(g_l - p_l)

    rows = []
    for lab in labels:
        tp, fp, fn = counts[lab]
        p, r, f = prf(tp, fp, fn)
        rows.append({"label": lab, "precision": p, "recall": r, "f1": f, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn})
    df = pd.DataFrame(rows).set_index("label")

    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    mp, mr, mf = prf(int(tp), int(fp), int(fn))
    support = df["support"].sum()
    weights = df["support"] / support if support else df["support"] * 0
    df.loc["micro avg"] = [mp, mr, mf, support, tp, fp, fn]
    df.loc["macro avg"] = [df.loc[labels, "precision"].mean(), df.loc[labels, "recall"].mean(), df.loc[labels, "f1"].mean(), support, tp, fp, fn]
    df.loc["weighted avg"] = [
        (df.loc[labels, "precision"] * weights).sum(),
        (df.loc[labels, "recall"] * weights).sum(),
        (df.loc[labels, "f1"] * weights).sum(),
        support, tp, fp, fn,
    ]
    return df


ERROR_TYPES = ("correct", "type_error", "boundary_error", "missed", "spurious")


def error_analysis(
    sentences: list[Sentence], pred: list[Iterable[Span]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categorise every gold / predicted entity.

    * ``correct``        - exact match (span and label)
    * ``type_error``     - same span, wrong label
    * ``boundary_error`` - overlaps a gold entity but boundaries differ
    * ``missed``         - gold entity with no overlapping prediction (false negative)
    * ``spurious``       - prediction overlapping no gold entity (false positive)

    Returns ``(summary, details)``.
    """
    rows = []
    for si, (sent, p) in enumerate(zip(sentences, pred)):
        gold = set(sent.spans)
        pset = set(p)

        def text(sp: Span, tokens=sent.tokens) -> str:
            return " ".join(tokens[sp[0]:sp[1]])

        def overlaps(a: Span, b: Span) -> bool:
            return a[0] < b[1] and b[0] < a[1]

        for g in sorted(gold):
            if g in pset:
                kind, other = "correct", g
            else:
                same_span = [x for x in pset if x[:2] == g[:2]]
                over = [x for x in pset if overlaps(x, g)]
                if same_span:
                    kind, other = "type_error", same_span[0]
                elif over:
                    kind, other = "boundary_error", over[0]
                else:
                    kind, other = "missed", None
            rows.append({"sentence_id": si, "kind": kind, "gold": text(g), "gold_label": g[2],
                         "pred": text(other) if other else "", "pred_label": other[2] if other else "",
                         "context": sent.text})
        for x in sorted(pset):
            if not any(overlaps(x, g) for g in gold):
                rows.append({"sentence_id": si, "kind": "spurious", "gold": "", "gold_label": "",
                             "pred": text(x), "pred_label": x[2], "context": sent.text})
    details = pd.DataFrame(rows, columns=["sentence_id", "kind", "gold", "gold_label", "pred", "pred_label", "context"])
    counts = details["kind"].value_counts().reindex(ERROR_TYPES, fill_value=0)
    summary = counts.rename_axis("kind").reset_index(name="count")
    return summary, details


@dataclass
class EvaluationResult:
    name: str
    report: pd.DataFrame
    boundary_report: pd.DataFrame
    error_summary: pd.DataFrame
    error_details: pd.DataFrame
    seconds: float
    n_sentences: int
    n_tokens: int
    predictions: list[list[Span]]


def evaluate_recognizer(rec: Recognizer, sentences: list[Sentence]) -> EvaluationResult:
    gold = [s.spans for s in sentences]
    start = time.perf_counter()
    pred = rec.predict_tokens([(s.tokens, s.spaces) for s in sentences])
    seconds = time.perf_counter() - start
    error_summary, error_details = error_analysis(sentences, pred)
    return EvaluationResult(
        name=rec.name,
        report=evaluate_spans(gold, pred),
        boundary_report=evaluate_spans(gold, pred, ignore_type=True),
        error_summary=error_summary,
        error_details=error_details,
        seconds=seconds,
        n_sentences=len(sentences),
        n_tokens=sum(len(s.tokens) for s in sentences),
        predictions=pred,
    )


def summary_table(results: list[EvaluationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        micro = r.report.loc["micro avg"]
        rows.append({
            "approach": r.name,
            "precision": micro["precision"], "recall": micro["recall"], "f1": micro["f1"],
            "macro_f1": r.report.loc["macro avg", "f1"],
            "boundary_f1": r.boundary_report.loc["micro avg", "f1"],
            "seconds": r.seconds,
            "tokens_per_sec": r.n_tokens / r.seconds if r.seconds else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def per_label_table(results: list[EvaluationResult]) -> pd.DataFrame:
    frames = []
    for r in results:
        df = r.report.loc[list(CONLL_LABELS)].reset_index()
        df.insert(0, "approach", r.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub markdown table (no ``tabulate`` needed)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_evaluation(
    sentences: list[Sentence],
    recognizers: list[Recognizer],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "evaluation",
    dataset_name: str = "dataset",
    make_plots: bool = True,
) -> list[EvaluationResult]:
    """Evaluate all recognizers and write CSVs, plots and a markdown report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [evaluate_recognizer(rec, sentences) for rec in recognizers]

    summary = summary_table(results)
    per_label = per_label_table(results)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    per_label.to_csv(out / "metrics_per_label.csv", index=False)
    errors = pd.concat(
        [r.error_summary.assign(approach=r.name) for r in results], ignore_index=True
    )[["approach", "kind", "count"]]
    errors.to_csv(out / "error_breakdown.csv", index=False)
    pd.concat(
        [r.error_details.assign(approach=r.name) for r in results], ignore_index=True
    ).to_csv(out / "error_details.csv", index=False)

    if make_plots:
        from .visualize import plot_error_breakdown, plot_f1_comparison, plot_per_label_f1

        plot_f1_comparison(summary, out / "f1_comparison.png")
        plot_per_label_f1(per_label, out / "per_label_f1.png")
        plot_error_breakdown(errors, out / "error_breakdown.png")

    n_ent = int(results[0].report.loc["micro avg", "support"]) if results else 0
    pivot_err = errors.pivot(index="approach", columns="kind", values="count").reset_index()
    report = [
        f"# NER evaluation - {dataset_name}",
        "",
        f"Sentences: **{len(sentences)}** | Gold entities: **{n_ent}** | Exact-match entity-level scores (CoNLL labels PER/ORG/LOC/MISC).",
        "",
        "## Overall (micro-averaged)",
        "",
        markdown_table(summary),
        "",
        "## Per entity type",
        "",
        markdown_table(per_label),
        "",
        "## Error breakdown (counts)",
        "",
        markdown_table(pivot_err, floatfmt="{:.0f}"),
        "",
    ]
    (out / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    return results
