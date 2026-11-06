"""Command line interface:  ``python -m ner_news <command>``  (or ``ner-news``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import spacy

from . import __version__
from .config import (
    DEFAULT_APPROACHES,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR,
    SAMPLE_ARTICLES,
    SAMPLE_CONLL,
)
from .data import dataset_statistics, load_dataset
from .evaluate import markdown_table, run_evaluation, summary_table
from .extract import entity_counts, extract_entities, read_articles
from .models import ModelNotInstalledError, build_recognizers

ALL_APPROACHES = (*DEFAULT_APPROACHES, "hybrid-rules-first")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--approaches", nargs="+", default=list(DEFAULT_APPROACHES), choices=ALL_APPROACHES,
                   help="which NER approaches to run (default: rule model hybrid)")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                   help="spaCy pipelines for the model/hybrid approaches (default: en_core_web_sm en_core_web_md)")
    p.add_argument("--keep-determiners", action="store_true",
                   help="do not strip a leading 'the/a/an' from model entities (CoNLL guideline alignment)")


def cmd_info(args: argparse.Namespace) -> int:
    print(f"ner-news {__version__} | spaCy {spacy.__version__}")
    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"):
        ok = spacy.util.is_package(name)
        version = spacy.util.get_package_version(name) if ok else "-"
        print(f"  {name:<18} {'installed ' + version if ok else 'not installed  (python -m spacy download ' + name + ')'}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    sentences = load_dataset(args.data, args.format, args.limit or None)
    print(f"{len(sentences)} sentences, {sum(len(s.tokens) for s in sentences)} tokens")
    print(dataset_statistics(sentences).to_string(index=False))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    sentences = load_dataset(args.data, args.format, args.limit or None)
    recs = build_recognizers(args.approaches, args.models, strip_determiners=not args.keep_determiners)
    print(f"Evaluating {len(recs)} recognizer(s) on {len(sentences)} sentences ...")
    results = run_evaluation(sentences, recs, args.output_dir, dataset_name=Path(args.data).name)
    print(markdown_table(summary_table(results)))
    print(f"\nFull report written to {Path(args.output_dir) / 'evaluation_report.md'}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from .visualize import plot_entity_counts, plot_top_entities, render_displacy

    texts = [args.text] if args.text else read_articles(args.input)
    scheme = "native" if args.native_labels else "conll"
    rec = build_recognizers(
        [args.approach], [args.model], label_scheme=scheme, strip_determiners=not args.keep_determiners
    )[0]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = extract_entities(rec, texts)
    df.to_csv(out / "entities.csv", index=False)
    df.to_json(out / "entities.json", orient="records", indent=2, force_ascii=False)
    docs = rec.analyze_many(texts)
    render_displacy(docs, out / "entities_displacy.html", title=f"NER - {rec.name}")
    if not df.empty:
        plot_entity_counts(df, out / "entity_counts.png", f"Entities per type ({rec.name})")
        plot_top_entities(df, out / "top_entities.png")
    print(f"{len(df)} entities from {len(texts)} text(s) using {rec.name}")
    print(entity_counts(df, 10).to_string(index=False))
    print(f"\nWrote entities.csv / entities.json / entities_displacy.html to {out}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    from .visualize import render_comparison_html

    texts = [args.text] if args.text else read_articles(args.input)
    recs = build_recognizers(args.approaches, args.models, strip_determiners=not args.keep_determiners)
    path = render_comparison_html(texts, recs, Path(args.output_dir) / "comparison.html")
    print(f"Wrote {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run everything on the bundled sample data (smoke test, not a benchmark)."""
    out = Path(args.output_dir)
    models = args.models
    print("== 1/3 extraction + displacy ==")
    cmd_extract(argparse.Namespace(text=None, input=str(SAMPLE_ARTICLES), approach="hybrid",
                                   model=models[0], native_labels=False, keep_determiners=False,
                                   output_dir=str(out / "extraction")))
    print("\n== 2/3 comparison page ==")
    cmd_visualize(argparse.Namespace(text=None, input=str(SAMPLE_ARTICLES), approaches=list(DEFAULT_APPROACHES), keep_determiners=False,
                                     models=models[:2], output_dir=str(out / "visualizations")))
    print("\n== 3/3 evaluation on bundled sample (tiny: smoke test only) ==")
    cmd_evaluate(argparse.Namespace(data=str(SAMPLE_CONLL), format="conll", limit=0,
                                    approaches=list(ALL_APPROACHES), models=models, keep_determiners=False,
                                    output_dir=str(out / "evaluation_sample")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ner-news", description="Named Entity Recognition for news articles with spaCy.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show versions and installed spaCy models").set_defaults(func=cmd_info)

    p = sub.add_parser("stats", help="entity statistics of a dataset")
    p.add_argument("--data", default=str(SAMPLE_CONLL))
    p.add_argument("--format", default="auto", choices=["auto", "conll", "csv"])
    p.add_argument("--limit", type=int, default=0, help="only the first N sentences (0 = all)")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("evaluate", help="evaluate approaches on a labelled dataset (e.g. CoNLL-2003 test)")
    p.add_argument("--data", default=str(SAMPLE_CONLL))
    p.add_argument("--format", default="auto", choices=["auto", "conll", "csv"])
    p.add_argument("--limit", type=int, default=0, help="only the first N sentences (0 = all)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "evaluation"))
    _add_model_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("extract", help="extract entities from raw articles into CSV/JSON + displacy HTML")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", default=str(SAMPLE_ARTICLES), help="text file, articles separated by blank lines")
    src.add_argument("--text", help="a single text passed on the command line")
    p.add_argument("--approach", default="hybrid", choices=ALL_APPROACHES)
    p.add_argument("--model", default=DEFAULT_MODELS[0])
    p.add_argument("--native-labels", action="store_true", help="keep spaCy's 18 OntoNotes labels (model approach only)")
    p.add_argument("--keep-determiners", action="store_true", help="do not strip a leading 'the/a/an' from model entities")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "extraction"))
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("visualize", help="side-by-side HTML comparison of approaches")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", default=str(SAMPLE_ARTICLES))
    src.add_argument("--text")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "visualizations"))
    _add_model_args(p)
    p.set_defaults(func=cmd_visualize)

    p = sub.add_parser("demo", help="run extraction, visualisation and evaluation on the bundled sample")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    pd.set_option("display.width", 160)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ModelNotInstalledError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
