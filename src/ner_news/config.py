"""Project-wide defaults."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODELS: tuple[str, ...] = ("en_core_web_sm", "en_core_web_md")
DEFAULT_APPROACHES: tuple[str, ...] = ("rule", "model", "hybrid")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_CONLL = SAMPLE_DIR / "sample_conll.txt"
SAMPLE_ARTICLES = SAMPLE_DIR / "sample_articles.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

GAZETTEER_DIR = Path(__file__).resolve().parent / "resources" / "gazetteers"

# Components that are not needed for NER and only slow things down.
DISABLED_COMPONENTS: tuple[str, ...] = ("parser", "lemmatizer")
