.PHONY: install model test lint demo evaluate

install:
	pip install -e ".[dev]"

model:
	python -m spacy download en_core_web_sm
	python -m spacy download en_core_web_md

test:
	pytest -q

lint:
	ruff check .

demo:
	python -m ner_news demo

# needs data/raw/conll2003/test.txt (see docs/DATASET.md)
evaluate:
	python -m ner_news evaluate --data data/raw/conll2003/test.txt --output-dir outputs/evaluation_conll2003_test
