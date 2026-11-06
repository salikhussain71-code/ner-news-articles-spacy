# Dataset: CoNLL-2003 (English)

The internship brief recommends **CoNLL-2003 from Kaggle**. The shared task
(Tjong Kim Sang & De Meulder, 2003) annotates Reuters news with four entity types:
`PER`, `ORG`, `LOC`, `MISC`.

## Download

1. Sign in to Kaggle and search for **"CoNLL003"** / **"CoNLL-2003 English"**. Several uploads exist;
   pick one that contains `train.txt`, `valid.txt` and `test.txt` (4 columns: token, POS, chunk, NER tag).
2. Download and unzip, then place the files like this:

```
data/raw/conll2003/
├── train.txt
├── valid.txt
└── test.txt
```

3. Check that it loads:

```bash
python -m ner_news stats --data data/raw/conll2003/test.txt
```

## Supported file layouts

* **CoNLL text** - one token per line, tag in the *last* column, blank line between sentences,
  `-DOCSTART-` between documents. Both IOB1 (original) and IOB2 tags are accepted.
* **CSV** - token-per-row files with `Sentence #` / `Word` / `Tag` style columns (also numeric
  Hugging Face tag ids). Older tag sets (`geo`, `per`, `org`, `gpe`, `art`, `eve`, `nat`) are mapped to
  CoNLL labels and `tim` is dropped. Use `--format csv` if auto-detection is wrong.

## Licensing note

The English CoNLL-2003 data is built from the Reuters Corpus and is **not** redistributed in this repository
(`data/raw/` is git-ignored). Everyone should download it themselves.

## Reference

Erik F. Tjong Kim Sang and Fien De Meulder. *Introduction to the CoNLL-2003 Shared Task:
Language-Independent Named Entity Recognition.* CoNLL 2003.
