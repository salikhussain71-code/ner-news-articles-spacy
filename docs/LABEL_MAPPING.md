# spaCy (OntoNotes) -> CoNLL-2003 label mapping

| spaCy label | Meaning | CoNLL-2003 |
|---|---|---|
| PERSON | people | **PER** |
| ORG | companies, agencies, institutions | **ORG** |
| GPE | countries, cities, states | **LOC** |
| LOC | non-GPE locations, mountains, rivers | **LOC** |
| FAC | buildings, airports, highways | **LOC** |
| NORP | nationalities, religious/political groups | **MISC** |
| EVENT | named events | **MISC** |
| LANGUAGE | languages | **MISC** |
| WORK_OF_ART | titles of works | **MISC** |
| PRODUCT | products | **MISC** |
| LAW | laws, treaties | **MISC** |
| DATE, TIME, MONEY, PERCENT, QUANTITY, CARDINAL, ORDINAL | numeric / temporal | *dropped* |

The mapping lives in `src/ner_news/labels.py` and is covered by unit tests.
Use `--native-labels` with `extract` to keep spaCy's own 18 labels.
