# Methodology

## 1. Task
Given news text, find *named entities* and classify them as `PER`, `ORG`, `LOC` or `MISC`
(CoNLL-2003 scheme). The project implements and compares three approaches.

## 2. Approaches

### Rule-based (`rule`)
No statistical model. Implemented with spaCy `PhraseMatcher` and `Matcher` on a *blank* English pipeline.

* **Gazetteers** (`src/ner_news/resources/gazetteers/`): countries, regions, cities, organisations,
  nationalities/religions/languages/events, plus a list of honorifics. Matching is case-sensitive; an
  ALL-CAPS variant is added for entries longer than 3 characters so newswire headlines match.
* **Patterns**: `Title + Name` (title stripped), initials + surname, `Name Name said`, organisation
  suffixes/prefixes (*Acme Holdings Ltd*, *University of X*), location patterns (*Lake X*, *Republic of X*).
* **Conflict resolution**: overlapping candidates -> longest span wins, ties broken by rule priority
  (documented in `rules.py`).

### Model-based (`model`)
Pre-trained spaCy pipelines (`en_core_web_sm`, `en_core_web_md`; `lg`/`trf` also work if installed).
The `parser` and `lemmatizer` are excluded because NER does not need them.
Because these models are trained on **OntoNotes 5** (18 types), their labels are mapped to CoNLL-2003
(`docs/LABEL_MAPPING.md`). Numeric/temporal types (DATE, MONEY, ...) are dropped as CoNLL has no such class.

### Hybrid (`hybrid`, `hybrid-rules-first`)
Runs the model, then merges rule entities:
* `hybrid` - model entities are kept; rule entities only fill gaps (recall-oriented).
* `hybrid-rules-first` - rule entities override overlapping model entities (use when gazetteers are trusted).

## 3. Evaluation
* **Pre-tokenised input.** Gold tokens are fed to every system as a spaCy `Doc`, so predictions and gold
  annotations share the same token indices (no alignment errors).
* **Entity-level exact match** (same definition as `conlleval`/`seqeval`): an entity is correct only if
  start, end *and* label match. Precision, recall and F1 are reported per type, micro-, macro- and
  weighted-averaged. The implementation was cross-checked against `seqeval` and gives identical scores.
* **Boundary-only F1** ignores the label, separating "found the span" from "got the type right".
* **Error analysis** classifies every gold/predicted entity as `correct`, `type_error`, `boundary_error`,
  `missed` or `spurious` (`error_details.csv` lists each case with its sentence).
* Speed (tokens/second) is also measured.

## 4. Limitations (read before quoting numbers)
* **Annotation mismatch.** OntoNotes and CoNLL-2003 follow different guidelines (e.g. sports teams are
  `ORG` or `LOC`, nationalities are `MISC`, facilities may be `LOC` or `ORG`), so pre-trained spaCy models
  are not expected to match models trained directly on CoNLL-2003. The label mapping is a documented approximation.
* **Gazetteers are small and hand-written** - they favour precision on common entities and cannot
  match unseen names. Ambiguous entries (*Washington*, *Jordan*, *Georgia*, *Apple*) are labelled with one
  fixed type regardless of context.
* **Bundled sample** is tiny and written by the author; its scores are only a smoke test.
* English only. (Extending to Urdu would need Urdu gazetteers and a multilingual or custom model.)

## 5. Possible extensions
Train a spaCy NER component directly on CoNLL-2003 `train.txt`; add `en_core_web_trf`; fine-tune a
transformer token classifier; CRF baseline; Urdu NER dataset.
