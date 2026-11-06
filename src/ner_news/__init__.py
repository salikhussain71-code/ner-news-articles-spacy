"""ner_news: Named Entity Recognition (NER) for news articles with spaCy.

The package compares three approaches on the CoNLL-2003 label scheme
(PER, ORG, LOC, MISC):

* ``rule``   - gazetteers + token patterns (no statistical model)
* ``model``  - pre-trained spaCy pipelines (e.g. ``en_core_web_sm`` / ``_md``)
* ``hybrid`` - model predictions completed by high-precision rules
"""

__version__ = "1.0.0"
__author__ = "Salik Hussain"

from .labels import CONLL_LABELS, map_spacy_label  # noqa: F401
