"""Rule-based NER: gazetteers (``PhraseMatcher``) + linguistic patterns (``Matcher``).

No statistical model is used, so this approach is fast, fully transparent and
easy to debug.  Every rule has a *priority*; when two candidate entities
overlap, the longer span wins and ties are broken by priority.

Priority table (higher = more trusted):

=====  ===================================================================
10     gazetteer entries (countries, cities, organisations, nationalities)
 8     ``Title + Name``      e.g. *Dr. Ayesha Khan*  -> PER (title stripped)
 7     organisation suffix / prefix patterns (*Acme Holdings Ltd*, *University of X*)
 7     location patterns (*Lake Victoria*, *Republic of X*)
 6     initials + surname    e.g. *J. Smith*
 3     ``Name Name said``    reporting-verb heuristic for PER
=====  ===================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc

from .config import GAZETTEER_DIR

Span = tuple[int, int, str]

GAZETTEER_LABELS: dict[str, str] = {
    "countries": "LOC",
    "regions": "LOC",
    "cities": "LOC",
    "organizations": "ORG",
    "nationalities": "MISC",
}

NAME_RE = r"^[A-Z][a-zA-Z'\u2019\-]+$"  # Title-case word, may contain ' or -
TITLECASE_OR_UPPER_RE = r"^[A-Z][A-Za-z&'\u2019\-]*$"

REPORTING_VERBS = [
    "said", "says", "told", "added", "announced", "stated", "explained",
    "noted", "warned", "confirmed", "denied", "argued", "replied", "claimed",
]  # fmt: skip

NOT_NAME_START = [
    "the", "a", "an", "this", "that", "these", "those", "our", "their", "his",
    "her", "its", "in", "on", "at", "by", "for", "from", "with", "after",
    "before", "during", "according", "however", "meanwhile", "officials",
]  # fmt: skip

ORG_SUFFIXES = [
    "inc", "inc.", "corp", "corp.", "corporation", "ltd", "ltd.", "limited", "llc",
    "plc", "co", "co.", "company", "group", "holdings", "bank", "airlines",
    "airways", "motors", "industries", "technologies", "systems", "foundation",
    "association", "committee", "commission", "council", "ministry", "department",
    "agency", "authority", "institute", "university", "college", "party", "union",
    "league", "federation", "organisation", "organization", "fc", "afc", "f.c.",
]  # fmt: skip

ORG_PREFIXES = [
    "University", "College", "Institute", "Bank", "Ministry", "Department",
    "Court", "Council", "Commission", "Committee", "Agency", "Authority",
    "Federation", "Association", "Organisation", "Organization",
]  # fmt: skip

LOC_SUFFIXES = [
    "River", "Lake", "Sea", "Ocean", "Island", "Islands", "Bay", "Gulf",
    "Strait", "Desert", "Mountains", "Valley", "Peninsula", "Canal",
]  # fmt: skip

LOC_PREFIXES = ["Mount", "Lake", "River", "Cape", "Fort", "Port", "Saint", "San", "Santa"]
LOC_OF_PREFIXES = ["Republic", "Kingdom", "State", "Emirate", "Province", "District", "City"]


@dataclass(frozen=True)
class RuleSpec:
    """A token-pattern rule."""

    name: str
    label: str
    pattern: list[dict]
    priority: int
    strip_titles: bool = False  # drop leading honorifics from the matched span
    trim_last: int = 0  # drop this many trailing tokens (e.g. the reporting verb)


def load_gazetteer(path: str | Path) -> list[str]:
    """Read a gazetteer file (one entry per line, ``#`` comments)."""
    entries: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def load_titles(directory: Path = GAZETTEER_DIR) -> set[str]:
    return {t.lower() for t in load_gazetteer(directory / "titles.txt")}


def build_rule_specs(titles: set[str]) -> list[RuleSpec]:
    """Create the token-pattern rules."""
    title_re = r"^(?:" + "|".join(sorted(map(re.escape, titles))) + r")\.?$"
    name = {"TEXT": {"REGEX": NAME_RE}}
    name_start = {"TEXT": {"REGEX": NAME_RE}, "LOWER": {"NOT_IN": NOT_NAME_START}}
    tc = {"TEXT": {"REGEX": TITLECASE_OR_UPPER_RE}}
    connector = {"LOWER": {"IN": ["and", "&"]}, "OP": "?"}

    return [
        # Dr. Ayesha Khan / President Smith / Mr. Daniel Carter
        RuleSpec(
            "title_name", "PER",
            [{"LOWER": {"REGEX": title_re}}, {"IS_PUNCT": True, "TEXT": ".", "OP": "?"},
             {**name, "OP": "{1,3}"}],
            priority=8, strip_titles=True,
        ),
        # J. Smith / A. B. Khan
        RuleSpec(
            "initials_surname", "PER",
            [{"TEXT": {"REGEX": r"^[A-Z]\.?$"}}, {"TEXT": ".", "OP": "?"},
             {"TEXT": {"REGEX": r"^[A-Z]\.?$"}, "OP": "*"}, {"TEXT": ".", "OP": "?"}, name],
            priority=6,
        ),
        # Sarah Johnson said ...
        *[
            RuleSpec(
                f"name_said_{n}", "PER",
                [name_start, *([name] * (n - 1)), {"LOWER": {"IN": REPORTING_VERBS}}],
                priority=3, trim_last=1,
            )
            for n in (2, 3)
        ],
        # Acme Holdings Ltd / Bayview Corp.
        RuleSpec(
            "org_suffix", "ORG",
            [{**tc, "LOWER": {"NOT_IN": NOT_NAME_START}}, {**tc, "OP": "{0,3}"}, {"LOWER": {"IN": ORG_SUFFIXES}}],
            priority=7,
        ),
        # University of Karachi / Ministry of Foreign Affairs
        RuleSpec(
            "org_of", "ORG",
            [{"TEXT": {"IN": ORG_PREFIXES}}, {"LOWER": "of"}, {**tc, "OP": "+"}, connector,
             {**tc, "OP": "*"}],
            priority=7,
        ),
        # Lake Victoria / Mount Kenya / Fort Worth
        RuleSpec("loc_prefix", "LOC", [{"TEXT": {"IN": LOC_PREFIXES}}, {**tc, "OP": "{1,2}"}], priority=7),
        # Indus River / Bering Strait / Canary Islands
        RuleSpec("loc_suffix", "LOC", [{**tc, "OP": "{1,2}"}, {"TEXT": {"IN": LOC_SUFFIXES}}], priority=7),
        # Republic of Korea / Kingdom of Jordan
        RuleSpec(
            "loc_of", "LOC",
            [{"TEXT": {"IN": LOC_OF_PREFIXES}}, {"LOWER": "of"}, {**tc, "OP": "{1,2}"}],
            priority=7,
        ),
    ]


class RuleEngine:
    """Finds entity candidates in a ``Doc`` and resolves overlaps."""

    GAZ_PRIORITY = 10

    def __init__(
        self,
        gazetteer_dir: Path = GAZETTEER_DIR,
        use_gazetteers: bool = True,
        use_patterns: bool = True,
    ) -> None:
        self.nlp = spacy.blank("en")
        self.titles = load_titles(gazetteer_dir)
        self.use_gazetteers = use_gazetteers
        self.use_patterns = use_patterns

        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        if use_gazetteers:
            for stem, label in GAZETTEER_LABELS.items():
                entries = load_gazetteer(gazetteer_dir / f"{stem}.txt")
                variants = set(entries)
                # ALL-CAPS headlines: "ISLAMABAD", "WORLD BANK" (skip short acronyms)
                variants |= {e.upper() for e in entries if len(e) > 3}
                docs = list(self.nlp.tokenizer.pipe(sorted(variants)))
                self.phrase_matcher.add(f"GAZ_{label}", docs)

        self.specs = build_rule_specs(self.titles)
        self.matcher = Matcher(self.nlp.vocab)
        self._spec_by_name = {spec.name: spec for spec in self.specs}
        if use_patterns:
            for spec in self.specs:
                self.matcher.add(spec.name, [spec.pattern], greedy="LONGEST")

    # ------------------------------------------------------------------ #
    def candidates(self, doc: Doc) -> list[tuple[int, int, str, int]]:
        """Return ``(start, end, label, priority)`` candidates (may overlap)."""
        cands: list[tuple[int, int, str, int]] = []
        if self.use_gazetteers:
            for match_id, start, end in self.phrase_matcher(doc):
                label = self.nlp.vocab.strings[match_id].replace("GAZ_", "")
                cands.append((start, end, label, self.GAZ_PRIORITY))
        if self.use_patterns:
            for match_id, start, end in self.matcher(doc):
                spec = self._spec_by_name[self.nlp.vocab.strings[match_id]]
                end -= spec.trim_last
                if spec.strip_titles:
                    while start < end and (
                        doc[start].lower_.rstrip(".") in self.titles or doc[start].text == "."
                    ):
                        start += 1
                if start < end:
                    cands.append((start, end, spec.label, spec.priority))
        return cands

    def find(self, doc: Doc) -> list[Span]:
        """Resolve overlaps (longest span first, then priority) and return sorted spans."""
        cands = sorted(self.candidates(doc), key=lambda c: (-(c[1] - c[0]), -c[3], c[0]))
        taken: set[int] = set()
        chosen: list[Span] = []
        for start, end, label, _ in cands:
            if any(i in taken for i in range(start, end)):
                continue
            taken.update(range(start, end))
            chosen.append((start, end, label))
        return sorted(chosen)

    def make_doc(self, tokens: list[str], spaces: list[bool] | None = None) -> Doc:
        return Doc(self.nlp.vocab, words=tokens, spaces=spaces)
