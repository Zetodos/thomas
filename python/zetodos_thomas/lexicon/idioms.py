# SPDX-License-Identifier: Apache-2.0
"""Multiword Sahidic Coptic idioms and empirical Koine Greek case-valency frames."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.schemas.linguistic import CopticMorpheme, GreekUnit


@dataclass(frozen=True)
class CopticIdiomRule:
    """Multiword Coptic lemma sequence -> target Koine lemma, POS & valency."""

    pattern_lemmas: tuple[str, ...]
    greek_lemma: str
    greek_pos: str
    required_case: str | None = None
    strip_preposition_after: str | None = None
    notes: str = ""


# Curated Sahidic phrasal verbs & multiword expressions + empirical MWE extractions
_CURATED_IDIOMS: tuple[CopticIdiomRule, ...] = (
    # "to find" = ϩⲉ + ⲉ (literally "fall upon/to"), takes Accusative in Koine εὑρίσκω
    CopticIdiomRule(
        ("ϩⲉ", "ⲉ"), "εὑρίσκω", "V-", required_case="A", strip_preposition_after="ⲉ"
    ),
    # "taste death" = ϫⲓ + ϯⲡⲉ (literal "take taste"), Koine γεύομαι (+ Genitive!)
    CopticIdiomRule(
        ("ϫⲓ", "ϯⲡⲉ"), "γεύομαι", "V-", required_case="G", strip_preposition_after="ⲛ"
    ),
    CopticIdiomRule(
        ("ϫⲓϯⲡⲉ",), "γεύομαι", "V-", required_case="G", strip_preposition_after="ⲛ"
    ),
    # "behold" = ⲉⲓⲥ + ϩⲏⲏⲧⲉ
    CopticIdiomRule(("ⲉⲓⲥ", "ϩⲏⲏⲧⲉ"), "ἰδού", "I-"),
    # "reign as king" = ⲣ + ⲣⲣⲟ / ⲣⲣⲣⲟ / ⲉⲓⲣⲉ + ⲣⲣⲟ
    CopticIdiomRule(("ⲉⲓⲣⲉ", "ⲣⲣⲟ"), "βασιλεύω", "V-"),
    CopticIdiomRule(("ⲣⲣⲣⲟ",), "βασιλεύω", "V-"),
    CopticIdiomRule(("ⲣ", "ϣⲡⲏⲣⲉ"), "θαυμάζω", "V-"),
    CopticIdiomRule(("ⲣϣⲡⲏⲣⲉ",), "θαυμάζω", "V-"),
    CopticIdiomRule(("ϥⲓ", "ⲣⲟⲟⲩϣ"), "μεριμνάω", "V-"),
    CopticIdiomRule(("ϥⲓⲣⲟⲟⲩϣ",), "μεριμνάω", "V-"),
    CopticIdiomRule(("ϫⲓ", "ϭⲟⲗ"), "ψεύδομαι", "V-"),
    CopticIdiomRule(("ϫⲓϭⲟⲗ",), "ψεύδομαι", "V-"),
    CopticIdiomRule(("ⲧⲁϣⲉ", "ⲟⲉⲓϣ"), "κηρύσσω", "V-"),
    CopticIdiomRule(("ⲧⲁϣⲉⲟⲉⲓϣ",), "κηρύσσω", "V-"),
    CopticIdiomRule(("ⲛⲁⲩ", "ⲉⲃⲟⲗ"), "διαβλέπω", "V-"),
    CopticIdiomRule(("ⲱϩⲉ", "ⲉ", "ⲣⲁⲧ"), "ἵστημι", "V-"),
    # "little child" = ϣⲏⲣⲉ + ϣⲏⲙ -> παιδίον / νήπιος
    CopticIdiomRule(("ϣⲏⲣⲉ", "ϣⲏⲙ"), "παιδίον", "N-"),
    # Interrogative & adverbial Sahidic periphrases
    CopticIdiomRule(("ⲁϣ", "ⲡⲉ", "ⲡ", "ϩⲉ"), "πῶς", "D-"),
    CopticIdiomRule(("ⲁϣ", "ⲛ", "ϩⲉ"), "πῶς", "D-"),
    CopticIdiomRule(("ⲁϣ", "ⲛ", "ϩⲟⲟⲩ"), "πότε", "D-"),
    CopticIdiomRule(("ϩⲁϩ", "ⲛ", "ⲥⲟⲡ"), "πολλάκις", "D-"),
    # Standard Sahidic comparative / spatial prepositional compounds
    CopticIdiomRule(("ⲛ", "ⲡ", "ϩⲉ", "ⲛ"), "ὡς", "C-"),
    CopticIdiomRule(
        ("ⲛ", "ⲡ", "ⲙⲧⲟ", "ⲉⲃⲟⲗ", "ⲛ"), "ἔμπροσθεν", "P-", required_case="G"
    ),
    CopticIdiomRule(("ⲛ", "ⲡ", "ⲙⲧⲟ", "ⲛ"), "ἔμπροσθεν", "P-", required_case="G"),
    # Phrasal verbs with adverbial particles (ⲉⲃⲟⲗ, ⲉϩⲟⲩⲛ, ⲉϩⲣⲁⲓ, ⲉⲡⲉⲥⲏⲧ)
    CopticIdiomRule(("ϭⲱⲗⲡ", "ⲉⲃⲟⲗ"), "ἀποκαλύπτω", "V-", required_case="A"),
    CopticIdiomRule(("ⲕⲱ", "ⲉⲃⲟⲗ"), "ἀφίημι", "V-", required_case="A"),
    CopticIdiomRule(("ⲃⲱⲕ", "ⲉϩⲟⲩⲛ"), "εἰσέρχομαι", "V-"),
    CopticIdiomRule(("ⲃⲱⲕ", "ⲉⲃⲟⲗ"), "ἐξέρχομαι", "V-"),
    CopticIdiomRule(("ⲉⲓ", "ⲉⲃⲟⲗ"), "ἐξέρχομαι", "V-"),
    CopticIdiomRule(("ⲉⲓ", "ⲉϩⲟⲩⲛ"), "εἰσέρχομαι", "V-"),
    CopticIdiomRule(("ⲛⲟⲩϫⲉ", "ⲉⲃⲟⲗ"), "ἐκβάλλω", "V-", required_case="A"),
    CopticIdiomRule(("ϥⲓ", "ⲉϩⲣⲁⲓ"), "αἴρω", "V-", required_case="A"),
    CopticIdiomRule(("ⲧⲱⲟⲩⲛ", "ⲉϩⲣⲁⲓ"), "ἐγείρω", "V-"),
    CopticIdiomRule(("ϩⲙⲟⲟⲥ", "ⲉϩⲣⲁⲓ"), "κάθημαι", "V-"),
    CopticIdiomRule(("ⲟⲩⲱⲛϩ", "ⲉⲃⲟⲗ"), "φανερόω", "V-", required_case="A"),
    CopticIdiomRule(("ⲕⲧⲟ", "ⲉⲃⲟⲗ"), "ἀποστρέφω", "V-"),
    CopticIdiomRule(
        ("ϣⲓⲛⲉ", "ⲛⲥⲁ"), "ζητέω", "V-", required_case="A", strip_preposition_after="ⲛⲥⲁ"
    ),
    CopticIdiomRule(("ϭⲱϣⲧ", "ⲉⲃⲟⲗ"), "προσδοκάω", "V-"),
    CopticIdiomRule(("ϯ", "ⲟⲩⲱ"), "βλαστάνω", "V-"),
    CopticIdiomRule(("ⲣ", "ϩⲟⲧⲉ"), "φοβέομαι", "V-"),
    CopticIdiomRule(("ⲣϩⲟⲧⲉ",), "φοβέομαι", "V-"),
    CopticIdiomRule(("ⲟⲩⲱϣⲃ", "ⲉϥϫⲱ"), "ἀποκρίνομαι", "V-"),
    # Prepositional compounds
    CopticIdiomRule(("ⲉⲃⲟⲗ", "ϩⲛ"), "ἐκ", "P-", required_case="G"),
    CopticIdiomRule(("ⲉⲃⲟⲗ", "ϩⲓⲧⲛ"), "διά", "P-", required_case="G"),
    CopticIdiomRule(("ϩⲣⲁⲓ", "ϩⲛ"), "ἐν", "P-", required_case="D"),
    CopticIdiomRule(("ⲉϩⲣⲁⲓ", "ⲉϫⲛ"), "ἐπί", "P-", required_case="A"),
    CopticIdiomRule(("ⲛ", "ϩⲏⲧ"), "ἐν", "P-", required_case="D"),
    CopticIdiomRule(("ⲉ", "ⲧⲙ"), "μή", "D-"),
    # Negated existence & lack
    CopticIdiomRule(("ⲙⲛ", "ⲗⲁⲁⲩ"), "οὐδείς", "A-"),
    CopticIdiomRule(("ⲛ", "ⲗⲁⲁⲩ"), "οὐδείς", "A-"),
)

# Canonical Koine preposition -> governed cases (ordered by default preference)
PREPOSITION_CASE_FRAMES: dict[str, str] = {
    "ἐν": "D",
    "σύν": "D",
    "εἰς": "A",
    "πρός": "A",
    "ἀπό": "G",
    "ἐκ": "G",
    "πρό": "G",
    "χωρίς": "G",
    "ἕνεκα": "G",
    "διά": "G",
    "κατά": "A",
    "ἐπί": "G",
    "μετά": "G",
    "παρά": "D",
    "ὑπό": "G",
    "περί": "G",
    "ὑπέρ": "G",
    "ἐνώπιον": "G",
    "ἔμπροσθεν": "G",
    "ὀπίσω": "G",
    "μέσον": "G",
    "ἕως": "G",
}

# Known non-default verb object cases in Koine (when different from Accusative)
_KNOWN_VERB_OBJECT_CASES: dict[str, str] = {
    "γεύομαι": "G",
    "ἅπτομαι": "G",
    "μιμνῄσκομαι": "G",
    "πληρόω": "G",
    "ἄρχω": "G",
    "κυριεύω": "G",
    "βασιλεύω": "G",
    "δέομαι": "G",
    "πείθομαι": "D",
    "ἀκολουθέω": "D",
    "διακονέω": "D",
    "προσκυνέω": "D",
    "πιστεύω": "D",
    "ὁμοιόω": "D",
    "ἔοικα": "D",
    "χράομαι": "D",
    "λέγω": "A",
    "εὑρίσκω": "A",
    "γινώσκω": "A",
    "ὁράω": "A",
    "ἔχω": "A",
    "ποιέω": "A",
}


class IdiomAndValencyTable:
    """Lookup for Coptic multiword idioms and empirical Greek case valency."""

    def __init__(
        self,
        *,
        idioms: tuple[CopticIdiomRule, ...] = _CURATED_IDIOMS,
        verb_cases: dict[str, str] | None = None,
        prep_cases: dict[str, str] | None = None,
    ) -> None:
        self._idioms = idioms
        self._by_first: dict[str, list[CopticIdiomRule]] = defaultdict(list)
        for rule in idioms:
            if rule.pattern_lemmas:
                self._by_first[rule.pattern_lemmas[0]].append(rule)
        for lst in self._by_first.values():
            lst.sort(key=lambda r: -len(r.pattern_lemmas))
        self.verb_cases: dict[str, str] = dict(_KNOWN_VERB_OBJECT_CASES)
        if verb_cases:
            # Empirical non-default preferences override only when strongly attested
            for k, v in verb_cases.items():
                if k not in _KNOWN_VERB_OBJECT_CASES:
                    self.verb_cases[k] = v
        self.prep_cases: dict[str, str] = dict(PREPOSITION_CASE_FRAMES)
        if prep_cases:
            for k, v in prep_cases.items():
                if k not in PREPOSITION_CASE_FRAMES:
                    self.prep_cases[k] = v

    @classmethod
    def build_from_morphgnt(
        cls,
        units: Iterable[GreekUnit],
        *,
        exclude_unit_ids: set[str] | None = None,
    ) -> IdiomAndValencyTable:
        """Extract verb & preposition object case preferences from MorphGNT train."""
        excluded = set(exclude_unit_ids or ())
        prep_cnt: dict[str, Counter[str]] = defaultdict(Counter)
        verb_cnt: dict[str, Counter[str]] = defaultdict(Counter)

        for unit in units:
            if unit.unit_id in excluded:
                continue
            toks = unit.tokens
            for i, tok in enumerate(toks):
                if tok.morph.pos == "P-":
                    for j in range(i + 1, min(i + 4, len(toks))):
                        nxt = toks[j]
                        if nxt.morph.pos in {
                            "N-",
                            "RP",
                            "RD",
                            "RI",
                            "A-",
                        } and nxt.morph.case in {"G", "D", "A"}:
                            prep_cnt[tok.lemma][nxt.morph.case] += 1
                            break
                elif tok.morph.pos == "V-":
                    for j in range(i + 1, min(i + 4, len(toks))):
                        nxt = toks[j]
                        if nxt.morph.pos in {"P-", "C-", "V-"}:
                            break
                        if nxt.morph.pos in {
                            "N-",
                            "RP",
                            "RD",
                            "A-",
                        } and nxt.morph.case in {"G", "D", "A"}:
                            verb_cnt[tok.lemma][nxt.morph.case] += 1
                            break

        prep_cases: dict[str, str] = {}
        for lem, cnt in prep_cnt.items():
            prep_cases[lem] = cnt.most_common(1)[0][0]
        verb_cases: dict[str, str] = {}
        for lem, cnt in verb_cnt.items():
            top_case, top_n = cnt.most_common(1)[0]
            if top_n >= 4 and top_case in {"G", "D", "A"}:
                verb_cases[lem] = top_case
        return cls(verb_cases=verb_cases, prep_cases=prep_cases)

    def match_at(
        self, morphemes: list[CopticMorpheme], idx: int
    ) -> tuple[CopticIdiomRule, int] | None:
        """Match longest Coptic MWE/idiom starting at ``morphemes[idx]``."""
        if idx >= len(morphemes):
            return None

        def _lem(m: CopticMorpheme) -> str:
            return normalize_coptic(m.lemma) or m.lemma

        cand_rules = self._by_first.get(_lem(morphemes[idx]))
        if not cand_rules:
            return None
        for rule in cand_rules:
            k = len(rule.pattern_lemmas)
            if idx + k <= len(morphemes) and all(
                _lem(morphemes[idx + j]) == rule.pattern_lemmas[j] for j in range(k)
            ):
                return rule, k
        # Also allow 1 intermediate PPERS/NEG/PPERO/ART token for split idioms
        # e.g. ϩⲉ (V) + ⲉ (PREP), ϭⲱⲗⲡ + pronoun + ⲉⲃⲟⲗ
        for rule in cand_rules:
            if len(rule.pattern_lemmas) != 2:
                continue
            target2 = rule.pattern_lemmas[1]
            if target2 in {"ⲉⲃⲟⲗ", "ⲉϩⲟⲩⲛ", "ⲉϩⲣⲁⲓ", "ⲉ"}:
                for span_end in range(idx + 2, min(idx + 5, len(morphemes))):
                    if morphemes[span_end].pos == "V":
                        break
                    if _lem(morphemes[span_end]) == target2:
                        # Contiguous 2-token rule may skip particles or negations only
                        if all(
                            morphemes[p].pos in {"PPERO", "PPERS", "PTC", "NEG"}
                            for p in range(idx + 1, span_end)
                        ):
                            return rule, span_end - idx + 1
        return None

    def verb_object_case(self, greek_lemma: str) -> str:
        """Preferred Greek object case for verb lemma (default Accusative 'A')."""
        return self.verb_cases.get(greek_lemma, "A")

    def prep_case(self, greek_lemma: str) -> str:
        """Preferred Greek case governed by preposition lemma (default Genitive)."""
        return self.prep_cases.get(greek_lemma, "G")

    def save_json(self, path: Path) -> None:
        """Serialize empirical valency frames compactly (< 1024 KiB)."""
        payload = {
            "verb_cases": dict(sorted(self.verb_cases.items())),
            "prep_cases": dict(sorted(self.prep_cases.items())),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> IdiomAndValencyTable:
        """Load saved verb/prep valency frames cleanly."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            verb_cases=data.get("verb_cases", {}),
            prep_cases=data.get("prep_cases", {}),
        )
