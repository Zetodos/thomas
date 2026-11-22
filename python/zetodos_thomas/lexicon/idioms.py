# SPDX-License-Identifier: Apache-2.0
"""Data-driven Sahidic Coptic multiword idioms and Koine Greek case-valency tables.

All multiword idiom rules and verb/preposition case frames are extracted purely
empirically from clean training bitext (NT TRAIN excluding holdout, dev, and
Synoptic parallels of P.Oxy. logia, plus Sahidic OT x Swete LXX). Zero curated
or Thomas-specific idioms are hardcoded.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.ingest.holdouts import assert_clean_training_unit
from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit, GreekUnit

# Structural Sahidic case/dative prepositions never fused into a lexical MWE head
_NON_FUSED_PREPS = frozenset({"ⲛ", "ⲙⲙⲟ", "ϩⲛ", "ⲙⲛ", "ⲛⲁ"})
# Closed-class Sahidic directional adverbs eligible for split verb-particle MWEs
_DIRECTIONAL_ADVS = frozenset({"ⲉⲃⲟⲗ", "ⲉϩⲟⲩⲛ", "ⲉϩⲣⲁⲓ", "ⲉⲡⲉⲥⲏⲧ"})


@dataclass(frozen=True)
class CopticIdiomRule:
    """Multiword Coptic lemma sequence -> target Koine lemma, POS & valency."""

    pattern_lemmas: tuple[str, ...]
    greek_lemma: str
    greek_pos: str
    required_case: str | None = None
    strip_preposition_after: str | None = None
    notes: str = ""


def _extract_valency_from_greek(
    units: Iterable[GreekUnit],
    *,
    excluded: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Extract preposition and verb direct-object case frames empirically."""
    prep_cnt: dict[str, Counter[str]] = defaultdict(Counter)
    verb_cnt: dict[str, Counter[str]] = defaultdict(Counter)

    for unit in units:
        if unit.unit_id in excluded:
            continue
        assert_clean_training_unit(
            unit.unit_id, source_id=unit.source_id, excluded_ids=excluded
        )
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
                # Restrict verb object case statistics to overt noun arguments so
                # dative indirect-object pronouns (e.g. εἶπεν αὐτῷ) never hijack
                # direct-object valency.
                for j in range(i + 1, min(i + 5, len(toks))):
                    nxt = toks[j]
                    if nxt.morph.pos in {"P-", "C-", "V-", "X-"}:
                        break
                    if nxt.morph.pos == "N-" and nxt.morph.case in {"G", "D", "A"}:
                        verb_cnt[tok.lemma][nxt.morph.case] += 1
                        break

    prep_cases: dict[str, str] = {
        lem: cnt.most_common(1)[0][0] for lem, cnt in prep_cnt.items() if cnt
    }
    verb_cases: dict[str, str] = {}
    for lem, cnt in verb_cnt.items():
        total = sum(cnt.values())
        top_case, top_n = cnt.most_common(1)[0]
        if top_case in {"G", "D"} and top_n >= 2 and (top_n / total) >= 0.70:
            verb_cases[lem] = top_case
        elif top_case == "A" and top_n >= 4 and (top_n / total) >= 0.65:
            verb_cases[lem] = top_case
    return verb_cases, prep_cases


def _coptic_mwe_candidates(
    morphemes: Sequence[CopticMorpheme],
) -> set[tuple[tuple[str, ...], str, bool]]:
    """Extract structural Sahidic MWE patterns from one analyzed Coptic unit.

    Returns tuples of ``(pattern_lemmas, target_greek_pos, absorbs_prep)``.
    """
    n = len(morphemes)
    lemmas = [normalize_coptic(m.lemma) or m.lemma for m in morphemes]
    found: set[tuple[tuple[str, ...], str, bool]] = set()

    for i in range(n):
        m0 = morphemes[i]
        l0 = lemmas[i]
        if not l0 or m0.pos == "PUNCT":
            continue

        # 2-token adjacent templates
        if i + 1 < n:
            m1 = morphemes[i + 1]
            l1 = lemmas[i + 1]
            if l1 and m1.pos != "PUNCT":
                if m0.pos in {"V", "VSTAT"}:
                    if m1.pos == "ADV" and l1 in _DIRECTIONAL_ADVS:
                        found.add(((l0, l1), "V-", False))
                    elif m1.pos == "N":
                        found.add(((l0, l1), "V-", False))
                    elif m1.pos == "PREP" and l1 not in _NON_FUSED_PREPS:
                        found.add(((l0, l1), "V-", True))
                elif m0.pos == "ADV" and m1.pos == "PREP":
                    found.add(((l0, l1), "P-", False))
                elif m0.pos == "PTC" and m1.pos in {"N", "ADV", "PTC"}:
                    found.add(((l0, l1), "I-", False))
                elif m0.pos in {"EXIST", "NEG"} and m1.pos in {"N", "PINT"}:
                    found.add(((l0, l1), "A-", False))

        # 3-token V + PREP("ⲉ") + N (e.g. ⲱϩⲉ ⲉ ⲣⲁⲧ -> ἵστημι)
        if (
            m0.pos in {"V", "VSTAT"}
            and i + 2 < n
            and morphemes[i + 1].pos == "PREP"
            and lemmas[i + 1] == "ⲉ"
            and morphemes[i + 2].pos == "N"
            and lemmas[i + 2]
        ):
            found.add(((l0, lemmas[i + 1], lemmas[i + 2]), "V-", False))

        # Split V ... {ⲉⲃⲟⲗ, ⲉϩⲟⲩⲛ, ⲉϩⲣⲁⲓ, ⲉⲡⲉⲥⲏⲧ} within clause
        if m0.pos in {"V", "VSTAT"}:
            for j in range(i + 2, min(n, i + 6)):
                mj = morphemes[j]
                if mj.pos in {"V", "VBD", "VIMP", "VSTAT", "CONJ", "CREL", "PUNCT"}:
                    break
                if mj.pos == "ADV" and lemmas[j] in _DIRECTIONAL_ADVS:
                    # Do not fuse ⲉⲃⲟⲗ if immediately followed by compound prep ϩⲛ/ϩⲓⲧⲛ
                    if (
                        j + 1 < n
                        and morphemes[j + 1].pos == "PREP"
                        and lemmas[j + 1] in {"ϩⲛ", "ϩⲓⲧⲛ", "ⲉϫⲛ"}
                    ):
                        continue
                    found.add(((l0, lemmas[j]), "V-", False))
                    break

    return found


class IdiomAndValencyTable:
    """Empirical lookup for Coptic multiword idioms and Koine Greek case valency."""

    def __init__(
        self,
        *,
        idioms: Sequence[CopticIdiomRule] = (),
        verb_cases: dict[str, str] | None = None,
        prep_cases: dict[str, str] | None = None,
    ) -> None:
        self._idioms: tuple[CopticIdiomRule, ...] = tuple(idioms)
        self._by_first: dict[str, list[CopticIdiomRule]] = defaultdict(list)
        for rule in self._idioms:
            if rule.pattern_lemmas:
                self._by_first[rule.pattern_lemmas[0]].append(rule)
        for lst in self._by_first.values():
            lst.sort(key=lambda r: (-len(r.pattern_lemmas), r.pattern_lemmas))
        self.verb_cases: dict[str, str] = dict(verb_cases or {})
        self.prep_cases: dict[str, str] = dict(prep_cases or {})

    @classmethod
    def build_from_morphgnt(
        cls,
        units: Iterable[GreekUnit],
        *,
        exclude_unit_ids: set[str] | None = None,
    ) -> IdiomAndValencyTable:
        """Extract verb & preposition object case frames purely from MorphGNT train."""
        excluded = set(exclude_unit_ids or ())
        verb_cases, prep_cases = _extract_valency_from_greek(units, excluded=excluded)
        return cls(idioms=(), verb_cases=verb_cases, prep_cases=prep_cases)

    @classmethod
    def build_from_bitext(
        cls,
        *,
        nt_pairs: Sequence[tuple[CopticUnit, GreekUnit]],
        ot_pairs: Sequence[tuple[CopticUnit, GreekUnit]] = (),
        morphgnt_units: Iterable[GreekUnit] | None = None,
        exclude_unit_ids: set[str] | None = None,
        min_count: float = 4.0,
        min_prob: float = 0.45,
        min_dice: float = 0.15,
    ) -> IdiomAndValencyTable:
        """Learn multiword Coptic->Greek idioms and Greek case valency empirically."""
        excluded = set(exclude_unit_ids or ())
        valency_source: Iterable[GreekUnit] = (
            morphgnt_units
            if morphgnt_units is not None
            else (g_u for _c_u, g_u in nt_pairs)
        )
        verb_cases, prep_cases = _extract_valency_from_greek(
            valency_source, excluded=excluded
        )

        # Accumulate weighted co-occurrence counts strictly over clean training pairs
        c_pat_cnt: dict[tuple[tuple[str, ...], str, bool], float] = defaultdict(float)
        g_lem_cnt: dict[tuple[str, str], float] = defaultdict(float)
        co_cnt: dict[
            tuple[tuple[str, ...], str, bool], dict[tuple[str, str], float]
        ] = defaultdict(lambda: defaultdict(float))
        # Track how often a Greek verb has NO following Greek preposition (for V+PREP)
        no_prep_cnt: dict[tuple[tuple[str, ...], str, bool, str], float] = defaultdict(
            float
        )

        weighted_corpora: list[tuple[Sequence[tuple[CopticUnit, GreekUnit]], float]] = [
            (nt_pairs, 1.0),
            (ot_pairs, 0.5),
        ]
        for pairs, weight in weighted_corpora:
            for c_u, g_u in pairs:
                assert_clean_training_unit(
                    c_u.unit_id, source_id=c_u.source_id, excluded_ids=excluded
                )
                assert_clean_training_unit(
                    g_u.unit_id, source_id=g_u.source_id, excluded_ids=excluded
                )
                toks = g_u.tokens
                g_present: set[tuple[str, str]] = set()
                verb_without_prep: set[str] = set()
                for idx_t, t in enumerate(toks):
                    if t.morph.pos not in {"V-", "N-", "A-", "D-", "P-", "C-", "I-"}:
                        continue
                    if not t.lemma:
                        continue
                    g_present.add((t.lemma, t.morph.pos))
                    if t.morph.pos == "V-":
                        # Verb governs a preposition only if P- precedes any N- object
                        has_adj_prep = False
                        for k in range(idx_t + 1, min(idx_t + 3, len(toks))):
                            kpos = toks[k].morph.pos
                            if kpos == "P-":
                                has_adj_prep = True
                                break
                            if kpos in {"N-", "RA", "V-", "X-"}:
                                break
                        if not has_adj_prep:
                            verb_without_prep.add(t.lemma)

                for gp in g_present:
                    g_lem_cnt[gp] += weight

                pats = _coptic_mwe_candidates(c_u.morphemes)
                for pat in pats:
                    c_pat_cnt[pat] += weight
                    target_pos = pat[1]
                    for gl, gpos in g_present:
                        if gpos != target_pos:
                            continue
                        co_cnt[pat][(gl, gpos)] += weight
                        if pat[2] and gl in verb_without_prep:
                            no_prep_cnt[(pat[0], pat[1], pat[2], gl)] += weight

        scored_rules: list[tuple[float, CopticIdiomRule]] = []
        seen_patterns: set[tuple[str, ...]] = set()
        # Rank candidates deterministically by association strength
        candidates: list[tuple[float, tuple[str, ...], str, str, bool, float]] = []
        for pat, c_n in c_pat_cnt.items():
            if c_n < min_count:
                continue
            pat_lems, target_pos, absorbs_prep = pat
            best: tuple[float, str, str, float] | None = None
            for (gl, gpos), co in co_cnt[pat].items():
                if co < min_count:
                    continue
                if absorbs_prep:
                    np_rate = (
                        no_prep_cnt.get((pat_lems, target_pos, True, gl), 0.0) / co
                    )
                    if np_rate < 0.75:
                        continue
                p_gc = co / c_n
                g_n = g_lem_cnt.get((gl, gpos), co)
                p_cg = co / max(g_n, 1e-9)
                dice = (2.0 * co) / max(c_n + g_n, 1e-9)
                if p_gc >= min_prob and dice >= min_dice:
                    score = p_gc * (p_cg**0.35)
                    if best is None or score > best[0]:
                        best = (score, gl, gpos, co)
            if best is not None:
                candidates.append(
                    (best[0], pat_lems, best[1], best[2], absorbs_prep, best[3])
                )

        candidates.sort(key=lambda item: (-item[0], -len(item[1]), item[1], item[2]))
        for score, pat_lems, gl, gpos, absorbs_prep, _co in candidates:
            if pat_lems in seen_patterns:
                continue
            seen_patterns.add(pat_lems)
            req_case = (
                verb_cases.get(gl)
                if gpos == "V-"
                else prep_cases.get(gl)
                if gpos == "P-"
                else None
            )
            strip_p = pat_lems[1] if (absorbs_prep and len(pat_lems) == 2) else None
            rule = CopticIdiomRule(
                pattern_lemmas=pat_lems,
                greek_lemma=gl,
                greek_pos=gpos,
                required_case=req_case,
                strip_preposition_after=strip_p,
            )
            scored_rules.append((score, rule))

        return cls(
            idioms=[r for _, r in scored_rules],
            verb_cases=verb_cases,
            prep_cases=prep_cases,
        )

    def match_at(
        self, morphemes: list[CopticMorpheme], idx: int
    ) -> tuple[CopticIdiomRule, int] | None:
        """Match longest empirical Coptic MWE/idiom starting at ``morphemes[idx]``."""
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
        # Allow intermediate clitics/negations for split directional-particle idioms
        for rule in cand_rules:
            if len(rule.pattern_lemmas) != 2:
                continue
            target2 = rule.pattern_lemmas[1]
            if target2 in _DIRECTIONAL_ADVS or target2 == "ⲉ":
                for span_end in range(idx + 2, min(idx + 5, len(morphemes))):
                    if morphemes[span_end].pos == "V":
                        break
                    if _lem(morphemes[span_end]) == target2:
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
        """Serialize empirical MWE idioms and valency frames compactly (< 1024 KiB)."""
        payload = {
            "idioms": [
                [
                    list(r.pattern_lemmas),
                    r.greek_lemma,
                    r.greek_pos,
                    r.required_case,
                    r.strip_preposition_after,
                ]
                for r in sorted(
                    self._idioms,
                    key=lambda item: (item.pattern_lemmas, item.greek_lemma),
                )
            ],
            "verb_cases": dict(sorted(self.verb_cases.items())),
            "prep_cases": dict(sorted(self.prep_cases.items())),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> IdiomAndValencyTable:
        """Load saved empirical MWE idioms and verb/prep valency frames."""
        data = json.loads(path.read_text(encoding="utf-8"))
        idioms = [
            CopticIdiomRule(
                pattern_lemmas=tuple(str(x) for x in row[0]),
                greek_lemma=str(row[1]),
                greek_pos=str(row[2]),
                required_case=str(row[3]) if row[3] is not None else None,
                strip_preposition_after=str(row[4]) if row[4] is not None else None,
            )
            for row in data.get("idioms", [])
        ]
        return cls(
            idioms=idioms,
            verb_cases=data.get("verb_cases", {}),
            prep_cases=data.get("prep_cases", {}),
        )
