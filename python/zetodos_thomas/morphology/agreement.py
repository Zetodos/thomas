# SPDX-License-Identifier: Apache-2.0
"""Zero-leakage Koine Greek morphological agreement critic & safe 1-edit repairer.

Built strictly from zero-holdout MorphGNT-train + Swete analyzer/paradigm tables
already carried by ``NeuralKoineDecoder`` (zero Thomas or holdout material).
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.schemas.linguistic import GreekMorphFeatures

# Complete Koine Greek definite article table: lowercase -> all (case, num, gen)
_ARTICLE_PARSES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "ὁ": (("N", "S", "M"),),
    "ἡ": (("N", "S", "F"),),
    "τό": (("N", "S", "N"), ("A", "S", "N")),
    "τοῦ": (("G", "S", "M"), ("G", "S", "N")),
    "τῆς": (("G", "S", "F"),),
    "τῷ": (("D", "S", "M"), ("D", "S", "N")),
    "τῇ": (("D", "S", "F"),),
    "τόν": (("A", "S", "M"),),
    "τήν": (("A", "S", "F"),),
    "οἱ": (("N", "P", "M"),),
    "αἱ": (("N", "P", "F"),),
    "τά": (("N", "P", "N"), ("A", "P", "N")),
    "τῶν": (("G", "P", "M"), ("G", "P", "F"), ("G", "P", "N")),
    "τοῖς": (("D", "P", "M"), ("D", "P", "N")),
    "ταῖς": (("D", "P", "F"),),
    "τούς": (("A", "P", "M"),),
    "τάς": (("A", "P", "F"),),
}

_ARTICLE_BY_CNG: dict[tuple[str, str, str], str] = {
    cng: form for form, cngs in _ARTICLE_PARSES.items() for cng in cngs
}

# Strictly single-case prepositions in Koine Greek (MorphGNT-invariant)
_STRICT_PREP_CASE: dict[str, str] = {
    "ἐκ": "G",
    "ἐξ": "G",
    "ἀπό": "G",
    "ἀπ’": "G",
    "ἀφ’": "G",
    "πρό": "G",
    "χωρίς": "G",
    "ἄνευ": "G",
    "ἀντί": "G",
    "εἰς": "A",
    "ἀνά": "A",
    "ἐν": "D",
    "σύν": "D",
}

# Indeclinable Greek numerals & biblical proper names never flagged for case/number
_INDECLINABLE: frozenset[str] = frozenset(
    {
        "δύο",
        "πέντε",
        "ἕξ",
        "ἑπτά",
        "ὀκτώ",
        "ἐννέα",
        "δέκα",
        "δώδεκα",
        "ἑκατόν",
        "Ἀδάμ",
        "Ἀβραάμ",
        "Ἰσραήλ",
        "Ἰερουσαλήμ",
        "Δαυίδ",
        "Μωϋσῆς",
        "Ἰσαάκ",
        "Ἰακώβ",
        "Ἰωσήφ",
        "Σαλώμη",
        "Θωμᾶς",
        "Μαριάμ",
        "Καπερναούμ",
        "Βηθλεέμ",
        "Ναζαρέθ",
        "Ναζαρέτ",
    }
)

# Common-gender (M or F) Koine nouns attested with both masculine and feminine articles
_COMMON_GENDER_LEMMAS: frozenset[str] = frozenset(
    {
        "βοῦς",
        "ἄνθρωπος",
        "παῖς",
        "θεός",
        "κύων",
        "λίθος",
        "στάμνος",
        "ὄνος",
        "λιμός",
        "βάτος",
        "ἔρημος",
        "παρθένος",
    }
)

# Clause boundary tokens that reset local subject-predicate tracking
_CLAUSE_BREAKS: frozenset[str] = frozenset(
    {
        "καί",
        "ἀλλά",
        "δέ",
        "ὅτι",
        "ἵνα",
        "ἐάν",
        "ὅταν",
        "εἰ",
        "ἤ",
        "οὐδέ",
        "οὔτε",
        "ὥστε",
        "ὅτε",
        "ὅπου",
        "ὅς",
        "ἥ",
        "ὅ",
        "οἵ",
        "αἵ",
        "ἅ",
    }
)


def _strip_bare(word: str) -> str:
    nfd = unicodedata.normalize("NFD", word)
    return (
        "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        .lower()
        .replace("ς", "σ")
    )


def _norm_accent(word: str) -> str:
    nfd = unicodedata.normalize("NFD", word).replace("\u0300", "\u0301")
    return unicodedata.normalize("NFC", nfd)


@dataclass(frozen=True)
class AgreementViolation:
    """One concrete morphological agreement violation in a Greek token sequence."""

    kind: str  # "art_head" | "prep_case" | "homoios_case" | "pred_gender"
    index: int  # primary token index (e.g. article or prep or predicate)
    target_index: int  # dependent token index (e.g. head noun or predicate)
    detail: str


class KoineAgreementCritic:
    """Syncretism-aware Koine Greek morph agreement scorer & safe repairer."""

    def __init__(
        self,
        analyzer_inflector: GreekInflector,
        paradigm_inflector: GreekInflector,
    ) -> None:
        self._analyzer = analyzer_inflector
        self._paradigm = paradigm_inflector
        multi: dict[str, list[tuple[str, GreekMorphFeatures]]] = defaultdict(list)
        seen: dict[str, set[tuple[str, str]]] = defaultdict(set)

        def _add(key: str, lem: str, mf: GreekMorphFeatures) -> None:
            sig = mf.signature()
            if (lem, sig) not in seen[key]:
                seen[key].add((lem, sig))
                multi[key].append((lem, mf))

        for (lem, sig), surf in paradigm_inflector._paradigms.items():
            mf = GreekInflector.parse_signature(sig)
            s_norm = _norm_accent(surf)
            _add(s_norm, lem, mf)
            _add(s_norm.lower(), lem, mf)

        for surf, plist in analyzer_inflector._surface_analyses.items():
            s_norm = _norm_accent(surf)
            for lem, sig, _ in plist:
                mf = GreekInflector.parse_signature(sig)
                _add(s_norm, lem, mf)
                _add(s_norm.lower(), lem, mf)

        self._multi: dict[str, list[tuple[str, GreekMorphFeatures]]] = dict(multi)
        self._parse_cache: dict[str, tuple[tuple[str, GreekMorphFeatures], ...]] = {}

    def parses(self, word: str) -> tuple[tuple[str, GreekMorphFeatures], ...]:
        """Return all syncretic ``(lemma, GreekMorphFeatures)`` parses for ``word``."""
        hit = self._parse_cache.get(word)
        if hit is not None:
            return hit
        w_norm = _norm_accent(word)
        low = w_norm.lower()
        if low in _ARTICLE_PARSES:
            out = tuple(
                ("ὁ", GreekMorphFeatures(pos="RA", case=c, number=n, gender=g))
                for c, n, g in _ARTICLE_PARSES[low]
            )
            self._parse_cache[word] = out
            return out

        base = list(self._multi.get(w_norm) or self._multi.get(low) or [])
        if not base:
            for lem, mf, _ in self._analyzer.analyze_surface(w_norm):
                base.append((lem, mf))

        expanded: list[tuple[str, GreekMorphFeatures]] = list(base)
        seen = {(lem, mf.signature()) for lem, mf in base}

        def _push(lem: str, mf: GreekMorphFeatures) -> None:
            k = (lem, mf.signature())
            if k not in seen:
                seen.add(k)
                expanded.append((lem, mf))

        for lem, mf in base:
            # Neuter Nominative <-> Accusative syncretism across all declensions
            if mf.gender == "N" and mf.case in {"N", "A"}:
                other = "A" if mf.case == "N" else "N"
                _push(lem, mf.model_copy(update={"case": other}))
            # Common-gender nouns (e.g. βοῦς, στάμνος, παρθένος) allow M and F
            if lem in _COMMON_GENDER_LEMMAS and mf.gender in {"M", "F"}:
                _push(
                    lem,
                    mf.model_copy(update={"gender": "F" if mf.gender == "M" else "M"}),
                )
            # 1st-declension masculine nouns in -ης (μαθητής, προφήτης, ψευδοπροφήτης)
            # take -αις in Dative Plural (`τοῖς ψευδοπροφήταις`).
            if (
                mf.pos == "N-"
                and low.endswith(("αις", "αῖς"))
                and (lem.endswith(("ης", "ής")) or low.endswith(("ταις", "ταῖς")))
            ):
                _push(
                    lem,
                    mf.model_copy(update={"case": "D", "number": "P", "gender": "M"}),
                )
            # Unambiguous Koine 1st/2nd-declension iota-subscript dative singular
            if mf.pos in {"N-", "A-"} and low.endswith(("ᾳ", "ῃ", "ᾴ", "ῄ", "ᾷ", "ῇ")):
                _push(
                    lem,
                    mf.model_copy(update={"case": "D", "number": "S", "gender": "F"}),
                )
            elif mf.pos in {"N-", "A-"} and low.endswith(("ῳ", "ῴ", "ῷ")):
                _push(
                    lem,
                    mf.model_copy(update={"case": "D", "number": "S", "gender": "M"}),
                )
                _push(
                    lem,
                    mf.model_copy(update={"case": "D", "number": "S", "gender": "N"}),
                )
            # Feminine 1st-declension genitive singular in -ᾶς / -ῆς (e.g. πλευρᾶς)
            if (
                mf.pos in {"N-", "A-"}
                and mf.gender == "F"
                and low.endswith(("ᾶς", "ῆς", "άς", "ής"))
                and lem.endswith(("α", "ά", "η", "ή"))
            ):
                _push(
                    lem,
                    mf.model_copy(update={"case": "G", "number": "S", "gender": "F"}),
                )
            # 2-termination / masculine-neuter adjective -ον and -μενον participles
            # (ASN = NSN = ASM)
            if (
                (
                    (mf.pos == "A-" and lem.endswith(("ος", "ός")))
                    or (
                        mf.pos == "V-"
                        and mf.mood == "P"
                        and low.endswith(("μενον", "μένον"))
                    )
                )
                and low.endswith(("ον", "όν"))
                and mf.number == "S"
            ):
                for c2, g2 in (("A", "M"), ("N", "N"), ("A", "N")):
                    _push(
                        lem,
                        mf.model_copy(update={"case": c2, "number": "S", "gender": g2}),
                    )
            # Genitive plural -ων adjectives/participles take all 3 genders
            if (
                (mf.pos == "A-" or (mf.pos == "V-" and mf.mood == "P"))
                and mf.case == "G"
                and mf.number == "P"
                and low.endswith(("ων", "ῶν"))
            ):
                for g2 in ("M", "F", "N"):
                    _push(lem, mf.model_copy(update={"gender": g2}))
            # Dative/Genitive singular M/N adjectives & participles share -ου/-ῳ/-οις
            if (
                (mf.pos == "A-" or (mf.pos == "V-" and mf.mood == "P"))
                and mf.gender in {"M", "N"}
                and mf.case in {"G", "D"}
            ):
                _push(
                    lem,
                    mf.model_copy(update={"gender": "N" if mf.gender == "M" else "M"}),
                )

        res = tuple(expanded)
        self._parse_cache[word] = res
        return res

    @staticmethod
    def _cng_match(a: GreekMorphFeatures, b: GreekMorphFeatures) -> bool:
        if a.case != b.case or a.number != b.number:
            return False
        if a.gender and b.gender and a.gender != b.gender:
            return False
        return True

    def detect_violations(self, words: Sequence[str]) -> list[AgreementViolation]:
        """Detect high-precision morphological agreement violations in ``words``."""
        n = len(words)
        if n < 2:
            return []
        ps = [self.parses(w) for w in words]
        violations: list[AgreementViolation] = []

        for i in range(n - 1):
            w0, w1 = words[i], words[i + 1]
            p0, p1 = ps[i], ps[i + 1]
            low0 = _norm_accent(w0).lower()
            low1 = _norm_accent(w1).lower()

            # 1. Definite Article + Head Nominal/Adj/Participle
            if low0 in _ARTICLE_PARSES and w1 not in _INDECLINABLE:
                hd1 = [
                    (lem, m)
                    for lem, m in p1
                    if m.pos in {"N-", "A-"} or (m.pos == "V-" and m.mood == "P")
                ]
                non_hd1 = [
                    (lem, m)
                    for lem, m in p1
                    if m.pos not in {"N-", "A-"}
                    and not (m.pos == "V-" and m.mood == "P")
                ]
                if hd1 and not non_hd1:
                    ok = any(self._cng_match(r, h) for _, r in p0 for _, h in hd1)
                    # Allow elliptic `RA + Genitive NPROP` (`τῶν Χλόης`)
                    if (
                        not ok
                        and w1[0].isupper()
                        and any(h.case == "G" for _, h in hd1)
                    ):
                        ok = True
                    # Allow sandwiched attributive modifier/object when words[i+2]
                    # (e.g., head noun or ptcp in `ὁ φαῦλα πράσσων`) agrees with RA.
                    if not ok and i + 2 < n:
                        hd2 = [
                            m
                            for _, m in ps[i + 2]
                            if m.pos in {"N-", "A-"}
                            or (m.pos == "V-" and m.mood == "P")
                        ]
                        if any(self._cng_match(r, h2) for _, r in p0 for h2 in hd2):
                            ok = True
                    if not ok:
                        violations.append(
                            AgreementViolation(
                                kind="art_head",
                                index=i,
                                target_index=i + 1,
                                detail=f"{w0} {w1}",
                            )
                        )

            # 2. Strict Single-Case Preposition + Following Nominal / Article
            pr_lemmas = [
                lem for lem, m in p0 if m.pos == "P-" and lem in _STRICT_PREP_CASE
            ]
            if pr_lemmas and w1 not in _INDECLINABLE:
                req = _STRICT_PREP_CASE[pr_lemmas[0]]
                # Skip articular infinitive (`εἰς τό φαγεῖν`, `ἐν τῷ εἶναι`)
                is_art_inf = (
                    low1 in _ARTICLE_PARSES
                    and i + 2 < n
                    and any(m.pos == "V-" and m.mood == "N" for _, m in ps[i + 2])
                )
                if not is_art_inf:
                    # Only check when w1 is unambiguously case-bearing (not adv/conj).
                    case_parses = [m for _, m in p1 if m.case is not None]
                    non_case_parses = [
                        m for _, m in p1 if m.case is None and m.pos != "UNK"
                    ]
                    if case_parses and not non_case_parses:
                        if not any(m.case == req for m in case_parses):
                            violations.append(
                                AgreementViolation(
                                    kind="prep_case",
                                    index=i,
                                    target_index=i + 1,
                                    detail=f"{w0}({req}) {w1}",
                                )
                            )

            # 3. `ὅμοιος` + [ἐστίν]? + bare Nominative noun (governs Dative in Koine)
            if (
                any(lem == "ὅμοιος" and m.pos == "A-" for lem, m in p0)
                and w1 not in _INDECLINABLE
            ):
                n1 = [
                    m
                    for lem, m in p1
                    if m.pos == "N-"
                    and lem not in {"θεός", "κύριος", "Χριστός", "Ἰησοῦς"}
                ]
                non_n1 = [m for _, m in p1 if m.pos != "N-"]
                if n1 and not non_n1 and all(m.case == "N" for m in n1):
                    violations.append(
                        AgreementViolation(
                            kind="homoios_case",
                            index=i,
                            target_index=i + 1,
                            detail=f"{w0} {w1}",
                        )
                    )
                # Also catch `ὅμοιος + ἐστίν/ἐστί + bare Nom N-` (`ὁμοία ἐστίν γυνή`)
                elif (
                    i + 2 < n
                    and low1 in {"ἐστίν", "ἐστί"}
                    and words[i + 2] not in _INDECLINABLE
                ):
                    n2_c = [
                        m
                        for lem, m in ps[i + 2]
                        if m.pos == "N-"
                        and lem not in {"θεός", "κύριος", "Χριστός", "Ἰησοῦς"}
                    ]
                    non_n2_c = [m for _, m in ps[i + 2] if m.pos != "N-"]
                    if n2_c and not non_n2_c and all(m.case == "N" for m in n2_c):
                        violations.append(
                            AgreementViolation(
                                kind="homoios_case",
                                index=i,
                                target_index=i + 2,
                                detail=f"{w0} {w1} {words[i + 2]}",
                            )
                        )
                # Catch `ὅμοιος + Dat.Sg N- + Nom.Pl or Acc.Sg appositive N-`
                # (e.g., `ὁμοία ἀνθρώπῳ ἔμποροι`, `ὁμοία ἀνθρώπῳ ποιμένα`).
                elif (
                    i + 2 < n
                    and n1
                    and any(m.case == "D" and m.number == "S" for m in n1)
                    and words[i + 2] not in _INDECLINABLE
                ):
                    n2 = [m for _, m in ps[i + 2] if m.pos == "N-"]
                    non_n2 = [m for _, m in ps[i + 2] if m.pos != "N-"]
                    if (
                        n2
                        and not non_n2
                        and (
                            all(m.case == "N" and m.number == "P" for m in n2)
                            or all(m.case == "A" and m.number == "S" for m in n2)
                        )
                    ):
                        violations.append(
                            AgreementViolation(
                                kind="app_dative",
                                index=i + 1,
                                target_index=i + 2,
                                detail=f"{w1} {words[i + 2]}",
                            )
                        )

            # 3c. Predicate `μακάριος`/`ταλαίπωρος` + copula + definite NP subject
            # (e.g. `μακαρία ἐστίν ὁ ἄνθρωπος`, `ταλαίπωρος ἐστίν τό σῶμα`).
            if (
                i + 3 < n
                and any(
                    lem in {"μακάριος", "ταλαίπωρος"} and m.pos == "A-" for lem, m in p0
                )
                and low1 in {"ἐστίν", "ἐστί", "εἰσίν", "εἰσί"}
            ):
                low2 = _norm_accent(words[i + 2]).lower()
                if low2 in {"ὁ", "ἡ", "τό", "οἱ", "αἱ", "τά"}:
                    ra_cng = _ARTICLE_PARSES[low2][0]
                    n3 = [
                        m
                        for _, m in ps[i + 3]
                        if m.pos == "N-"
                        and m.case == "N"
                        and m.number == ra_cng[1]
                        and m.gender == ra_cng[2]
                    ]
                    if n3 and not any(
                        m.pos == "A-"
                        and m.case == "N"
                        and m.number == ra_cng[1]
                        and m.gender == ra_cng[2]
                        for _, m in p0
                    ):
                        violations.append(
                            AgreementViolation(
                                kind="pred_gender",
                                index=i + 3,
                                target_index=i,
                                detail=f"{w0} .. {words[i + 2]} {words[i + 3]}",
                            )
                        )

            # 3d. Direct numeral or post-nominal modifier agreement:
            # - `τρεῖς`/`τρία` + following noun (`τρεῖς ῥήματα` -> `τρία ῥήματα`)
            # - noun + following `εἷς`/`μία`/`ἕν` (`μαργαρίτην ἕν` -> `ἕνα`)
            # - noun + immediately following unambiguous post-nominal adj/ptcp
            #   (`περιτομή ἀληθινῆς`, `στάμνος μεστόν`, `πῦρ ἐξελθών`,
            #    `θησαυρόν ἀποκεκρυμμένη`).
            if low0 in {"τρεῖς", "τρία"} and w1 not in _INDECLINABLE:
                n1_t = [m for _, m in p1 if m.pos == "N-"]
                non_n1_t = [m for _, m in p1 if m.pos != "N-"]
                if (
                    n1_t
                    and not non_n1_t
                    and not any(self._cng_match(r, h) for _, r in p0 for h in n1_t)
                ):
                    violations.append(
                        AgreementViolation(
                            kind="noun_mod",
                            index=i + 1,
                            target_index=i,
                            detail=f"{w0} {w1}",
                        )
                    )
            if (
                w0 not in _INDECLINABLE
                and w1 not in _INDECLINABLE
                and not w0[0].isupper()
            ):
                n0_m = [
                    (lem, m)
                    for lem, m in p0
                    if m.pos == "N-"
                    and m.case in {"N", "A"}
                    and m.number == "S"
                    and lem not in {"θεός", "κύριος", "Χριστός", "Ἰησοῦς"}
                ]
                non_n0_m = [m for _, m in p0 if m.pos != "N-"]
                if n0_m and not non_n0_m:
                    # Noun + unambiguous post-nominal attributive adj or participle
                    mod1 = [
                        (lem, m)
                        for lem, m in p1
                        if (
                            (
                                m.pos == "A-"
                                and lem
                                not in {
                                    "εἷς",
                                    "ὅμοιος",
                                    "μακάριος",
                                    "ταλαίπωρος",
                                    "πολύς",
                                    "πᾶς",
                                    "οὐδείς",
                                    "μηδείς",
                                    "ἄξιος",
                                    "πρῶτος",
                                    "ἔσχατος",
                                    "μόνος",
                                    "ἄλλος",
                                    "ἕτερος",
                                    "μέγας",
                                    "μικρός",
                                    "θαυμαστός",
                                    "κενός",
                                    "πλήρης",
                                    "φανερός",
                                }
                            )
                            or (
                                m.pos == "V-"
                                and m.mood == "P"
                                and m.case in {"N", "A"}
                                and m.number == "S"
                            )
                        )
                    ]
                    non_mod1 = [
                        m
                        for _, m in p1
                        if m.pos != "A-" and not (m.pos == "V-" and m.mood == "P")
                    ]
                    if (
                        mod1
                        and not non_mod1
                        and not any(
                            self._cng_match(h, r) for _, h in n0_m for _, r in mod1
                        )
                        and not (
                            i + 2 < n
                            and _norm_accent(words[i + 2]).lower()
                            in {"ἐστίν", "ἐστί", "ἦν", "ἔσται"}
                        )
                        and not (
                            i + 2 < n
                            and any(
                                h2.pos == "N-"
                                and any(self._cng_match(r, h2) for _, r in mod1)
                                for _, h2 in ps[i + 2]
                            )
                        )
                    ):
                        # Check same-case/number gender clash (e.g.
                        # `στάμνος μεστόν`, `πῦρ ἐξελθών`), or Acc noun +
                        # passive ptcp (`θησαυρόν ἀποκεκρυμμένη` — never
                        # active/middle `ἐξουσίαν ἔχων` or `ὡς N ptcp`),
                        # or Nom.Sg noun + Gen.Sg adj (`περιτομή ἀληθινῆς`).
                        prev_is_hos = i > 0 and _norm_accent(words[i - 1]).lower() in {
                            "ὡς",
                            "ὡσεί",
                        }
                        has_gender_clash = not prev_is_hos and any(
                            h.number == r.number
                            and h.gender != r.gender
                            and (
                                h.case == r.case
                                or (
                                    h.case == "A"
                                    and r.case == "N"
                                    and r.pos == "V-"
                                    and r.voice == "P"
                                )
                            )
                            for _, h in n0_m
                            for _, r in mod1
                        )
                        has_nom_gen_adj = all(
                            r.pos == "A-" and r.case == "G" and r.number == "S"
                            for _, r in mod1
                        ) and any(
                            h.case == "N" and h.gender == r.gender
                            for _, h in n0_m
                            for _, r in mod1
                        )
                        if has_gender_clash or has_nom_gen_adj:
                            violations.append(
                                AgreementViolation(
                                    kind="noun_mod",
                                    index=i,
                                    target_index=i + 1,
                                    detail=f"{w0} {w1}",
                                )
                            )

            # 3b. Direct anarthrous Nom.Sg Adj (`κακός`, `ἀγαθός`, ...) + Gen.Sg Noun
            # of same gender (e.g. `κακός ἀνθρώπου` modifying `ἄνθρωπος`), excluding
            # quantifiers/partitives like `πολλοί`, `πᾶς`, `εἷς`, `ἄξιος`, `μεστός`.
            if (
                w0 not in _INDECLINABLE
                and w1 not in _INDECLINABLE
                and not (
                    i > 0 and _norm_accent(words[i - 1]).lower() in _ARTICLE_PARSES
                )
            ):
                a0 = [
                    (lem, m)
                    for lem, m in p0
                    if m.pos == "A-"
                    and m.case == "N"
                    and m.number == "S"
                    and m.gender in {"M", "F"}
                    and lem
                    not in {
                        "ὅμοιος",
                        "πᾶς",
                        "πολύς",
                        "εἷς",
                        "ἕκαστος",
                        "ἄξιος",
                        "μεστός",
                        "πλήρης",
                        "ἔρημος",
                        "κενός",
                        "μέγας",
                        "μικρός",
                        "πρῶτος",
                        "ἔσχατος",
                        "μόνος",
                        "ἄλλος",
                        "ἕτερος",
                    }
                ]
                non_a0 = [m for _, m in p0 if m.pos != "A-"]
                n1 = [
                    m
                    for lem, m in p1
                    if m.pos == "N-"
                    and lem not in {"θεός", "κύριος", "Χριστός", "Ἰησοῦς"}
                ]
                non_n1 = [m for _, m in p1 if m.pos != "N-"]
                if (
                    a0
                    and not non_a0
                    and n1
                    and not non_n1
                    and all(m.case == "G" and m.number == "S" for m in n1)
                    and any(m.gender == a0[0][1].gender for m in n1)
                    and not (
                        i + 2 < n
                        and any(
                            m.pos == "N-" and m.case == "N" and m.number == "S"
                            for _, m in ps[i + 2]
                        )
                    )
                ):
                    violations.append(
                        AgreementViolation(
                            kind="adj_noun",
                            index=i,
                            target_index=i + 1,
                            detail=f"{w0} {w1}",
                        )
                    )

        # 4. Clause-local Nominative subject (`ἡ βασιλεία`, `ἥτις`, `οἱ ἄνδρες`) vs
        # predicate adj/ptcp gender clash OR M/F plural subject vs 3sg verb clash.
        subj: tuple[int, str, str] | None = None  # (idx, number, gender)
        for i in range(n):
            w = words[i]
            low = _norm_accent(w).lower()
            p = ps[i]
            if low in _CLAUSE_BREAKS or low in {"οὐαί", "ἰδού", "ἀμήν"}:
                subj = None
                # Feminine/Masculine relative pronouns (`ἥτις`, `ὅστις`) start a subject
                if low == "ἥτις":
                    subj = (i, "S", "F")
                elif low == "ὅστις":
                    subj = (i, "S", "M")
                continue
            if low == "ἥτις":
                subj = (i, "S", "F")
                continue
            if low == "ὅστις":
                subj = (i, "S", "M")
                continue
            # Speech-introducer verbs (`εἶπεν`, `λέγει`, etc.) end the framing clause
            if any(
                lem in {"λέγω", "εἶπον", "λαλέω", "ἀποκρίνομαι"} and m.pos == "V-"
                for lem, m in p
            ):
                subj = None
                continue
            if low in {"πολλοί", "πολλαί"}:
                subj = (i, "P", "M" if low == "πολλοί" else "F")
                continue
            if i + 1 < n and low in {"ὁ", "ἡ", "οἱ", "αἱ"}:
                # Allow postpositive `δέ`/`γάρ`/`μέν` between article and noun
                # (e.g. `οἱ δέ μαθηταί σου εἶπεν`).
                n_pos = i + 1
                if i + 2 < n and _norm_accent(words[i + 1]).lower() in {
                    "δέ",
                    "γάρ",
                    "μέν",
                }:
                    n_pos = i + 2
                if words[n_pos] not in {
                    "Ἰησοῦς",
                    "Χριστός",
                    "Κύριος",
                    "Πέτρος",
                    "Θωμᾶς",
                } and not any(m.pos == "V-" and m.mood == "P" for _, m in ps[n_pos]):
                    ra_cng = _ARTICLE_PARSES[low][0]
                    n_hits = [
                        m
                        for _, m in ps[n_pos]
                        if m.pos == "N-"
                        and m.case == "N"
                        and m.number == ra_cng[1]
                        and m.gender == ra_cng[2]
                    ]
                    if n_hits:
                        subj = (n_pos, ra_cng[1], ra_cng[2])
                continue
            if subj is not None and i > subj[0]:
                # M/F Plural Nom subject followed closely by [poss/neg]? + 3sg verb
                # (Never flag neuter plural subjects — neuter plural + 3sg is valid!)
                between_ok = i - subj[0] <= 2 or (
                    i - subj[0] == 3
                    and all(
                        _norm_accent(words[k]).lower()
                        in {
                            "μου",
                            "σου",
                            "αὐτοῦ",
                            "αὐτῆς",
                            "ἡμῶν",
                            "ὑμῶν",
                            "αὐτῶν",
                            "οὐ",
                            "μή",
                            "δέ",
                            "γάρ",
                            "οὖν",
                        }
                        for k in range(subj[0] + 1, i)
                    )
                )
                if (
                    subj[1] == "P"
                    and subj[2] in {"M", "F"}
                    and between_ok
                    and any(
                        m.pos == "V-"
                        and m.mood in {"I", "S"}
                        and m.person == "3"
                        and m.number == "S"
                        for _, m in p
                    )
                    and not any(m.pos == "V-" and m.number == "P" for _, m in p)
                ):
                    violations.append(
                        AgreementViolation(
                            kind="subj_verb_num",
                            index=subj[0],
                            target_index=i,
                            detail=f"{words[subj[0]]}(P/{subj[2]}) .. {w}(3S)",
                        )
                    )
                    subj = None
                    continue
                is_homoios_w = any(lem == "ὅμοιος" and m.pos == "A-" for lem, m in p)
                if (
                    i <= subj[0] + 1
                    and not is_homoios_w
                    and words[subj[0]].lower() not in {"ἥτις", "ὅστις"}
                ):
                    continue
                # Reset if a transitive/non-copula finite verb intervenes
                if any(
                    lem not in {"εἰμί", "γίνομαι"}
                    and m.pos == "V-"
                    and m.mood in {"I", "S", "M", "O"}
                    for lem, m in p
                ):
                    subj = None
                    continue
                # Check predicate adjective (`ὅμοιος`, or adj directly before copula
                # `ἐστίν`/`ἦν`/`ἔσται`, e.g. `ἡ εἰκών ... φανερόν ἐστίν`) or
                # middle/passive participle (`ἐστρωμένον`, `κρεμάμενον`).
                next_is_cop = i + 1 < n and _norm_accent(words[i + 1]).lower() in {
                    "ἐστίν",
                    "ἐστί",
                    "εἰσίν",
                    "εἰσί",
                    "ἦν",
                    "ἔσται",
                }
                pred_cands = [
                    (lem, m)
                    for lem, m in p
                    if (
                        (lem == "ὅμοιος" and m.pos == "A-")
                        or (
                            next_is_cop
                            and m.pos == "A-"
                            and lem
                            in {
                                "φανερός",
                                "κρυπτός",
                                "ἄξιος",
                                "καλός",
                                "ἀγαθός",
                                "δίκαιος",
                                "μέγας",
                                "μικρός",
                                "πλήρης",
                                "κενός",
                            }
                        )
                        or (m.pos == "V-" and m.mood == "P" and m.voice in {"P", "M"})
                    )
                    and m.case in {"N", "A"}
                ]
                prev_low = _norm_accent(words[i - 1]).lower() if i > 0 else ""
                if (
                    pred_cands
                    and prev_low not in _ARTICLE_PARSES
                    and not any(m.pos == "P-" for _, m in ps[i - 1])
                ):
                    s_num, s_gen = subj[1], subj[2]
                    if s_gen in {"M", "F"} and not any(
                        m.case == "N" and m.number == s_num and m.gender == s_gen
                        for _, m in pred_cands
                    ):
                        if all(
                            m.gender != s_gen or m.number != s_num
                            for _, m in pred_cands
                        ):
                            violations.append(
                                AgreementViolation(
                                    kind="pred_gender",
                                    index=subj[0],
                                    target_index=i,
                                    detail=f"{words[subj[0]]}({s_num}/{s_gen}) .. {w}",
                                )
                            )
                            subj = None
                    elif is_homoios_w:
                        # Once `ἡ βασιλεία ὁμοία ...` has been checked, subsequent
                        # participles modify the parable noun, not `ἡ βασιλεία`.
                        subj = None

        return violations

    def agreement_penalty(self, words: Sequence[str]) -> float:
        """Return non-negative penalty proportional to clear morph agreement errors."""
        viols = self.detect_violations(words)
        if not viols:
            return 0.0
        pen = 0.0
        for v in viols:
            if v.kind in {"art_head", "prep_case", "subj_verb_num", "pred_gender"}:
                pen += 1.0
            elif v.kind in {"app_dative", "adj_noun", "noun_mod"}:
                pen += 0.75
            else:
                pen += 0.50
        return pen

    def _safe_inflect(
        self,
        lemma: str,
        target: GreekMorphFeatures,
        orig_word: str,
    ) -> str | None:
        """Inflect ``lemma`` to ``target`` ONLY if verified parseable & non-trivial."""
        cand = self._paradigm.inflect(lemma, target)
        if not cand or _norm_accent(cand) == _norm_accent(orig_word):
            return None
        # Reject unchanged dictionary lemma fallback when asking for an oblique form
        if (
            _strip_bare(cand) == _strip_bare(lemma)
            and target.case in {"G", "D", "A"}
            and target.pos == "V-"
        ):
            return None
        # Verify the inflected form is parseable (either in paradigm/analyzer table
        # or parsed non-UNK by analyzer with matching target features).
        c_parses = self.parses(cand)
        if not c_parses or all(m.pos == "UNK" for _, m in c_parses):
            return None
        if target.case is not None and not any(
            m.case == target.case
            and (target.number is None or m.number == target.number)
            and (target.gender is None or m.gender is None or m.gender == target.gender)
            for _, m in c_parses
        ):
            return None
        return _norm_accent(cand)

    def _inflect_participle_simple(
        self,
        word: str,
        lemma: str,
        src_mf: GreekMorphFeatures,
        target_case: str,
        target_num: str,
        target_gen: str,
    ) -> str | None:
        """Try empirical paradigm first, then regular -μενος/-μένη/-μενον ending."""
        tgt = src_mf.model_copy(
            update={"case": target_case, "number": target_num, "gender": target_gen}
        )
        emp = self._safe_inflect(lemma, tgt, word)
        if emp is not None:
            return emp
        # Regular middle/passive participle in -μενος/-μένη/-μενον (1st/2nd decl)
        w_n = _norm_accent(word)
        for suf in (
            "μένον",
            "μένος",
            "μένη",
            "μένων",
            "μένοι",
            "μένους",
            "μένας",
            "μένῳ",
            "μένῃ",
            "μενον",
            "μενος",
            "μενη",
            "μενων",
            "μενοι",
            "μενους",
            "μενας",
            "μενω",
            "μενῳ",
            "μενῃ",
        ):
            if w_n.endswith(suf) and len(w_n) > len(suf) + 2:
                stem = w_n[: -len(suf)]
                end_map = {
                    ("N", "S", "M"): "μενος",
                    ("N", "S", "F"): "μένη",
                    ("N", "S", "N"): "μενον",
                    ("G", "S", "M"): "μένου",
                    ("G", "S", "F"): "μένης",
                    ("G", "S", "N"): "μένου",
                    ("D", "S", "M"): "μένῳ",
                    ("D", "S", "F"): "μένῃ",
                    ("D", "S", "N"): "μένῳ",
                    ("A", "S", "M"): "μενον",
                    ("A", "S", "F"): "μένην",
                    ("A", "S", "N"): "μενον",
                    ("N", "P", "M"): "μενοι",
                    ("N", "P", "F"): "μεναι",
                    ("N", "P", "N"): "μενα",
                    ("G", "P", "M"): "μένων",
                    ("G", "P", "F"): "μένων",
                    ("G", "P", "N"): "μένων",
                    ("D", "P", "M"): "μένοις",
                    ("D", "P", "F"): "μέναις",
                    ("D", "P", "N"): "μένοις",
                    ("A", "P", "M"): "μένους",
                    ("A", "P", "F"): "μένας",
                    ("A", "P", "N"): "μενα",
                }
                new_end = end_map.get((target_case, target_num, target_gen))
                if not new_end:
                    return None
                # Remove acute/circumflex on stem if suffix carries accent
                bare_stem = stem
                if any(ch in new_end for ch in "άέήίόύώᾶῆῖῦῶᾴῄῴᾷῇῷ"):
                    nfd = unicodedata.normalize("NFD", stem)
                    bare_stem = unicodedata.normalize(
                        "NFC",
                        "".join(
                            ch for ch in nfd if ch not in {"\u0301", "\u0300", "\u0342"}
                        ),
                    )
                cand = _norm_accent(bare_stem + new_end)
                if cand != w_n:
                    p = self._analyzer.analyze_surface(cand)[0]
                    if p[1].pos != "UNK":
                        return cand
        return None

    def propose_repairs(
        self,
        words: Sequence[str],
        *,
        max_candidates: int = 8,
    ) -> list[list[str]]:
        """Generate verified, parseable morphological agreement repair candidates."""
        viols = self.detect_violations(words)
        if not viols:
            return []

        ps = [self.parses(w) for w in words]
        per_viol_edits: list[list[dict[int, str]]] = []

        for v in viols:
            edits: list[dict[int, str]] = []
            if v.kind == "art_head":
                i, j = v.index, v.target_index
                w0, w1 = words[i], words[j]
                p0, p1 = ps[i], ps[j]
                # Check if a strict preposition immediately precedes the article
                req_case: str | None = None
                if i > 0:
                    pr = [
                        lem
                        for lem, m in ps[i - 1]
                        if m.pos == "P-" and lem in _STRICT_PREP_CASE
                    ]
                    if pr:
                        req_case = _STRICT_PREP_CASE[pr[0]]

                # When a Genitive article (`τοῦ`/`τῆς`/`τῶν`) follows a noun
                # (e.g. `ἡ ἀκοή τοῦ στάμνος`), prefer inflecting the head noun
                # to Genitive (`τοῦ στάμνου`) BEFORE changing the article to Nom.
                prefer_head_first = (
                    _norm_accent(w0).lower() in {"τοῦ", "τῆς", "τῶν"}
                    and i > 0
                    and any(m.pos == "N-" for _, m in ps[i - 1])
                )
                art_edits: list[dict[int, str]] = []
                head_edits: list[dict[int, str]] = []

                # Option A: fix the article to match head w1's valid (case, num, gen),
                # skipping OOV-synthesized `-ων/-ῶν` GPM fallback lemmas (e.g. fake
                # `σίτευος` for `σιτεύων`) so we never rewrite `ὁ` -> `τῶν` on an
                # unattested participle-shaped surface.
                hd1 = [
                    (lem, m)
                    for lem, m in p1
                    if m.pos in {"N-", "A-"} or (m.pos == "V-" and m.mood == "P")
                ]
                w1_norm = _norm_accent(w1)
                w1_attested = w1_norm in self._multi or w1_norm.lower() in self._multi
                seen_arts: set[str] = set()
                for l1, m1 in hd1:
                    if (
                        not w1_attested
                        and _strip_bare(l1) not in self._analyzer._known_lemmas
                        and w1_norm.lower().endswith(("ων", "ῶν"))
                    ):
                        continue
                    if not (m1.case and m1.number and m1.gender):
                        continue
                    if req_case is not None and m1.case != req_case:
                        continue
                    art_fix = _ARTICLE_BY_CNG.get((m1.case, m1.number, m1.gender))
                    if (
                        art_fix
                        and art_fix != _norm_accent(w0).lower()
                        and art_fix not in seen_arts
                    ):
                        seen_arts.add(art_fix)
                        art_edits.append({i: art_fix})

                # Option B: fix head w1 to match article w0's (case, num, gen)
                seen_hds: set[str] = set()
                if hd1:
                    l1, m1 = hd1[0]
                    for _la, r in p0:
                        if not (r.case and r.number and r.gender):
                            continue
                        if req_case is not None and r.case != req_case:
                            continue
                        if m1.pos == "V-" and m1.mood == "P":
                            h_fix = self._inflect_participle_simple(
                                w1, l1, m1, r.case, r.number, r.gender
                            )
                        else:
                            # Keep noun's inherent gender when re-inflecting noun
                            # (allowing M/F interchange on common-gender nouns).
                            tgt_g = (
                                r.gender
                                if (
                                    l1 in _COMMON_GENDER_LEMMAS
                                    and r.gender in {"M", "F"}
                                )
                                else (
                                    m1.gender
                                    if m1.pos == "N-" and m1.gender
                                    else r.gender
                                )
                            )
                            if tgt_g != r.gender:
                                continue
                            h_fix = self._safe_inflect(
                                l1,
                                m1.model_copy(
                                    update={
                                        "case": r.case,
                                        "number": r.number,
                                        "gender": tgt_g,
                                    }
                                ),
                                w1,
                            )
                        if h_fix and h_fix not in seen_hds:
                            seen_hds.add(h_fix)
                            head_edits.append({j: h_fix})
                    # Option C: fix BOTH article + head to head's inherent gender
                    # keeping article's case (or req_case) and trying both S/P
                    if m1.pos == "N-" and m1.gender:
                        c_tgt = req_case or p0[0][1].case
                        if c_tgt:
                            for num_try in (m1.number, p0[0][1].number):
                                if not num_try:
                                    continue
                                a_fix = _ARTICLE_BY_CNG.get((c_tgt, num_try, m1.gender))
                                h_fix = self._safe_inflect(
                                    l1,
                                    m1.model_copy(
                                        update={
                                            "case": c_tgt,
                                            "number": num_try,
                                            "gender": m1.gender,
                                        }
                                    ),
                                    w1,
                                )
                                if a_fix and h_fix and h_fix not in seen_hds:
                                    seen_hds.add(h_fix)
                                    head_edits.append({i: a_fix, j: h_fix})
                if prefer_head_first:
                    edits.extend(head_edits)
                    edits.extend(art_edits)
                else:
                    edits.extend(art_edits)
                    edits.extend(head_edits)

            elif v.kind == "prep_case":
                i, j = v.index, v.target_index
                w1 = words[j]
                p1 = ps[j]
                pr = [
                    lem
                    for lem, m in ps[i]
                    if m.pos == "P-" and lem in _STRICT_PREP_CASE
                ]
                if pr:
                    req = _STRICT_PREP_CASE[pr[0]]
                    if _norm_accent(w1).lower() in _ARTICLE_PARSES:
                        r0 = p1[0][1]
                        if r0.number and r0.gender:
                            a_fix = _ARTICLE_BY_CNG.get((req, r0.number, r0.gender))
                            if a_fix:
                                edits.append({j: a_fix})
                    else:
                        for l1, m1 in p1[:2]:
                            if m1.pos in {"N-", "A-"} and m1.number:
                                h_fix = self._safe_inflect(
                                    l1, m1.model_copy(update={"case": req}), w1
                                )
                                if h_fix:
                                    edits.append({j: h_fix})
                                    break

            elif v.kind == "homoios_case":
                j = v.target_index
                w1 = words[j]
                for l1, m1 in ps[j][:2]:
                    if m1.pos == "N-" and m1.number:
                        h_fix = self._safe_inflect(
                            l1, m1.model_copy(update={"case": "D"}), w1
                        )
                        if h_fix:
                            edits.append({j: h_fix})
                            break

            elif v.kind == "app_dative":
                j = v.target_index
                w1 = words[j]
                for l1, m1 in ps[j][:2]:
                    if m1.pos == "N-":
                        h_fix = self._safe_inflect(
                            l1, m1.model_copy(update={"case": "D", "number": "S"}), w1
                        )
                        if h_fix:
                            edits.append({j: h_fix})
                            break

            elif v.kind == "adj_noun":
                j = v.target_index
                w1 = words[j]
                for l1, m1 in ps[j][:2]:
                    if m1.pos == "N-":
                        h_fix = self._safe_inflect(
                            l1, m1.model_copy(update={"case": "N", "number": "S"}), w1
                        )
                        if h_fix:
                            edits.append({j: h_fix})
                            break

            elif v.kind == "subj_verb_num":
                j = v.target_index
                wt = words[j]
                for lt, mt in ps[j][:2]:
                    if mt.pos == "V-" and mt.person == "3" and mt.number == "S":
                        v_fix = self._safe_inflect(
                            lt, mt.model_copy(update={"number": "P"}), wt
                        )
                        if v_fix:
                            edits.append({j: v_fix})
                            break

            elif v.kind == "noun_mod":
                h_idx, t_idx = v.index, v.target_index
                wt = words[t_idx]
                h_hits = [
                    m
                    for _, m in ps[h_idx]
                    if m.pos == "N-" and m.case and m.number and m.gender
                ]
                if h_hits:
                    hm = h_hits[0]
                    h_case, h_num, h_gen = hm.case, hm.number, hm.gender
                    if h_case and h_num and h_gen:
                        for lt, mt in ps[t_idx][:2]:
                            if mt.pos == "V-" and mt.mood == "P":
                                t_fix = self._inflect_participle_simple(
                                    wt, lt, mt, h_case, h_num, h_gen
                                )
                            else:
                                t_fix = self._safe_inflect(
                                    lt,
                                    mt.model_copy(
                                        update={
                                            "case": h_case,
                                            "number": h_num,
                                            "gender": h_gen,
                                        }
                                    ),
                                    wt,
                                )
                            if t_fix:
                                edits.append({t_idx: t_fix})
                                break

            elif v.kind == "pred_gender":
                s_idx, t_idx = v.index, v.target_index
                s_low = _norm_accent(words[s_idx]).lower()
                if s_low == "ἥτις":
                    s_num, s_gen = "S", "F"
                elif s_low == "ὅστις":
                    s_num, s_gen = "S", "M"
                else:
                    s_hits = [
                        m for _, m in ps[s_idx] if m.pos == "N-" and m.case == "N"
                    ]
                    s_num = s_hits[0].number if s_hits else None
                    s_gen = s_hits[0].gender if s_hits else None
                if s_num and s_gen:
                    wt = words[t_idx]
                    for lt, mt in ps[t_idx][:2]:
                        if mt.pos == "V-" and mt.mood == "P":
                            t_fix = self._inflect_participle_simple(
                                wt, lt, mt, "N", s_num, s_gen
                            )
                        else:
                            t_fix = self._safe_inflect(
                                lt,
                                mt.model_copy(
                                    update={
                                        "case": "N",
                                        "number": s_num,
                                        "gender": s_gen,
                                    }
                                ),
                                wt,
                            )
                        if t_fix:
                            edits.append({t_idx: t_fix})
                            break

            if edits:
                per_viol_edits.append(edits[:3])

        if not per_viol_edits:
            return []

        candidates: list[list[str]] = []
        seen_cands: set[tuple[str, ...]] = {tuple(words)}

        def _apply(edit_map: dict[int, str]) -> None:
            new_w = list(words)
            for idx, rep in edit_map.items():
                new_w[idx] = rep
            key = tuple(new_w)
            if key not in seen_cands:
                seen_cands.add(key)
                candidates.append(new_w)

        # 1. Combined fix applying the top repair for every non-overlapping violation
        if len(per_viol_edits) >= 2:
            combo: dict[int, str] = {}
            for e_list in per_viol_edits:
                for k, val in e_list[0].items():
                    combo.setdefault(k, val)
            if combo:
                _apply(combo)

        # 2. Individual single-violation alternatives so neural reranker chooses best
        for e_list in per_viol_edits:
            for ed in e_list:
                _apply(ed)
                if len(candidates) >= max_candidates:
                    return candidates

        return candidates[:max_candidates]
