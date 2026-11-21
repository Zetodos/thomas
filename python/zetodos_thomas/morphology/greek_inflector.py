# SPDX-License-Identifier: Apache-2.0
"""Empirical + rule-based Koine Greek analyzer, inflector, and sandhi realizer.

Trained on MorphGNT/SBLGNT (with optional ``exclude_unit_ids`` so NT holdout
verses never leak into paradigm selection) plus optional CCL/LXX vocabulary.
Falls back to pure-Python regular Koine nominal, adjectival, participial, and
verbal paradigms + MIT ``greek-accentuation`` accent/breathing placement.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from greek_accentuation import accentuation, characters, syllabify

from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt
from zetodos_thomas.ingest.ot_swete import parse_swete_file
from zetodos_thomas.schemas.linguistic import GreekMorphFeatures

if TYPE_CHECKING:
    from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon

_SMOOTH_APOSTROPHE = "\u2019"
_PUNCT_CLOSE = {".", ",", ";", "·", ":", "!", "?", ";", ")", "]", "»"}
_PUNCT_OPEN = {"(", "[", "«"}

_PREVERBS: tuple[str, ...] = (
    "περι",
    "προσ",
    "ἀπο",
    "ἀνα",
    "δια",
    "εἰσ",
    "ἐπι",
    "κατα",
    "μετα",
    "παρα",
    "ὑπερ",
    "ὑπο",
    "συν",
    "ἐξ",
    "ἐκ",
    "ἐν",
    "προ",
)
# Elided preverb -> full vowel restoration if augmented
_TEMPORAL_AUGMENT: dict[str, str] = {
    "αι": "ῃ",
    "αυ": "ηυ",
    "ει": "ῃ",
    "ευ": "ηυ",
    "οι": "ῳ",
    "α": "η",
    "ε": "η",
    "ο": "ω",
    "ι": "ι",
    "υ": "υ",
}

_ELIDABLE_PARTICLES = {
    "δέ": "δ’",
    "ἀλλά": "ἀλλ’",
    "ἀπό": "ἀπ’",
    "διά": "δι’",
    "ἐπί": "ἐπ’",
    "κατά": "κατ’",
    "μετά": "μετ’",
    "παρά": "παρ’",
    "ὑπό": "ὑπ’",
}

_ENCLITIC_LOWER = frozenset(
    {
        "τε",
        "γε",
        "τοι",
        "περ",
        "πως",
        "που",
        "ποθέν",
        "ποτε",
        "τις",
        "τι",
        "τινος",
        "τινι",
        "τινα",
        "τινες",
        "τινων",
        "τισι",
        "τισιν",
        "τινας",
        "μου",
        "μοι",
        "με",
        "σου",
        "σοι",
        "σε",
        "εἰμι",
        "εἶ",
        "ἐστι",
        "ἐστιν",
        "ἐσμεν",
        "ἐστε",
        "εἰσι",
        "εἰσιν",
        "φημί",
        "φησί",
        "φησίν",
    }
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _strip_diacritics_keep_base(word: str) -> str:
    return characters.strip_accents(characters.strip_breathing(word)).lower()


def _bare_greek_key(word: str) -> str:
    """Fold apostrophes, lowercase, and strip all combining diacritics + final sigma."""
    normalized = (
        word.replace("\u02bc", _SMOOTH_APOSTROPHE)
        .replace("\u02bd", _SMOOTH_APOSTROPHE)
        .replace("'", _SMOOTH_APOSTROPHE)
        .replace("\u1fbf", _SMOOTH_APOSTROPHE)
    )
    nfd = unicodedata.normalize("NFD", normalized)
    bare = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").lower()
    if len(bare) > 1 and bare.endswith("σ"):
        bare = bare[:-1] + "ς"
    return bare


def _normalize_polytonic_for_match(word: str) -> str:
    """Normalize apostrophes, grave -> acute, and lowercase keeping final sigma."""
    normalized = (
        word.replace("\u02bc", _SMOOTH_APOSTROPHE)
        .replace("\u02bd", _SMOOTH_APOSTROPHE)
        .replace("'", _SMOOTH_APOSTROPHE)
        .replace("\u1fbf", _SMOOTH_APOSTROPHE)
    )
    nfd = unicodedata.normalize("NFD", normalized)
    replaced = nfd.replace("\u0300", "\u0301")  # VARIA -> OXIA
    low = unicodedata.normalize("NFC", replaced).lower()
    if len(low) > 1 and low.endswith("σ"):
        low = low[:-1] + "ς"
    return low


_ELIDED_SURFACE_PARSE: dict[str, tuple[str, str]] = {
    "δ’": ("δέ", "C-:--------"),
    "ἀλλ’": ("ἀλλά", "C-:--------"),
    "ἀπ’": ("ἀπό", "P-:--------"),
    "ἀφ’": ("ἀπό", "P-:--------"),
    "δι’": ("διά", "P-:--------"),
    "ἐπ’": ("ἐπί", "P-:--------"),
    "ἐφ’": ("ἐπί", "P-:--------"),
    "κατ’": ("κατά", "P-:--------"),
    "καθ’": ("κατά", "P-:--------"),
    "μετ’": ("μετά", "P-:--------"),
    "μεθ’": ("μετά", "P-:--------"),
    "παρ’": ("παρά", "P-:--------"),
    "ὑπ’": ("ὑπό", "P-:--------"),
    "ὑφ’": ("ὑπό", "P-:--------"),
}
_ELIDED_BARE_PARSE: dict[str, tuple[str, str]] = {
    **{_bare_greek_key(k): v for k, v in _ELIDED_SURFACE_PARSE.items()},
    "θειη": ("τίθημι", "V-:3AAO-S--"),
}

# (bare_prefix, canonical_preverb_with_breathing, may_carry_syllabic_augment)
_PREVERB_ANALYSIS_RULES: tuple[tuple[str, str, bool], ...] = tuple(
    sorted(
        (
            ("προσε", "προσ", True),
            ("περιε", "περι", True),
            ("υπερε", "ὑπερ", True),
            ("προσ", "προσ", False),
            ("περι", "περι", False),
            ("υπερ", "ὑπερ", False),
            ("συνε", "συν", True),
            ("εισε", "εἰσ", True),
            ("κατε", "κατα", True),
            ("παρε", "παρα", True),
            ("μετε", "μετα", True),
            ("επι", "ἐπι", False),
            ("επε", "ἐπι", True),
            ("απο", "ἀπο", False),
            ("απε", "ἀπο", True),
            ("ανα", "ἀνα", False),
            ("ανε", "ἀνα", True),
            ("κατα", "κατα", False),
            ("δια", "δια", False),
            ("διε", "δια", True),
            ("παρα", "παρα", False),
            ("μετα", "μετα", False),
            ("υπο", "ὑπο", False),
            ("υπε", "ὑπο", True),
            ("εξε", "ἐκ", True),
            ("προε", "προ", True),
            ("συν", "συν", False),
            ("εισ", "εἰσ", False),
            ("καθ", "κατα", True),
            ("μεθ", "μετα", True),
            ("κατ", "κατα", True),
            ("παρ", "παρα", True),
            ("μετ", "μετα", True),
            ("ενε", "ἐν", True),
            ("προ", "προ", False),
            ("εφ", "ἐπι", True),
            ("αφ", "ἀπο", True),
            ("υφ", "ὑπο", True),
            ("επ", "ἐπι", True),
            ("απ", "ἀπο", True),
            ("υπ", "ὑπο", True),
            ("δι", "δια", True),
            ("εξ", "ἐκ", True),
            ("εκ", "ἐκ", False),
            ("εν", "ἐν", False),
        ),
        key=lambda item: -len(item[0]),
    )
)

# Suffix stripping tables sorted longest-first within each group:
# (bare_suffix, candidate_nom_endings, morphgnt_sig, allow_oov_synthesis)
_VERB_SUFFIX_RULES: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    ("θησονται", ("ω",), "V-:3PFI-P-", True),
    ("ησονται", ("ω",), "V-:3PFI-M-", True),
    ("θησεται", ("ω",), "V-:3SFI-P-", True),
    ("ησεται", ("ω",), "V-:3SFI-M-", True),
    ("σουσιν", ("ω",), "V-:3PFI-A-", False),
    ("εσθαι", ("ομαι", "ω"), "V-:--PMN--", True),
    ("ασθαι", ("ω", "ομαι"), "V-:--AMN--", True),
    ("ουσιν", ("ω",), "V-:3PPI-A-", True),
    ("ονται", ("ομαι", "ω"), "V-:3PPI-M-", True),
    ("ησομεν", ("εω", "αω", "ω"), "V-:1FAI-P--", False),
    ("ημενος", ("εω", "αω"), "V-:-XPPNSM-", False),
    ("ημενον", ("εω", "αω"), "V-:-XPPNSN-", False),
    ("σουσι", ("ω",), "V-:3PFI-A-", False),
    ("θησαν", ("ω",), "V-:3PAI-P-", True),
    ("ουμεν", ("εω", "ω"), "V-:1PPI-A-", True),
    ("ουσι", ("ω",), "V-:3PPI-A-", True),
    ("οντο", ("ομαι", "ω"), "V-:3PII-M-", True),
    ("ειται", ("εω", "ω"), "V-:3SPI-M-", False),
    ("σομεν", ("ω",), "V-:1FAI-P--", False),
    ("οντας", ("ω",), "V-:-PAPAPM-", False),
    ("οντες", ("ω",), "V-:-PAPNPM-", False),
    ("οντος", ("ω",), "V-:-PAPGSM-", False),
    ("ουσης", ("ω",), "V-:-PAPGSF-", False),
    ("ουσαν", ("ω",), "V-:-PAPASF-", False),
    ("ημενη", ("εω", "αω"), "V-:-XPPNSF-", False),
    ("μενος", ("ω", "ομαι"), "V-:-PPPNSM-", False),
    ("μενον", ("ω", "ομαι"), "V-:-PPPNSN-", False),
    ("μενοι", ("ω", "ομαι"), "V-:-PPPNPM-", False),
    ("σασθω", ("ω", "ομαι"), "V-:3AMD-S--", False),
    ("εται", ("ομαι", "ω"), "V-:3SPI-M-", True),
    ("οντα", ("ω",), "V-:-PAPASM-", False),
    ("ουσα", ("ω",), "V-:-PAPNSF-", False),
    ("σητε", ("ω",), "V-:2AAS-P--", False),
    ("ασθω", ("ω", "ομαι"), "V-:3AMD-S--", False),
    ("μενη", ("ω", "ομαι"), "V-:-PPPNSF-", False),
    ("μενα", ("ω", "ομαι"), "V-:-PPPNPN-", False),
    ("ετο", ("ω", "ομαι"), "V-:3SII-M-", True),
    ("ετε", ("ω",), "V-:2PPI-A-", False),
    ("ειν", ("ω",), "V-:--PAN--", False),
    ("σον", ("ω", "ζω", "δω", "τω"), "V-:2AAD-S--", False),
    ("ον", ("ω",), "V-:2PAD-S--", False),
    ("ει", ("εω",), "V-:3PAI-S--", False),
)

_NOUN_SUFFIX_RULES: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    ("ματων", ("μα",), "N-:----GPN-", True),
    ("ματος", ("μα",), "N-:----GSN-", True),
    ("μασιν", ("μα",), "N-:----DPN-", True),
    ("ματι", ("μα",), "N-:----DSN-", True),
    ("ματα", ("μα",), "N-:----APN-", True),
    ("μασι", ("μα",), "N-:----DPN-", True),
    ("σεως", ("σις",), "N-:----GSF-", True),
    ("σεων", ("σις",), "N-:----GPF-", True),
    ("σεις", ("σις",), "N-:----APF-", True),
    ("σεσι", ("σις",), "N-:----DPF-", True),
    ("ψεως", ("ψις",), "N-:----GSF-", False),
    ("ξεως", ("ξις",), "N-:----GSF-", False),
    ("σει", ("σις",), "N-:----DSF-", False),
    ("ους", ("ος",), "N-:----APM-", True),
    ("οις", ("ος", "ον"), "N-:----DPM-", True),
    ("αις", ("α", "η"), "N-:----DPF-", True),
    ("ων", ("ος", "η", "α", "ον"), "N-:----GPM-", True),
    ("ας", ("α", "η"), "N-:----APF-", False),
    ("ης", ("η",), "N-:----GSF-", False),
    ("ην", ("η",), "N-:----ASF-", False),
    ("ου", ("ος", "ον"), "N-:----GSM-", False),
    ("ον", ("ος", "ον"), "N-:----ASM-", False),
    ("οι", ("ος",), "N-:----NPM-", False),
    ("αι", ("α", "η"), "N-:----NPF-", False),
)


def _strip_initial_breathing(word: str) -> str:
    """Remove smooth/rough breathing from leading vowel/diphthong, keeping accents."""
    if not word:
        return word
    head_len = min(len(word), 2)
    head_nfd = unicodedata.normalize("NFD", word[:head_len])
    cleaned = "".join(ch for ch in head_nfd if ch not in {"\u0313", "\u0314"})
    return _nfc(cleaned + word[head_len:])


def _compose_preverb_lemma(canon_prev: str, simplex_lemma: str) -> str:
    """Join a Koine preverb (e.g. ``ἐπι``) onto ``simplex_lemma`` cleanly."""
    if not simplex_lemma:
        return simplex_lemma
    bare_prev = _bare_greek_key(canon_prev).replace("ς", "σ")
    simplex_nfd = unicodedata.normalize("NFD", _nfc(simplex_lemma))
    first_alpha = ""
    for ch in simplex_nfd[:3]:
        if ch.isalpha():
            first_alpha = ch.lower()
            break
    starts_vowel = first_alpha in {"α", "ε", "η", "ι", "ο", "υ", "ω"}
    has_rough = syllabify.debreath(simplex_lemma).startswith("h")
    prev_out = bare_prev
    if starts_vowel:
        if prev_out == "εκ":
            prev_out = "εξ"
        elif prev_out not in {"περι", "προ", "προσ", "εισ", "συν", "εν", "υπερ"} and (
            prev_out.endswith(("α", "ε", "ι", "ο"))
        ):
            prev_out = prev_out[:-1]
            if has_rough and prev_out.endswith(("π", "τ", "κ")):
                asp = {"π": "φ", "τ": "θ", "κ": "χ"}
                prev_out = prev_out[:-1] + asp[prev_out[-1]]
    body = _strip_initial_breathing(simplex_lemma)
    return _preserve_initial_breathing(prev_out + body, canon_prev)


def _apply_recessive_with_breathing(bare: str, orig_lemma: str) -> str:
    """Apply recessive accent while preserving initial vowel/diphthong breathing."""
    cleaned = characters.strip_accents(characters.strip_breathing(bare))
    try:
        accented = accentuation.recessive(cleaned)
    except Exception:
        return _nfc(cleaned)
    if not accented:
        return accented
    return _preserve_initial_breathing(accented, orig_lemma)


def _preserve_initial_breathing(accented: str, orig_lemma: str) -> str:
    """Attach original lemma's smooth/rough breathing onto an already-accented form."""
    if not accented:
        return accented
    debr = syllabify.debreath(orig_lemma)
    has_rough = debr.startswith("h")
    first_two = characters.strip_accents(
        characters.strip_breathing(accented[:2])
    ).lower()
    starts_vowel = first_two[:1] in {"α", "ε", "η", "ι", "ο", "υ", "ω"}
    if not starts_vowel and not accented.startswith(("ρ", "ῥ")):
        return _nfc(accented)
    if has_rough:
        target = "h" + characters.strip_accents(characters.strip_breathing(accented))
        rebr = syllabify.rebreath(target)
        acc_list = list(accented)
        re_list = list(rebr)
        for idx, ch in enumerate(acc_list):
            if characters.accent(ch):
                acc_mark = characters.accent(ch)
                if idx < len(re_list) and acc_mark:
                    return _nfc(
                        "".join(
                            [
                                *re_list[:idx],
                                characters.add_diacritic(re_list[idx], acc_mark),
                                *re_list[idx + 1 :],
                            ]
                        )
                    )
        return _nfc(rebr)
    is_diph = first_two in {"αι", "αυ", "ει", "ευ", "οι", "ου", "υι", "ηυ"}
    b_idx = 1 if is_diph and len(accented) > 1 else 0
    chars = list(accented)
    if (
        first_two[:1] in {"α", "ε", "η", "ο", "ω"}
        or is_diph
        or first_two[:1] in {"ι", "υ"}
    ):
        target_i = (
            b_idx if (first_two[:1] in {"α", "ε", "η", "ο", "ω"} or is_diph) else 0
        )
        ch = chars[target_i]
        acc = characters.accent(ch)
        base_ch = characters.strip_accents(characters.strip_breathing(ch))
        upd = characters.add_diacritic(base_ch, characters.SMOOTH)
        chars[target_i] = characters.add_diacritic(upd, acc) if acc else upd
    return _nfc("".join(chars))


def _apply_persistent_with_breathing(bare: str, orig_lemma: str) -> str:
    """Place persistent accent using lemma accent type and preserve breathing."""
    cleaned = characters.strip_accents(characters.strip_breathing(bare))
    norm_lem = _nfc(orig_lemma)
    debr = syllabify.debreath(norm_lem)
    has_rough = debr.startswith("h")
    try:
        accented = accentuation.persistent(cleaned, norm_lem)
    except Exception:
        accented = accentuation.recessive(cleaned)
    first_two = characters.strip_accents(
        characters.strip_breathing(cleaned[:2])
    ).lower()
    if first_two[:1] not in {"α", "ε", "η", "ι", "ο", "υ", "ω"}:
        return _nfc(accented)
    is_diph = first_two in {"αι", "αυ", "ει", "ευ", "οι", "ου", "υι", "ηυ"}
    b_idx = 1 if is_diph and len(accented) > 1 else 0
    chars = list(accented)
    b_mark = characters.ROUGH if has_rough else characters.SMOOTH
    chars[b_idx] = characters.add_diacritic(chars[b_idx], b_mark)
    return _nfc("".join(chars))


def _augment_verb_stem(lemma: str) -> str:
    """Apply Koine syllabic or temporal augment (respecting preverbs)."""
    plain = characters.strip_accents(characters.strip_breathing(lemma)).lower()
    base_stem = plain
    if base_stem.endswith("εω"):
        base_stem = base_stem[:-2]
    elif base_stem.endswith("αω"):
        base_stem = base_stem[:-2]
    elif base_stem.endswith("οω"):
        base_stem = base_stem[:-2]
    elif base_stem.endswith("ω"):
        base_stem = base_stem[:-1]
    elif base_stem.endswith("ομαι"):
        base_stem = base_stem[:-4]

    # Check preverb prefix in the original lemma
    plain_nfd = _strip_diacritics_keep_base(lemma)
    for pre in _PREVERBS:
        if plain_nfd.startswith(pre) and len(base_stem) > len(pre) + 1:
            rest = base_stem[len(pre) :]
            aug_pre = pre
            if pre == "ἐκ" and rest[0] in {"α", "ε", "η", "ι", "ο", "υ", "ω"}:
                aug_pre = "ἐξ"
            elif pre in {"ἀπο", "ἐπι", "κατα", "μετα", "παρα", "ὑπο", "δια"}:
                aug_pre = pre[:-1]
            aug_rest = _aug_simple(rest)
            return aug_pre + aug_rest
    return _aug_simple(base_stem)


def _aug_simple(stem: str) -> str:
    if not stem:
        return stem
    for src, dst in _TEMPORAL_AUGMENT.items():
        if stem.startswith(src):
            return dst + stem[len(src) :]
    return "ε" + stem


_INDECLINABLE_LEMMAS = frozenset(
    {
        "πέντε",
        "ἕξ",
        "ἑπτά",
        "ὀκτώ",
        "ἐννέα",
        "δέκα",
        "ἕνδεκα",
        "δώδεκα",
        "ἑκατόν",
    }
)

_IRREGULAR_NOUN_PARADIGMS: dict[str, dict[tuple[str, str], str]] = {
    "οὖς": {
        ("S", "N"): "οὖς",
        ("S", "G"): "ὠτός",
        ("S", "D"): "ὠτί",
        ("S", "A"): "οὖς",
        ("S", "V"): "οὖς",
        ("P", "N"): "ὦτα",
        ("P", "G"): "ὤτων",
        ("P", "D"): "ὠσί",
        ("P", "A"): "ὦτα",
        ("P", "V"): "ὦτα",
    },
    "ὕδωρ": {
        ("S", "N"): "ὕδωρ",
        ("S", "G"): "ὕδατος",
        ("S", "D"): "ὕδατι",
        ("S", "A"): "ὕδωρ",
        ("S", "V"): "ὕδωρ",
        ("P", "N"): "ὕδατα",
        ("P", "G"): "ὑδάτων",
        ("P", "D"): "ὕδασι",
        ("P", "A"): "ὕδατα",
        ("P", "V"): "ὕδατα",
    },
    "πῦρ": {
        ("S", "N"): "πῦρ",
        ("S", "G"): "πυρός",
        ("S", "D"): "πυρί",
        ("S", "A"): "πῦρ",
        ("S", "V"): "πῦρ",
    },
    "γυνή": {
        ("S", "N"): "γυνή",
        ("S", "G"): "γυναικός",
        ("S", "D"): "γυναικί",
        ("S", "A"): "γυναῖκα",
        ("S", "V"): "γύναι",
        ("P", "N"): "γυναῖκες",
        ("P", "G"): "γυναικῶν",
        ("P", "D"): "γυναιξί",
        ("P", "A"): "γυναῖκας",
        ("P", "V"): "γυναῖκες",
    },
    "χείρ": {
        ("S", "N"): "χείρ",
        ("S", "G"): "χειρός",
        ("S", "D"): "χειρί",
        ("S", "A"): "χεῖρα",
        ("S", "V"): "χείρ",
        ("P", "N"): "χεῖρες",
        ("P", "G"): "χειρῶν",
        ("P", "D"): "χερσί",
        ("P", "A"): "χεῖρας",
        ("P", "V"): "χεῖρες",
    },
    "πούς": {
        ("S", "N"): "πούς",
        ("S", "G"): "ποδός",
        ("S", "D"): "ποδί",
        ("S", "A"): "πόδα",
        ("S", "V"): "πούς",
        ("P", "N"): "πόδες",
        ("P", "G"): "ποδῶν",
        ("P", "D"): "ποσί",
        ("P", "A"): "πόδας",
        ("P", "V"): "πόδες",
    },
}

_IRREGULAR_GENDERS: dict[str, str] = {
    "οὖς": "N",
    "ὕδωρ": "N",
    "πῦρ": "N",
    "γυνή": "F",
    "χείρ": "F",
    "πούς": "M",
}


def _regular_noun_inflect(
    lemma: str,
    *,
    case: str,
    number: str,
    gender: str | None,
) -> str | None:
    """Pure-Python regular 1st/2nd/3rd declension Koine nominal inflection."""
    norm_lem = _nfc(lemma).lower()
    if norm_lem in _INDECLINABLE_LEMMAS:
        return _nfc(lemma)
    n = number or "S"
    c = case or "N"
    if norm_lem in _IRREGULAR_NOUN_PARADIGMS:
        return _IRREGULAR_NOUN_PARADIGMS[norm_lem].get((n, c), _nfc(lemma))
    bare = characters.strip_accents(characters.strip_breathing(lemma)).lower()
    if len(bare) < 3:
        return None
    if bare.endswith("ος"):
        stem = bare[:-2]
        if gender == "F" or bare.endswith("οδος"):
            # 2nd decl feminine follows -ος/-ον/-ου/-ῳ
            pass
        if n == "S":
            end_map = {"N": "ος", "G": "ου", "D": "ῳ", "A": "ον", "V": "ε"}
        else:
            end_map = {"N": "οι", "G": "ων", "D": "οις", "A": "ους", "V": "οι"}
        return _apply_persistent_with_breathing(stem + end_map.get(c, "ος"), lemma)

    if bare.endswith("ον"):
        stem = bare[:-2]
        if n == "S":
            end_map = {"N": "ον", "G": "ου", "D": "ῳ", "A": "ον", "V": "ον"}
        else:
            end_map = {"N": "α", "G": "ων", "D": "οις", "A": "α", "V": "α"}
        return _apply_persistent_with_breathing(stem + end_map.get(c, "ον"), lemma)

    if bare.endswith("μα"):
        stem = bare[:-2] + "ματ"
        if n == "S":
            end_map = {
                "N": bare,
                "G": f"{stem}ος",
                "D": f"{stem}ι",
                "A": bare,
                "V": bare,
            }
        else:
            end_map = {
                "N": f"{stem}α",
                "G": f"{stem}ων",
                "D": f"{bare[:-2]}μασι",
                "A": f"{stem}α",
                "V": f"{stem}α",
            }
        return _apply_recessive_with_breathing(end_map.get(c, bare), lemma)

    if bare.endswith(("α", "η")):
        stem = bare[:-1]
        is_iota_rho_e = bare.endswith("α") and (
            len(stem) >= 1 and stem[-1] in {"ε", "ι", "ρ"}
        )
        v_s = "α" if is_iota_rho_e else "η"
        if n == "S":
            end_map = {
                "N": bare[-1],
                "G": f"{v_s}ς",
                "D": "ᾳ" if v_s == "α" else "ῃ",
                "A": f"{bare[-1]}ν",
                "V": bare[-1],
            }
        else:
            end_map = {"N": "αι", "G": "ων", "D": "αις", "A": "ας", "V": "αι"}
        return _apply_persistent_with_breathing(stem + end_map.get(c, bare[-1]), lemma)

    if bare.endswith(("ης", "ας")):
        stem = bare[:-2]
        v_s = bare[-2]
        if n == "S":
            end_map = {
                "N": bare[-2:],
                "G": "ου",
                "D": "ῃ" if v_s == "η" else "ᾳ",
                "A": f"{v_s}ν",
                "V": v_s,
            }
        else:
            end_map = {"N": "αι", "G": "ων", "D": "αις", "A": "ας", "V": "αι"}
        return _apply_persistent_with_breathing(stem + end_map.get(c, bare[-2:]), lemma)

    if bare.endswith("ις"):
        stem = bare[:-2]
        if n == "S":
            end_map = {
                "N": bare,
                "G": f"{stem}εως",
                "D": f"{stem}ει",
                "A": f"{stem}ιν",
                "V": f"{stem}ι",
            }
        else:
            end_map = {
                "N": f"{stem}εις",
                "G": f"{stem}εων",
                "D": f"{stem}εσι",
                "A": f"{stem}εις",
                "V": f"{stem}εις",
            }
        return _apply_recessive_with_breathing(end_map.get(c, bare), lemma)

    return None


# Principal parts for irregular/suppletive core Koine verbs:
# bare_lemma -> {tag: stem_without_ending}
# Tags:
#   "FA": fut act stem (-ω/-εις/-ει, suffix 'ῶ' if starts/ends liquid),
#   "FM": fut mid stem (-ομαι/-εται),
#   "AA": 1st aor act stem (-α/-ας/-ε),
#   "A2A": 2nd aor act stem (-ον/-ες/-ε, inf -εῖν),
#   "AP": aor pass stem (-θην/-θῶ),
#   "IMPV_2SA": special 2S aor act imperative form
_SUPPLETIVE_VERBS: dict[str, dict[str, str]] = {
    "ὁράω": {
        "FM": "ὀψ",
        "A2A": "εἰδ",
        "A2_STEM": "ἰδ",
        "AP": "ὠφθ",
        "AP_UNAWG": "ὀφθ",
    },
    "ἐσθίω": {"FM": "φαγ", "A2A": "ἐφαγ", "A2_STEM": "φαγ"},
    "γινώσκω": {
        "FM": "γνωσ",
        "AA": "ἐγνω",
        "AP": "ἐγνωσθ",
        "AP_UNAWG": "γνωσθ",
        "A2_IMP_S": "γνῶθι",
    },
    "λέγω": {
        "FA": "ἐρ",
        "A2A": "εἰπ",
        "A2_STEM": "εἰπ",
        "AP": "ἐρρηθ",
        "AP_UNAWG": "ῥηθ",
    },
    "ἔρχομαι": {"FM": "ἐλευσ", "A2A": "ἠλθ", "A2_STEM": "ἐλθ"},
    "εἰσέρχομαι": {"FM": "εἰσελευσ", "A2A": "εἰσηλθ", "A2_STEM": "εἰσελθ"},
    "ἐξέρχομαι": {"FM": "ἐξελευσ", "A2A": "ἐξηλθ", "A2_STEM": "ἐξελθ"},
    "ἀπέρχομαι": {"FM": "ἀπελευσ", "A2A": "ἀπηλθ", "A2_STEM": "ἀπελθ"},
    "βάλλω": {
        "FA": "βαλ",
        "A2A": "ἐβαλ",
        "A2_STEM": "βαλ",
        "AP": "ἐβληθ",
        "AP_UNAWG": "βληθ",
    },
    "ἐκβάλλω": {
        "FA": "ἐκβαλ",
        "A2A": "ἐξεβαλ",
        "A2_STEM": "ἐκβαλ",
        "AP": "ἐξεβληθ",
        "AP_UNAWG": "ἐκβληθ",
    },
    "πίπτω": {"FM": "πεσ", "A2A": "ἐπεσ", "A2_STEM": "πεσ"},
    "λαμβάνω": {
        "FM": "λημψ",
        "A2A": "ἐλαβ",
        "A2_STEM": "λαβ",
        "AP": "ἐλημφθ",
        "AP_UNAWG": "λημφθ",
    },
    "φέρω": {"FA": "οἰσ", "AA": "ἠνεγκ", "AP": "ἠνεχθ", "AP_UNAWG": "ἐνεχθ"},
    "πίνω": {"FM": "πι", "A2A": "ἐπι", "A2_STEM": "πι"},
    "ἔχω": {"FA": "ἑξ", "A2A": "ἐσχ", "A2_STEM": "σχ"},
    "εὑρίσκω": {
        "FA": "εὑρησ",
        "A2A": "εὑρ",
        "A2_STEM": "εὑρ",
        "AP": "εὑρεθ",
        "AP_UNAWG": "εὑρεθ",
    },
    "ἀποθνῄσκω": {"FM": "ἀποθαν", "A2A": "ἀπεθαν", "A2_STEM": "ἀποθαν"},
    "γίνομαι": {"FM": "γενησ", "A2A": "ἐγεν", "AP": "ἐγενηθ", "AP_UNAWG": "γενηθ"},
    "αἴρω": {
        "FA": "ἀρ",
        "AA": "ἠρ",
        "A_STEM": "ἀρ",
        "AP": "ἠρθ",
        "AP_UNAWG": "ἀρθ",
    },
    "ἐγείρω": {
        "FA": "ἐγερ",
        "AA": "ἠγειρ",
        "A_STEM": "ἐγειρ",
        "AP": "ἠγερθ",
        "AP_UNAWG": "ἐγερθ",
    },
    "σχίζω": {"FA": "σχισ", "AA": "ἐσχισ", "A_STEM": "σχισ"},
    "κρύπτω": {
        "FA": "κρυψ",
        "AA": "ἐκρυψ",
        "A_STEM": "κρυψ",
        "AP": "ἐκρυβ",
        "AP_UNAWG": "κρυβ",
    },
    "ἀποκαλύπτω": {"AP": "ἀπεκαλυφθ", "AP_UNAWG": "ἀποκαλυφθ"},
    "φανερόω": {"AP": "ἐφανερωθ", "AP_UNAWG": "φανερωθ"},
    "ταράσσω": {"AP": "ἐταραχθ", "AP_UNAWG": "ταραχθ"},
    "αἰσχύνω": {"AP": "ᾐσχυνθ", "AP_UNAWG": "αἰσχυνθ"},
    "παύω": {"FM": "παυσ", "AA": "ἐπαυσ", "A_STEM": "παυσ"},
    "ἀκούω": {
        "FA": "ἀκουσ",
        "AA": "ἠκουσ",
        "A_STEM": "ἀκουσ",
        "AP": "ἠκουσθ",
        "AP_UNAWG": "ἀκουσθ",
    },
    "βλέπω": {"FA": "βλεψ", "AA": "ἐβλεψ", "A_STEM": "βλεψ"},
    "διαβλέπω": {"FA": "διαβλεψ", "AA": "διεβλεψ", "A_STEM": "διαβλεψ"},
    "ζητέω": {"FA": "ζητησ", "AA": "ἐζητησ", "A_STEM": "ζητησ"},
    "ποιέω": {
        "FA": "ποιησ",
        "AA": "ἐποιησ",
        "A_STEM": "ποιησ",
        "AP": "ἐποιηθ",
        "AP_UNAWG": "ποιηθ",
    },
    "γράφω": {
        "FA": "γραψ",
        "AA": "ἐγραψ",
        "A_STEM": "γραψ",
        "AP": "ἐγραφ",
        "AP_UNAWG": "γραφ",
    },
    "φεύγω": {"FM": "φευξ", "A2A": "ἐφυγ", "A2_STEM": "φυγ"},
}


def _regular_verb_inflect(lemma: str, target: GreekMorphFeatures) -> str | None:
    """Pure-Python regular + suppletive Koine verb inflection."""
    bare = characters.strip_accents(characters.strip_breathing(lemma)).lower()
    if len(bare) < 3 or not (bare.endswith("ω") or bare.endswith("ομαι")):
        return None
    norm_lem = _nfc(lemma).lower()
    supp = _SUPPLETIVE_VERBS.get(norm_lem, {})
    is_contract_e = bare.endswith("έω") or bare.endswith("εω")
    is_contract_a = bare.endswith("άω") or bare.endswith("αω")
    if bare.endswith("ω"):
        stem = bare[:-2] if (is_contract_e or is_contract_a) else bare[:-1]
    else:
        stem = bare[:-4]

    tense = target.tense or "P"
    voice = target.voice or ("M" if bare.endswith("ομαι") else "A")
    mood = target.mood or "I"
    person = target.person or "3"
    number = target.number or "S"

    def _sigmatic_stem(st: str, *, for_aug: bool = False) -> str:
        # Handle suppletive/liquid stems first (never generate *βάλλσω)
        if not for_aug and "A_STEM" in supp:
            return supp["A_STEM"]
        if for_aug and "AA" in supp:
            return supp["AA"]
        if is_contract_e or is_contract_a:
            return st + "ησ"
        if st.endswith("σκ"):
            return st[:-2] + "σ"
        if st.endswith("λλ"):
            return st[:-1]
        if st.endswith(("λ", "ρ", "μ", "ν")):
            return st
        if st.endswith(("ζ", "δ", "τ", "θ")):
            return st[:-1] + "σ"
        if st.endswith(("π", "β", "φ")):
            return st[:-1] + "ψ"
        if st.endswith(("κ", "γ", "χ")):
            return st[:-1] + "ξ"
        return st + "σ"

    def _aor_pass_stem(*, with_augment: bool) -> str:
        if with_augment and "AP" in supp:
            return characters.strip_accents(characters.strip_breathing(supp["AP"]))
        if not with_augment and "AP_UNAWG" in supp:
            return characters.strip_accents(
                characters.strip_breathing(supp["AP_UNAWG"])
            )
        base = _augment_verb_stem(lemma) if with_augment else stem
        if is_contract_e or is_contract_a:
            return base + "ηθ"
        if bare.endswith(("όω", "οω")):
            return (
                base[:-1] if (not with_augment and base.endswith("ο")) else base
            ) + "ωθ"
        if base.endswith(("π", "β", "φ")):
            return base[:-1] + "φθ"
        if base.endswith(("σσ", "ττ")):
            return base[:-2] + "χθ"
        if base.endswith(("κ", "γ", "χ")):
            return base[:-1] + "χθ"
        if base.endswith(("ζ", "δ", "τ", "θ")):
            return base[:-1] + "σθ"
        return base + "θ"

    if mood == "P":
        case = target.case or "N"
        gnd = target.gender or "M"
        if tense == "A" and voice == "A":
            if "A2_STEM" in supp:
                a2 = supp["A2_STEM"]
                if gnd == "F":
                    f_end = {
                        ("S", "N"): "οῦσα",
                        ("S", "G"): "ούσης",
                        ("S", "D"): "ούσῃ",
                        ("S", "A"): "οῦσαν",
                        ("P", "N"): "οῦσαι",
                        ("P", "G"): "ουσῶν",
                        ("P", "D"): "ούσαις",
                        ("P", "A"): "ούσας",
                    }.get((number, case), "οῦσα")
                    return _preserve_initial_breathing(a2 + f_end, lemma)
                a2_map = {
                    ("S", "M", "N"): "ών",
                    ("S", "M", "G"): "όντος",
                    ("S", "M", "D"): "όντι",
                    ("S", "M", "A"): "όντα",
                    ("S", "M", "V"): "ών",
                    ("S", "N", "N"): "όν",
                    ("S", "N", "G"): "όντος",
                    ("S", "N", "D"): "όντι",
                    ("S", "N", "A"): "όν",
                    ("S", "N", "V"): "όν",
                    ("P", "M", "N"): "όντες",
                    ("P", "M", "G"): "όντων",
                    ("P", "M", "D"): "οῦσι",
                    ("P", "M", "A"): "όντας",
                    ("P", "M", "V"): "όντες",
                    ("P", "N", "N"): "όντα",
                    ("P", "N", "G"): "όντων",
                    ("P", "N", "D"): "οῦσι",
                    ("P", "N", "A"): "όντα",
                    ("P", "N", "V"): "όντα",
                }
                return _preserve_initial_breathing(
                    a2 + a2_map.get((number, gnd, case), "ών"), lemma
                )
            astem = _sigmatic_stem(stem)
            if gnd == "F":
                return _apply_recessive_with_breathing(f"{astem}ασα", lemma)
            is_neut_sg = gnd == "N" and number == "S" and case in {"N", "A"}
            end = "αν" if is_neut_sg else "ας"
            return _apply_recessive_with_breathing(f"{astem}{end}", lemma)
        if voice in {"M", "P"}:
            vow = "η" if (is_contract_e and tense != "P") else "ο"
            part_stem = (
                f"{_sigmatic_stem(stem)}αμεν"
                if tense == "A"
                else f"{stem}{'ουμεν' if is_contract_e else vow + 'μεν'}"
            )
            adj = _regular_noun_inflect(
                f"{part_stem}ος"
                if gnd == "M"
                else f"{part_stem}ον"
                if gnd == "N"
                else f"{part_stem}η",
                case=case,
                number=number,
                gender=gnd,
            )
            return adj
        if gnd == "F":
            fem_raw = (
                f"{stem}οῦσα"
                if is_contract_e
                else f"{stem}ῶσα"
                if is_contract_a
                else f"{stem}ουσα"
            )
            fem_nom = (
                _preserve_initial_breathing(fem_raw, lemma)
                if (is_contract_e or is_contract_a)
                else _apply_recessive_with_breathing(fem_raw, lemma)
            )
            return (
                _regular_noun_inflect(fem_nom, case=case, number=number, gender="F")
                or fem_nom
            )
        if is_contract_e:
            ce_map = {
                ("S", "M", "N"): "ῶν",
                ("S", "M", "G"): "οῦντος",
                ("S", "M", "D"): "οῦντι",
                ("S", "M", "A"): "οῦντα",
                ("S", "M", "V"): "ῶν",
                ("S", "N", "N"): "οῦν",
                ("S", "N", "G"): "οῦντος",
                ("S", "N", "D"): "οῦντι",
                ("S", "N", "A"): "οῦν",
                ("S", "N", "V"): "οῦν",
                ("P", "M", "N"): "οῦντες",
                ("P", "M", "G"): "ούντων",
                ("P", "M", "D"): "οῦσι",
                ("P", "M", "A"): "οῦντας",
                ("P", "M", "V"): "οῦντες",
                ("P", "N", "N"): "οῦντα",
                ("P", "N", "G"): "ούντων",
                ("P", "N", "D"): "οῦσι",
                ("P", "N", "A"): "οῦντα",
                ("P", "N", "V"): "οῦντα",
            }
            return _preserve_initial_breathing(
                stem + ce_map.get((number, gnd, case), "ῶν"), lemma
            )
        if is_contract_a:
            ca_map = {
                ("S", "M", "N"): "ῶν",
                ("S", "M", "G"): "ῶντος",
                ("S", "M", "D"): "ῶντι",
                ("S", "M", "A"): "ῶντα",
                ("S", "M", "V"): "ῶν",
                ("S", "N", "N"): "ῶν",
                ("S", "N", "G"): "ῶντος",
                ("S", "N", "D"): "ῶντι",
                ("S", "N", "A"): "ῶν",
                ("S", "N", "V"): "ῶν",
                ("P", "M", "N"): "ῶντες",
                ("P", "M", "G"): "ώντων",
                ("P", "M", "D"): "ῶσι",
                ("P", "M", "A"): "ῶντας",
                ("P", "M", "V"): "ῶντες",
                ("P", "N", "N"): "ῶντα",
                ("P", "N", "G"): "ώντων",
                ("P", "N", "D"): "ῶσι",
                ("P", "N", "A"): "ῶντα",
                ("P", "N", "V"): "ῶντα",
            }
            return _preserve_initial_breathing(
                stem + ca_map.get((number, gnd, case), "ῶν"), lemma
            )
        reg_act_map = {
            ("S", "M", "N"): "ων",
            ("S", "M", "G"): "οντος",
            ("S", "M", "D"): "οντι",
            ("S", "M", "A"): "οντα",
            ("S", "M", "V"): "ων",
            ("S", "N", "N"): "ον",
            ("S", "N", "G"): "οντος",
            ("S", "N", "D"): "οντι",
            ("S", "N", "A"): "ον",
            ("S", "N", "V"): "ον",
            ("P", "M", "N"): "οντες",
            ("P", "M", "G"): "οντων",
            ("P", "M", "D"): "ουσι",
            ("P", "M", "A"): "οντας",
            ("P", "M", "V"): "οντες",
            ("P", "N", "N"): "οντα",
            ("P", "N", "G"): "οντων",
            ("P", "N", "D"): "ουσι",
            ("P", "N", "A"): "οντα",
            ("P", "N", "V"): "οντα",
        }
        return _apply_recessive_with_breathing(
            stem + reg_act_map.get((number, gnd, case), "ων"), lemma
        )

    if mood == "N":
        if tense == "A" and voice == "P":
            return _preserve_initial_breathing(
                _aor_pass_stem(with_augment=False) + "ῆναι",
                supp.get("AP_UNAWG", lemma),
            )
        if tense == "A" and voice in {"A", "M"} and "A2_STEM" in supp:
            return _nfc(supp["A2_STEM"] + "εῖν")
        if tense == "P" and voice == "A":
            end_inf = "ᾶν" if is_contract_a else "εῖν" if is_contract_e else "ειν"
            if is_contract_e or is_contract_a:
                return _preserve_initial_breathing(stem + end_inf, lemma)
            return _apply_recessive_with_breathing(stem + end_inf, lemma)
        if tense == "A" and voice == "A":
            return _apply_recessive_with_breathing(_sigmatic_stem(stem) + "αι", lemma)
        if voice in {"M", "P"}:
            return _apply_recessive_with_breathing(
                f"{_sigmatic_stem(stem)}ασθαι" if tense == "A" else f"{stem}εσθαι",
                lemma,
            )

    if mood == "D":
        if voice == "A":
            if tense == "A":
                if number == "S" and "A2_IMP_S" in supp:
                    return supp["A2_IMP_S"]
                if "A2_STEM" in supp:
                    a2 = supp["A2_STEM"]
                    return _apply_recessive_with_breathing(
                        f"{a2}{'ε' if number == 'S' else 'ετε'}", lemma
                    )
                ast = _sigmatic_stem(stem)
                return _apply_recessive_with_breathing(
                    f"{ast}{'ον' if number == 'S' else 'ατε'}", lemma
                )
            if is_contract_a:
                end_d = "α" if number == "S" else "ᾶτε"
                return _preserve_initial_breathing(stem + end_d, lemma)
            if is_contract_e:
                end_d = "ει" if number == "S" else "εῖτε"
                return _preserve_initial_breathing(stem + end_d, lemma)
            return _apply_recessive_with_breathing(
                f"{stem}{'ε' if number == 'S' else 'ετε'}",
                lemma,
            )
        if is_contract_a:
            return _preserve_initial_breathing(
                stem + ("ῶ" if number == "S" else "ᾶσθε"), lemma
            )
        if is_contract_e:
            return _preserve_initial_breathing(
                stem + ("οῦ" if number == "S" else "εῖσθε"), lemma
            )
        return _apply_recessive_with_breathing(
            f"{stem}{'ου' if number == 'S' else 'εσθε'}", lemma
        )

    if mood == "S":
        if voice == "P":
            pas_subj = {
                ("1", "S"): "ῶ",
                ("2", "S"): "ῇς",
                ("3", "S"): "ῇ",
                ("1", "P"): "ῶμεν",
                ("2", "P"): "ῆτε",
                ("3", "P"): "ῶσι",
            }
            return _preserve_initial_breathing(
                _aor_pass_stem(with_augment=False) + pas_subj[(person, number)],
                supp.get("AP_UNAWG", lemma),
            )
        endings = {
            ("1", "S"): "ω",
            ("2", "S"): "ῃς",
            ("3", "S"): "ῃ",
            ("1", "P"): "ωμεν",
            ("2", "P"): "ητε",
            ("3", "P"): "ωσι",
        }
        mid_end = {
            ("1", "S"): "ωμαι",
            ("2", "S"): "ῃ",
            ("3", "S"): "ηται",
            ("1", "P"): "ωμεθα",
            ("2", "P"): "ησθε",
            ("3", "P"): "ωνται",
        }
        vstem = (
            supp["A2_STEM"]
            if (tense == "A" and "A2_STEM" in supp)
            else _sigmatic_stem(stem)
            if tense == "A"
            else stem
        )
        table = endings if voice == "A" else mid_end
        return _apply_recessive_with_breathing(vstem + table[(person, number)], lemma)

    # Indicative (P, I, F, A, X)
    if tense == "P" and voice == "A":
        pres = {
            ("1", "S"): "ῶ" if (is_contract_e or is_contract_a) else "ω",
            ("2", "S"): "ᾷς" if is_contract_a else "εῖς" if is_contract_e else "εις",
            ("3", "S"): "ᾷ" if is_contract_a else "εῖ" if is_contract_e else "ει",
            ("1", "P"): "ῶμεν"
            if is_contract_a
            else "οῦμεν"
            if is_contract_e
            else "ομεν",
            ("2", "P"): "ᾶτε" if is_contract_a else "εῖτε" if is_contract_e else "ετε",
            ("3", "P"): "ῶσι" if is_contract_a else "οῦσι" if is_contract_e else "ουσι",
        }
        if is_contract_e or is_contract_a:
            return _preserve_initial_breathing(stem + pres[(person, number)], lemma)
        return _apply_recessive_with_breathing(stem + pres[(person, number)], lemma)

    if tense == "F" and voice == "A":
        if "FM" in supp and "FA" not in supp:
            # Deponent future (e.g. ὄψομαι, φάγομαι, γνώσομαι)
            mid_f = {
                ("1", "S"): "ομαι",
                ("2", "S"): "ῃ",
                ("3", "S"): "εται",
                ("1", "P"): "ομεθα",
                ("2", "P"): "εσθε",
                ("3", "P"): "ονται",
            }
            return _apply_recessive_with_breathing(
                supp["FM"] + mid_f[(person, number)], lemma
            )
        fst = supp.get("FA") or _sigmatic_stem(stem)
        is_liq = fst.endswith(("λ", "ρ", "μ", "ν"))
        if is_liq:
            fut_l = {
                ("1", "S"): "ῶ",
                ("2", "S"): "εῖς",
                ("3", "S"): "εῖ",
                ("1", "P"): "οῦμεν",
                ("2", "P"): "εῖτε",
                ("3", "P"): "οῦσι",
            }
            return _preserve_initial_breathing(fst + fut_l[(person, number)], lemma)
        fut = {
            ("1", "S"): "ω",
            ("2", "S"): "εις",
            ("3", "S"): "ει",
            ("1", "P"): "ομεν",
            ("2", "P"): "ετε",
            ("3", "P"): "ουσι",
        }
        return _apply_recessive_with_breathing(fst + fut[(person, number)], lemma)

    if voice == "P" and tense in {"F", "A"}:
        if tense == "F":
            fp_end = {
                ("1", "S"): "ήσομαι",
                ("2", "S"): "ήσῃ",
                ("3", "S"): "ήσεται",
                ("1", "P"): "ησόμεθα",
                ("2", "P"): "ήσεσθε",
                ("3", "P"): "ήσονται",
            }
            return _preserve_initial_breathing(
                _aor_pass_stem(with_augment=False) + fp_end[(person, number)],
                supp.get("AP_UNAWG", lemma),
            )
        ap_end = {
            ("1", "S"): "ην",
            ("2", "S"): "ης",
            ("3", "S"): "η",
            ("1", "P"): "ημεν",
            ("2", "P"): "ητε",
            ("3", "P"): "ησαν",
        }
        return _apply_recessive_with_breathing(
            _aor_pass_stem(with_augment=True) + ap_end[(person, number)],
            supp.get("AP", lemma),
        )

    if tense == "F" and voice in {"M", "P"}:
        fst_m = supp.get("FM") or _sigmatic_stem(stem)
        mid_f = {
            ("1", "S"): "ομαι",
            ("2", "S"): "ῃ",
            ("3", "S"): "εται",
            ("1", "P"): "ομεθα",
            ("2", "P"): "εσθε",
            ("3", "P"): "ονται",
        }
        return _apply_recessive_with_breathing(fst_m + mid_f[(person, number)], lemma)

    if tense in {"A", "I"} and voice == "A":
        if tense == "A" and "A2A" in supp:
            impf2 = {
                ("1", "S"): "ον",
                ("2", "S"): "ες",
                ("3", "S"): "ε",
                ("1", "P"): "ομεν",
                ("2", "P"): "ετε",
                ("3", "P"): "ον",
            }
            return _apply_recessive_with_breathing(
                supp["A2A"] + impf2[(person, number)], lemma
            )
        aug = _augment_verb_stem(lemma)
        if tense == "A":
            ast = supp.get("AA") or _sigmatic_stem(aug, for_aug=True)
            aor = {
                ("1", "S"): "α",
                ("2", "S"): "ας",
                ("3", "S"): "ε",
                ("1", "P"): "αμεν",
                ("2", "P"): "ατε",
                ("3", "P"): "αν",
            }
            return _apply_recessive_with_breathing(ast + aor[(person, number)], lemma)
        impf = {
            ("1", "S"): "ον",
            ("2", "S"): "ες",
            ("3", "S"): "ε",
            ("1", "P"): "ομεν",
            ("2", "P"): "ετε",
            ("3", "P"): "ον",
        }
        return _apply_recessive_with_breathing(aug + impf[(person, number)], lemma)

    if voice in {"M", "P"} and tense == "P":
        mp = {
            ("1", "S"): "ομαι",
            ("2", "S"): "ῃ",
            ("3", "S"): "εται",
            ("1", "P"): "ομεθα",
            ("2", "P"): "εσθε",
            ("3", "P"): "ονται",
        }
        return _apply_recessive_with_breathing(stem + mp[(person, number)], lemma)

    return None


def realize_greek_tokens(surfaces: list[str]) -> str:
    """Apply Koine phonological sandhi and punctuation spacing to token list."""
    if not surfaces:
        return ""
    raw_tokens = [s for s in surfaces if s]
    out: list[str] = []
    for idx, tok in enumerate(raw_tokens):
        nxt = raw_tokens[idx + 1] if idx + 1 < len(raw_tokens) else ""
        low = _nfc(tok).lower()
        nxt_low = _nfc(nxt).lower() if nxt else ""

        # Check next token initial breathing if vowel
        nxt_debr = (
            syllabify.debreath(characters.strip_accents(nxt_low)) if nxt_low else ""
        )
        nxt_starts_rough = nxt_debr.startswith("h")
        nxt_bare0 = (
            characters.strip_accents(characters.strip_breathing(nxt_low[:1]))
            if nxt_low
            else ""
        )
        nxt_is_vowel = nxt_bare0 in {"α", "ε", "η", "ι", "ο", "υ", "ω"}

        cur = tok
        if low in {"οὐ", "οὐκ", "οὐχ"}:
            if not nxt or nxt in _PUNCT_CLOSE:
                cur = "οὔ" if tok[0].islower() else "Οὔ"
            elif nxt_is_vowel:
                cur = (
                    ("οὐχ" if nxt_starts_rough else "οὐκ")
                    if tok[0].islower()
                    else ("Οὐχ" if nxt_starts_rough else "Οὐκ")
                )
            else:
                cur = "οὐ" if tok[0].islower() else "Οὐ"
        elif low in {"ἐκ", "ἐξ"} and nxt_is_vowel:
            cur = "ἐξ" if tok[0].islower() else "Ἐξ"
        elif low == "ἐξ" and nxt and not nxt_is_vowel and nxt not in _PUNCT_CLOSE:
            cur = "ἐκ" if tok[0].islower() else "Ἐκ"
        elif _strip_diacritics_keep_base(tok).endswith(
            ("ουσι", "ασι", "ωσι", "ειπε", "ελεγε", "εστι", "εσι")
        ) and (not nxt or nxt in _PUNCT_CLOSE or nxt_is_vowel):
            cur = tok + "ν"

        # Oxytone acute -> grave before non-enclitic word (matches SBLGNT orthography)
        if nxt and nxt not in _PUNCT_CLOSE and nxt_low not in _ENCLITIC_LOWER:
            try:
                if accentuation.get_accent_type(cur) == accentuation.OXYTONE:
                    nfd = unicodedata.normalize("NFD", cur)
                    idx_a = nfd.rfind("\u0301")
                    if idx_a >= 0:
                        cur = _nfc(nfd[:idx_a] + "\u0300" + nfd[idx_a + 1 :])
            except Exception:
                pass

        out.append(cur)

    # Render with punctuation attachment (no space before closing punct / after opening)
    rendered: list[str] = []
    for tok in out:
        if not rendered:
            rendered.append(tok)
        elif (
            tok in _PUNCT_CLOSE
            or rendered[-1] in _PUNCT_OPEN
            or rendered[-1].endswith(_SMOOTH_APOSTROPHE)
        ):
            rendered[-1] = rendered[-1] + tok
        else:
            rendered.append(" " + tok)
    return "".join(rendered)


def _guess_gender_from_suffix(lemma: str) -> str | None:
    norm_lem = _nfc(lemma).lower()
    if norm_lem in _IRREGULAR_GENDERS:
        return _IRREGULAR_GENDERS[norm_lem]
    bare = characters.strip_accents(characters.strip_breathing(lemma)).lower()
    if bare.endswith(("ία", "εία", "σις", "τις", "της", "ότης", "ύτης", "η")):
        return "F"
    if bare.endswith(("ον", "μα", "ῖον", "εῖον")):
        return "N"
    if bare.endswith(("ος", "ης", "ας", "εύς", "τήρ", "τωρ")):
        return "M"
    return None


class GreekInflector:
    """Empirical + rule-based Koine Greek paradigm generator and surface analyzer."""

    def __init__(
        self,
        *,
        paradigms: dict[tuple[str, str], str],
        coarse_paradigms: dict[tuple[str, str], str],
        lemma_genders: dict[str, str],
        surface_analyses: dict[str, list[tuple[str, str, float]]],
        bare_surface_analyses: dict[str, list[tuple[str, str, float]]] | None = None,
        known_lemmas: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._paradigms = paradigms
        self._coarse = coarse_paradigms
        self._genders = lemma_genders
        self._surface_analyses = surface_analyses
        if bare_surface_analyses is not None:
            self._bare_analyses = bare_surface_analyses
        else:
            bare_acc: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
            for surf, entries in surface_analyses.items():
                bk = _bare_greek_key(surf)
                for lem, sig, lp in entries:
                    prev = bare_acc[bk].get((lem, sig))
                    if prev is None or lp > prev:
                        bare_acc[bk][(lem, sig)] = lp
            self._bare_analyses = {
                bk: sorted(
                    ((lem, sig, lp) for (lem, sig), lp in pairs.items()),
                    key=lambda item: (-item[2], item[0]),
                )[:3]
                for bk, pairs in bare_acc.items()
            }
        kl: dict[str, tuple[str, str]] = {}
        for entries in surface_analyses.values():
            for lem, sig, _ in entries:
                pos = sig.split(":")[0] if ":" in sig else sig.split("|")[0]
                bk = _bare_greek_key(lem)
                if bk and bk not in kl:
                    kl[bk] = (_nfc(lem), pos)
        if known_lemmas:
            kl.update(known_lemmas)
        self._known_lemmas = kl

    @classmethod
    def build(
        cls,
        *,
        morphgnt_dir: Path,
        exclude_unit_ids: set[str] | None = None,
        ccl_lexicon: CclLexicon | None = None,
        swete_dir: Path | None = None,
    ) -> GreekInflector:
        """Build paradigm and analyzer tables from MorphGNT (+ CCL/Swete LXX)."""
        excluded = set(exclude_unit_ids or ())
        units = load_morphgnt_nt(morphgnt_dir)

        exact_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        coarse_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        gender_counts: dict[str, Counter[str]] = defaultdict(Counter)
        surf_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        known_lemmas: dict[str, tuple[str, str]] = {}
        lemma_freq: Counter[tuple[str, str]] = Counter()

        for unit_id, unit in sorted(units.items()):
            if unit_id in excluded:
                continue
            for tok in unit.tokens:
                surf = _normalize_polytonic_for_match(tok.surface)
                lemma = _nfc(tok.lemma)
                sig = tok.morph.signature()
                exact_counts[(lemma, sig)][surf] += 1
                lemma_freq[(lemma, tok.morph.pos)] += 1

                # Relaxed signature dropping degree (and gender if V/RA/RP)
                c_sig = "|".join(
                    [
                        tok.morph.pos,
                        tok.morph.person or "-",
                        tok.morph.tense or "-",
                        tok.morph.voice or "-",
                        tok.morph.mood or "-",
                        tok.morph.case or "-",
                        tok.morph.number or "-",
                        tok.morph.gender or "-" if tok.morph.pos != "V-" else "-",
                    ]
                )
                coarse_counts[(lemma, c_sig)][surf] += 1
                if tok.morph.pos == "N-" and tok.morph.gender:
                    gender_counts[lemma][tok.morph.gender] += 1
                surf_counts[surf][(lemma, sig)] += 1

        for (lem, pos), _ in lemma_freq.most_common():
            bk = _bare_greek_key(lem)
            if bk and bk not in known_lemmas:
                known_lemmas[bk] = (lem, pos)

        paradigms = {k: cnt.most_common(1)[0][0] for k, cnt in exact_counts.items()}
        coarse_paradigms = {
            k: cnt.most_common(1)[0][0] for k, cnt in coarse_counts.items()
        }
        lemma_genders = {
            lemma: cnt.most_common(1)[0][0] for lemma, cnt in gender_counts.items()
        }

        if ccl_lexicon is not None:
            for entries in ccl_lexicon._loanwords.values():
                for e in entries:
                    lem = _nfc(e.greek_lemma)
                    bk = _bare_greek_key(lem)
                    if bk and bk not in known_lemmas:
                        known_lemmas[bk] = (lem, e.greek_pos)
            for lem, gnd in ccl_lexicon._greek_genders.items():
                nlem = _nfc(lem)
                if nlem not in lemma_genders:
                    lemma_genders[nlem] = gnd

        surface_analyses: dict[str, list[tuple[str, str, float]]] = {}
        bare_acc: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        for surf, cnt in surf_counts.items():
            tot = max(sum(cnt.values()), 1)
            surface_analyses[surf] = [
                (lem, sig, round(math.log((c + 0.1) / (tot + 0.5)), 4))
                for (lem, sig), c in cnt.most_common(3)
            ]
            bk = _bare_greek_key(surf)
            for pair, c in cnt.items():
                bare_acc[bk][pair] += c

        bare_surface_analyses: dict[str, list[tuple[str, str, float]]] = {}
        for bk, cnt in bare_acc.items():
            tot = max(sum(cnt.values()), 1)
            bare_surface_analyses[bk] = [
                (lem, sig, round(math.log((c + 0.1) / (tot + 0.5)), 4))
                for (lem, sig), c in cnt.most_common(3)
            ]

        if swete_dir is not None and swete_dir.is_dir():
            data_dir = (
                swete_dir / "data" if (swete_dir / "data").is_dir() else swete_dir
            )
            swete_word_counts: Counter[str] = Counter()
            for p in sorted(data_dir.glob("*.txt")):
                parsed = parse_swete_file(p.read_text(encoding="utf-8"))
                for words in parsed.values():
                    for w in words:
                        if w and not w.isupper() and any(ch.isalpha() for ch in w):
                            swete_word_counts[w] += 1

            base_inf = cls(
                paradigms=paradigms,
                coarse_paradigms=coarse_paradigms,
                lemma_genders=lemma_genders,
                surface_analyses=surface_analyses,
                bare_surface_analyses=bare_surface_analyses,
                known_lemmas=known_lemmas,
            )
            sw_exact: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
            sw_coarse: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
            for word, cnt in swete_word_counts.items():
                if cnt < 3:
                    continue
                parses = base_inf.analyze_surface(word)
                lem, mf, lp = parses[0]
                if mf.pos not in {"N-", "A-", "V-"} or lp <= -4.5:
                    continue
                surf = _normalize_polytonic_for_match(word)
                sig = mf.signature()
                c_sig = "|".join(
                    [
                        mf.pos,
                        mf.person or "-",
                        mf.tense or "-",
                        mf.voice or "-",
                        mf.mood or "-",
                        mf.case or "-",
                        mf.number or "-",
                        mf.gender or "-" if mf.pos != "V-" else "-",
                    ]
                )
                if mf.case != "D" and lp >= -2.6:
                    sw_exact[(lem, sig)][surf] += cnt
                    sw_coarse[(lem, c_sig)][surf] += cnt
                if surf not in surface_analyses:
                    surf_counts[surf][(lem, sig)] += cnt

            for k, cnt in sw_exact.items():
                if k not in paradigms and sum(cnt.values()) >= 3:
                    paradigms[k] = cnt.most_common(1)[0][0]
            for k, cnt in sw_coarse.items():
                if k not in coarse_paradigms and sum(cnt.values()) >= 3:
                    coarse_paradigms[k] = cnt.most_common(1)[0][0]
            for surf, cnt in surf_counts.items():
                if surf not in surface_analyses:
                    tot = max(sum(cnt.values()), 1)
                    surface_analyses[surf] = [
                        (lem, sig, round(math.log((c + 0.1) / (tot + 0.5)), 4))
                        for (lem, sig), c in cnt.most_common(3)
                    ]
                    bk = _bare_greek_key(surf)
                    if bk not in bare_surface_analyses:
                        bare_surface_analyses[bk] = surface_analyses[surf]

        return cls(
            paradigms=paradigms,
            coarse_paradigms=coarse_paradigms,
            lemma_genders=lemma_genders,
            surface_analyses=surface_analyses,
            bare_surface_analyses=bare_surface_analyses,
            known_lemmas=known_lemmas,
        )

    def lemma_gender(self, lemma: str) -> str | None:
        """Return attested grammatical gender ('M'/'F'/'N') or morphological guess."""
        norm = _nfc(lemma)
        if norm in self._genders:
            return self._genders[norm]
        return _guess_gender_from_suffix(norm)

    def _stem_suffix_parse(
        self, bare: str, orig_norm: str, *, verb_only: bool = False
    ) -> list[tuple[str, str, float]]:
        """Analyze an OOV surface by stripping regular Koine inflectional suffixes."""
        for suf, nom_ends, sig, allow_oov in _VERB_SUFFIX_RULES:
            if not bare.endswith(suf):
                continue
            stem = bare[: -len(suf)]
            if len(stem) < 2:
                continue
            cand_stems = [stem]
            if len(stem) >= 4 and stem.startswith("ε"):
                cand_stems.append(stem[1:])
            if suf.startswith(("μεν", "ημεν")):
                extra_stems: list[str] = []
                for base_s in list(cand_stems):
                    if (
                        len(base_s) >= 5
                        and base_s[1] == "ε"
                        and (
                            base_s[0] == base_s[2]
                            or (base_s[0], base_s[2])
                            in {("τ", "θ"), ("θ", "θ"), ("π", "φ"), ("κ", "χ")}
                        )
                    ):
                        extra_stems.append(base_s[2:])
                for bs in extra_stems:
                    if bs not in cand_stems:
                        cand_stems.append(bs)
                deassim: list[str] = []
                for bs in cand_stems:
                    if bs.endswith("μ"):
                        for repl in ("πτ", "π", "β", "φ"):
                            deassim.append(bs[:-1] + repl)
                    elif bs.endswith("γ"):
                        for repl in ("ζ", "γ", "κ", "χ"):
                            deassim.append(bs[:-1] + repl)
                    elif bs.endswith("σ"):
                        for repl in ("ζ", "δ", "τ", "θ"):
                            deassim.append(bs[:-1] + repl)
                for ds in deassim:
                    if ds not in cand_stems:
                        cand_stems.append(ds)
            for cs in cand_stems:
                for ne in nom_ends:
                    hit = self._known_lemmas.get(cs + ne)
                    if hit is not None and hit[1] == "V-":
                        return [(hit[0], sig, -2.2 if allow_oov else -2.7)]
            if allow_oov and len(stem) >= 3:
                base_stem = (
                    stem[1:] if (len(stem) >= 4 and stem.startswith("ε")) else stem
                )
                synth = _apply_recessive_with_breathing(
                    base_stem + nom_ends[0], orig_norm
                )
                return [(synth, sig, -3.6)]

        if verb_only:
            return []

        for suf, nom_ends, sig, allow_oov in _NOUN_SUFFIX_RULES:
            if not bare.endswith(suf):
                continue
            stem = bare[: -len(suf)]
            if len(stem) < (1 if not allow_oov else 2):
                continue
            for ne in nom_ends:
                hit = self._known_lemmas.get(stem + ne)
                if hit is not None and hit[1] in {"N-", "A-"}:
                    adj_sig = hit[1] + sig[2:] if ":" in sig else sig
                    if hit[1] == "N-":
                        gnd = self._genders.get(hit[0]) or _guess_gender_from_suffix(
                            hit[0]
                        )
                        if gnd:
                            adj_sig = (
                                self.parse_signature(adj_sig)
                                .model_copy(update={"gender": gnd})
                                .signature()
                            )
                    return [(hit[0], adj_sig, -2.1 if allow_oov else -2.7)]
            if allow_oov and len(stem) >= 3:
                synth = _apply_recessive_with_breathing(stem + nom_ends[0], orig_norm)
                return [(synth, sig, -3.6)]
        return []

    def _preverb_decompose(
        self, bare: str, orig_norm: str
    ) -> list[tuple[str, str, float]]:
        """Decompose compound Koine verbs across preverb + augment boundaries."""
        known_out: list[tuple[str, str, float]] = []
        synth_out: list[tuple[str, str, float]] = []
        for pref, canon_prev, may_aug in _PREVERB_ANALYSIS_RULES:
            if not bare.startswith(pref):
                continue
            rest = bare[len(pref) :]
            if len(rest) < 3:
                continue
            rest_cands = [f"ε{rest}", rest] if may_aug else [rest]
            matched = False
            for rc in rest_cands:
                for rc_var in (rc, rc[:-1] if rc.endswith(("σιν", "εν")) else ""):
                    if not rc_var:
                        continue
                    hits = self._bare_analyses.get(rc_var)
                    if not hits and rc_var in _ELIDED_BARE_PARSE:
                        eb_lem, eb_sig = _ELIDED_BARE_PARSE[rc_var]
                        hits = [(eb_lem, eb_sig, -2.8)]
                    if not hits:
                        continue
                    for s_lem, s_sig, lp in hits:
                        if not s_sig.startswith("V-:"):
                            continue
                        comp = _compose_preverb_lemma(canon_prev, s_lem)
                        known = self._known_lemmas.get(_bare_greek_key(comp))
                        if known is not None and known[1] == "V-":
                            known_out.append((known[0], s_sig, max(lp - 1.0, -3.5)))
                        else:
                            synth_out.append((comp, s_sig, max(lp - 1.5, -4.0)))
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                for s_lem, s_sig, lp in self._stem_suffix_parse(
                    rest, rest, verb_only=True
                ):
                    comp = _compose_preverb_lemma(canon_prev, s_lem)
                    known = self._known_lemmas.get(_bare_greek_key(comp))
                    if known is not None and known[1] == "V-":
                        known_out.append((known[0], s_sig, -2.5))
                    else:
                        synth_out.append((comp, s_sig, max(lp - 0.4, -4.1)))
        if known_out:
            known_out.sort(key=lambda item: -item[2])
            return known_out[:2]
        if synth_out:
            synth_out.sort(key=lambda item: -item[2])
            return synth_out[:2]
        return []

    def analyze_surface(
        self, surface: str
    ) -> list[tuple[str, GreekMorphFeatures, float]]:
        """Return candidate (lemma, GreekMorphFeatures, log_prob) parses."""
        norm = _normalize_polytonic_for_match(surface)
        variants = [norm]
        if norm.endswith(("ν", "νς")) and len(norm) > 3:
            if norm.endswith("σιν") or norm.endswith("εν"):
                variants.append(norm[:-1])
        else:
            if norm.endswith(("σι", "ε")) and len(norm) > 2:
                variants.append(norm + "ν")

        # 1. Exact polytonic surface match
        for cand in variants:
            hits = self._surface_analyses.get(cand)
            if hits:
                return [(lem, self.parse_signature(sig), lp) for lem, sig, lp in hits]

        # 2. Elided prepositions/conjunctions (ἐπ’, δ’, ἀπ’, ...)
        bare = _bare_greek_key(norm)
        if norm in _ELIDED_SURFACE_PARSE:
            lem, sig = _ELIDED_SURFACE_PARSE[norm]
            return [(lem, self.parse_signature(sig), -0.5)]
        if bare in _ELIDED_BARE_PARSE:
            lem, sig = _ELIDED_BARE_PARSE[bare]
            return [(lem, self.parse_signature(sig), -0.8)]

        # 3. Diacritic-stripped exact/all-caps/iota-subscript/diaeresis match
        bare_vars = [bare]
        if bare.endswith(("σιν", "εν")) and len(bare) > 3:
            bare_vars.append(bare[:-1])
        elif bare.endswith(("σι", "ε")) and len(bare) > 2:
            bare_vars.append(bare + "ν")
        for bc in bare_vars:
            bhits = self._bare_analyses.get(bc)
            if bhits:
                return [
                    (lem, self.parse_signature(sig), round(lp - 0.15, 4))
                    for lem, sig, lp in bhits
                ]

        # 4. Preverb + augment decomposition (e.g. ἐπεφέρετο -> ἐπιφέρω)
        if bare.isalpha() and len(bare) >= 6:
            pv_hits = self._preverb_decompose(bare, norm)
            if pv_hits:
                return [
                    (lem, self.parse_signature(sig), lp) for lem, sig, lp in pv_hits
                ]

        # 5. Regular Koine suffix stripping (e.g. ἀλλοφύλων -> ἀλλόφυλος)
        if bare.isalpha():
            suf_hits = self._stem_suffix_parse(bare, norm, verb_only=False)
            if suf_hits:
                return [
                    (lem, self.parse_signature(sig), lp) for lem, sig, lp in suf_hits
                ]

        return [(bare, GreekMorphFeatures(pos="UNK"), -5.0)]

    @staticmethod
    def parse_signature(sig: str) -> GreekMorphFeatures:
        """Reconstruct ``GreekMorphFeatures`` from canonical signature string."""
        sep = ":" if ":" in sig else "|"
        parts = sig.split(sep)
        pos = parts[0] if parts else "UNK"
        rest = parts[1] if len(parts) > 1 else "--------"
        return GreekMorphFeatures.from_morphgnt(pos, rest)

    def _lookup_paradigm(self, norm_lem: str, feat: GreekMorphFeatures) -> str | None:
        sig = feat.signature()
        exact = self._paradigms.get((norm_lem, sig))
        if exact is not None:
            return exact
        c_sig = "|".join(
            [
                feat.pos,
                feat.person or "-",
                feat.tense or "-",
                feat.voice or "-",
                feat.mood or "-",
                feat.case or "-",
                feat.number or "-",
                feat.gender or "-" if feat.pos != "V-" else "-",
            ]
        )
        return self._coarse.get((norm_lem, c_sig))

    def inflect(self, lemma: str, target: GreekMorphFeatures) -> str:
        """Inflect ``lemma`` to match ``target`` morphology (empirical -> rules)."""
        norm_lem = _nfc(lemma)

        def _cap(surf: str) -> str:
            if lemma and lemma[0].isupper() and surf and surf[0].isalpha():
                return unicodedata.normalize("NFC", surf[0].upper() + surf[1:])
            return surf

        # Normalize pronoun slots to match MorphGNT (person='-', 1st/2nd gender='-')
        eff_target = target
        if target.pos in {"RP", "RD", "RI", "RR", "RA"}:
            updates: dict[str, str | None] = {}
            if target.person is not None:
                updates["person"] = None
            if norm_lem in {"ἐγώ", "σύ"} and target.gender is not None:
                updates["gender"] = None
            elif norm_lem == "αὐτός" and target.gender is None:
                updates["gender"] = "M"
            if updates:
                eff_target = target.model_copy(update=updates)

        hit = self._lookup_paradigm(norm_lem, eff_target)
        if hit is not None:
            return _cap(hit)

        # Defective οἶδα (MorphGNT Perfect tense='X') and suppletive γινώσκω delegation
        if eff_target.pos == "V-" and norm_lem == "οἶδα":
            if eff_target.tense == "P" and eff_target.mood in {"I", "S", "N", "P"}:
                oid_hit = self._lookup_paradigm(
                    "οἶδα",
                    eff_target.model_copy(update={"tense": "X", "voice": "A"}),
                )
                if oid_hit is not None:
                    return _cap(oid_hit)
            return _cap(self.inflect("γινώσκω", eff_target))

        # Deponent / semi-deponent empirical voice fallback (e.g. ὄψεσθε, εἰσῆλθον)
        if eff_target.pos == "V-":
            alt_voices: tuple[str, ...] = ()
            if eff_target.voice == "A":
                alt_voices = ("M",)
            elif eff_target.voice == "M":
                alt_voices = ("A", "P")
            elif eff_target.voice == "P" and eff_target.mood == "P":
                alt_voices = ("A",)
            for v_alt in alt_voices:
                v_hit = self._lookup_paradigm(
                    norm_lem, eff_target.model_copy(update={"voice": v_alt})
                )
                if v_hit is not None:
                    return _cap(v_hit)

        # Non-indicative aspect fallback (swap Aorist <-> Present across valid voices)
        if eff_target.pos == "V-" and eff_target.mood in {"N", "D", "S"}:
            t_alt = "A" if eff_target.tense == "P" else "P"
            v_orig = eff_target.voice or "A"
            for v_try in (v_orig, "M" if v_orig == "A" else "A", "P"):
                asp_hit = self._lookup_paradigm(
                    norm_lem,
                    eff_target.model_copy(update={"tense": t_alt, "voice": v_try}),
                )
                if asp_hit is not None:
                    return _cap(asp_hit)

        # Fallback for active-only participles (e.g. ζάω, μένω, ἔρχομαι)
        if eff_target.pos == "V-" and eff_target.mood == "P":
            for t_alt, v_alt in (("P", "A"), ("X", "A"), ("A", "A")):
                alt_p = self._lookup_paradigm(
                    norm_lem,
                    eff_target.model_copy(update={"tense": t_alt, "voice": v_alt}),
                )
                if alt_p is not None:
                    return _cap(alt_p)

        # Neuter Nom <-> Acc syncretism
        if eff_target.gender == "N" and eff_target.case in {"N", "A"}:
            alt_case = "A" if eff_target.case == "N" else "N"
            alt_hit = self._lookup_paradigm(
                norm_lem, eff_target.model_copy(update={"case": alt_case})
            )
            if alt_hit is not None:
                return _cap(alt_hit)

        # Rule-based nominal/adjectival/participial/verbal inflection
        if eff_target.pos in {"N-", "A-"}:
            reg = _regular_noun_inflect(
                norm_lem,
                case=eff_target.case or "N",
                number=eff_target.number or "S",
                gender=eff_target.gender or self.lemma_gender(norm_lem),
            )
            if reg is not None:
                return _cap(reg)
        elif eff_target.pos == "V-":
            reg = _regular_verb_inflect(norm_lem, eff_target)
            if reg is not None:
                return _cap(reg)

        return _cap(norm_lem)

    def save_shards(self, out_dir: Path, *, max_bytes: int = 680_000) -> list[Path]:
        """Serialize compact model tables under ``out_dir``, each file < 1024 KiB."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("greek-paradigms-*.json"):
            old.unlink()
        files: list[Path] = []

        # Shard paradigms cleanly so every JSON stays strictly < max_bytes (< 1024 KiB)
        items = sorted(self._paradigms.items())
        shard: dict[str, dict[str, str]] = defaultdict(dict)
        shard_idx = 0
        est_size = 2
        for (lem, sig), surf in items:
            key_overhead = 0 if lem in shard else len(lem.encode("utf-8")) + 8
            entry_len = (
                key_overhead + len(sig.encode("utf-8")) + len(surf.encode("utf-8")) + 8
            )
            if est_size + entry_len > max_bytes and shard:
                p = out_dir / f"greek-paradigms-{shard_idx}.json"
                p.write_text(
                    json.dumps(shard, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                files.append(p)
                shard = defaultdict(dict)
                shard_idx += 1
                est_size = 2
                entry_len = (
                    len(lem.encode("utf-8"))
                    + len(sig.encode("utf-8"))
                    + len(surf.encode("utf-8"))
                    + 16
                )
            shard[lem][sig] = surf
            est_size += entry_len
        if shard:
            p = out_dir / f"greek-paradigms-{shard_idx}.json"
            p.write_text(
                json.dumps(shard, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            files.append(p)

        meta = {
            "genders": dict(sorted(self._genders.items())),
            "coarse_size": len(self._coarse),
        }
        meta_p = out_dir / "greek-meta.json"
        meta_p.write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        files.append(meta_p)
        return files

    @classmethod
    def load_shards(cls, out_dir: Path) -> GreekInflector:
        """Load empirical paradigm shards without needing MorphGNT downloads."""
        paradigms: dict[tuple[str, str], str] = {}
        coarse: dict[tuple[str, str], str] = {}
        for p in sorted(out_dir.glob("greek-paradigms-*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            for lem, sig_map in data.items():
                for sig, surf in sig_map.items():
                    paradigms[(lem, sig)] = surf
                    pos, _, feat_str = (
                        sig.partition(":") if ":" in sig else sig.partition("|")
                    )
                    if len(feat_str) >= 8:
                        g_ch = feat_str[6] if pos != "V-" else "-"
                        c_sig = f"{pos}|{'|'.join(feat_str[:6])}|{g_ch}"
                        coarse.setdefault((lem, c_sig), surf)
        meta_p = out_dir / "greek-meta.json"
        genders = (
            json.loads(meta_p.read_text(encoding="utf-8")).get("genders", {})
            if meta_p.is_file()
            else {}
        )
        return cls(
            paradigms=paradigms,
            coarse_paradigms=coarse,
            lemma_genders=genders,
            surface_analyses={},
        )
