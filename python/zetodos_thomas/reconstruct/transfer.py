# SPDX-License-Identifier: Apache-2.0
"""Coptic-to-Koine morphosyntactic transfer grammar and lexical slot builder."""

from __future__ import annotations

from dataclasses import dataclass, field

from zetodos_thomas.align.lemma_aligner import (
    LexicalTranslationModel,
    extract_coptic_right_context,
)
from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.lexicon.idioms import CopticIdiomRule, IdiomAndValencyTable
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.reconstruct.chunker import CopticClause, chunk_unit
from zetodos_thomas.schemas.linguistic import (
    CopticMorpheme,
    CopticUnit,
    GreekMorphFeatures,
)

# Canonical Coptic -> Greek function candidates (lemma, pos, score, fixed_surface)
_CONJ_MAP: dict[str, list[tuple[str, str, float, str | None]]] = {
    "ⲁⲩⲱ": [("καί", "C-", 2.0, "καί")],
    "ϫⲉ": [("ὅτι", "C-", 1.6, "ὅτι")],
    "ⲁⲗⲗⲁ": [("ἀλλά", "C-", 2.2, "ἀλλά")],
    "ϩⲟⲧⲁⲛ": [("ὅταν", "C-", 2.2, "ὅταν")],
    "ⲉϣⲱⲡⲉ": [("ἐάν", "C-", 2.1, "ἐάν")],
    "ⲉϣϫⲉ": [("εἰ", "C-", 2.1, "εἰ")],
    "ϩⲓⲛⲁ": [("ἵνα", "C-", 2.2, "ἵνα")],
    "ϫⲉⲕⲁⲥ": [("ἵνα", "C-", 2.2, "ἵνα")],
    "ⲉⲡⲉⲓ": [("ἐπεί", "C-", 2.0, "ἐπεί")],
    "ⲟⲩⲇⲉ": [("οὐδέ", "C-", 2.2, "οὐδέ")],
    "ⲟⲩⲧⲉ": [("οὔτε", "C-", 2.2, "οὔτε")],
    "ⲏ": [("ἤ", "C-", 2.1, "ἤ")],
    "ⲙⲏ": [("μή", "C-", 2.0, "μή")],
    "ⲙⲏⲡⲱⲥ": [("μήποτε", "C-", 2.0, "μήποτε")],
    "ϩⲱⲥ": [("ὡς", "C-", 2.0, "ὡς")],
    "ⲉⲓⲙⲏⲧⲓ": [("εἰ", "C-", 1.8, "εἰ")],
    "ⲡⲗⲏⲛ": [("πλήν", "C-", 2.0, "πλήν")],
    "ⲛⲧⲉⲣⲉ": [("ὅτε", "C-", 2.0, "ὅτε")],
    "ϣⲁⲛⲧⲉ": [("ἕως", "C-", 2.0, "ἕως")],
    "ⲇⲉ": [("δέ", "C-", 2.2, "δέ")],
    "ⲅⲁⲣ": [("γάρ", "C-", 2.2, "γάρ")],
    "ⲙⲉⲛ": [("μέν", "C-", 2.2, "μέν")],
    "ⲟⲩⲛ": [("οὖν", "C-", 2.0, "οὖν")],
    "ϭⲉ": [("οὖν", "C-", 1.9, "οὖν")],
    "ⲉⲓⲥ": [("ἰδού", "I-", 2.0, "ἰδού")],
    "ⲱ": [("ὦ", "I-", 2.2, "ὦ")],
}

_PREP_MAP: dict[str, list[tuple[str, str, float, str]]] = {
    "ϩⲛ": [("ἐν", "P-", 2.2, "D"), ("ἐκ", "P-", 0.4, "G")],
    "ⲉ": [("εἰς", "P-", 1.8, "A"), ("πρός", "P-", 1.2, "A")],
    "ⲛⲁ": [("πρός", "P-", 1.4, "A")],
    "ⲙⲛ": [("μετά", "P-", 1.9, "G"), ("σύν", "P-", 0.9, "D")],
    "ϩⲓ": [("ἐπί", "P-", 1.8, "G"), ("ἐν", "P-", 0.8, "D")],
    "ϩⲁ": [("ὑπό", "P-", 1.8, "A"), ("περί", "P-", 0.8, "G")],
    "ⲉϫⲛ": [("ἐπί", "P-", 2.1, "G")],
    "ϩⲓϫⲛ": [("ἐπί", "P-", 2.1, "G")],
    "ϣⲁ": [("πρός", "P-", 1.7, "A"), ("ἕως", "P-", 1.2, "G")],
    "ⲉⲧⲃⲉ": [("διά", "P-", 2.1, "A")],
    "ϩⲓⲧⲛ": [("διά", "P-", 2.1, "G"), ("ὑπό", "P-", 1.0, "G")],
    "ⲛⲥⲁ": [("ὀπίσω", "P-", 1.7, "G"), ("κατά", "P-", 0.9, "A")],
    "ⲕⲁⲧⲁ": [("κατά", "P-", 2.2, "A")],
    "ⲭⲱⲣⲓⲥ": [("χωρίς", "P-", 2.2, "G")],
    "ⲟⲩⲃⲉ": [("πρός", "P-", 1.6, "A")],
    "ⲛⲧⲛ": [("παρά", "P-", 1.8, "G")],
    "ϩⲁⲧⲛ": [("παρά", "P-", 1.8, "D")],
    "ⲙⲛⲛⲥⲁ": [("μετά", "P-", 2.1, "A")],
    "ⲛⲛⲁϩⲣⲛ": [("ἐνώπιον", "P-", 2.0, "G")],
    "ⲡⲁⲣⲁ": [("παρά", "P-", 2.0, "A")],
}

_POSSESSED_LOCATIVE_MAP: dict[str, str] = {
    "ϩⲟⲩⲛ": "ἐντός",
    "ⲃⲟⲗ": "ἐκτός",
    "ϩⲏ": "ἔμπροσθεν",
    "ⲡⲁϩⲟⲩ": "ὄπισθεν",
}

_DIRECT_OBJ_E_VERBS: frozenset[str] = frozenset(
    {
        "ⲛⲁⲩ",
        "ⲥⲱⲧⲙ",
        "ⲙⲟⲩⲧⲉ",
        "ⲥⲙⲟⲩ",
        "ⲥⲟⲟⲩⲛ",
        "ⲥⲟⲩⲱⲛ",
        "ⲙⲉ",
        "ⲙⲉⲣⲉ",
        "ⲙⲟⲥⲧⲉ",
    }
)

_PPERS_MAP: dict[str, tuple[str, str]] = {
    "ⲁⲛⲟⲕ": ("1", "S"),
    "ⲛⲧⲟⲕ": ("2", "S"),
    "ⲛⲧⲟ": ("2", "S"),
    "ⲛⲧⲟϥ": ("3", "S"),
    "ⲛⲧⲟⲥ": ("3", "S"),
    "ⲁⲛⲟⲛ": ("1", "P"),
    "ⲛⲧⲱⲧⲛ": ("2", "P"),
    "ⲛⲧⲟⲟⲩ": ("3", "P"),
    "ⲉⲣⲉ_ⲁⲛⲟⲕ": ("1", "S"),
    "ⲉⲣⲉ_ⲛⲧⲟⲕ": ("2", "S"),
    "ⲉⲣⲉ_ⲛⲧⲟϥ": ("3", "S"),
    "ⲉⲣⲉ_ⲛⲧⲟⲥ": ("3", "S"),
    "ⲉⲣⲉ_ⲁⲛⲟⲛ": ("1", "P"),
    "ⲉⲣⲉ_ⲛⲧⲱⲧⲛ": ("2", "P"),
    "ⲉⲣⲉ_ⲛⲧⲟⲟⲩ": ("3", "P"),
    "ⲛⲧⲉ_ⲁⲛⲟⲕ": ("1", "S"),
    "ⲉⲣϣⲁⲛ_ⲁⲛⲟⲕ": ("1", "S"),
    "ⲉⲣϣⲁⲛ_ⲛⲧⲟⲕ": ("2", "S"),
    "ⲉⲣϣⲁⲛ_ⲛⲧⲟ": ("2", "S"),
    "ⲉⲣϣⲁⲛ_ⲛⲧⲟϥ": ("3", "S"),
    "ⲉⲣϣⲁⲛ_ⲛⲧⲟⲥ": ("3", "S"),
    "ⲉⲣϣⲁⲛ_ⲁⲛⲟⲛ": ("1", "P"),
    "ⲉⲣϣⲁⲛ_ⲛⲧⲱⲧⲛ": ("2", "P"),
    "ⲉⲧⲉⲧⲛϣⲁⲛ_ⲛⲧⲱⲧⲛ": ("2", "P"),
    "ⲉⲣϣⲁⲛ_ⲛⲧⲟⲟⲩ": ("3", "P"),
    "ϯ": ("1", "S"),
    "ⲕ": ("2", "S"),
    "ⲧⲉ": ("2", "S"),
    "ϥ": ("3", "S"),
    "ⲥ": ("3", "S"),
    "ⲧⲛ": ("1", "P"),
    "ⲧⲉⲧⲛ": ("2", "P"),
    "ⲥⲉ": ("3", "P"),
}


def _ppers_person_num(m: CopticMorpheme) -> tuple[str, str]:
    """Resolve person/number from fused or plain subject prefix morpheme."""
    if m.norm in _PPERS_MAP:
        return _PPERS_MAP[m.norm]
    if m.norm == "ⲉⲧⲉⲧⲛϣⲁⲛ":
        return ("2", "P")
    if m.lemma in _PPERS_MAP:
        return _PPERS_MAP[m.lemma]
    if "_" in m.lemma:
        tail = m.lemma.rsplit("_", 1)[-1]
        if tail in _PPERS_MAP:
            return _PPERS_MAP[tail]
    return ("3", "S")


_PRONOUN_LEMMA: dict[str, tuple[str, str, str | None, str | None]] = {
    "ⲁⲛⲟⲕ": ("ἐγώ", "RP", "1", "S"),
    "ⲛⲧⲟⲕ": ("σύ", "RP", "2", "S"),
    "ⲛⲧⲟ": ("σύ", "RP", "2", "S"),
    "ⲛⲧⲟϥ": ("αὐτός", "RP", "3", "S"),
    "ⲛⲧⲟⲥ": ("αὐτός", "RP", "3", "S"),
    "ⲁⲛⲟⲛ": ("ἐγώ", "RP", "1", "P"),
    "ⲛⲧⲱⲧⲛ": ("σύ", "RP", "2", "P"),
    "ⲛⲧⲟⲟⲩ": ("αὐτός", "RP", "3", "P"),
    "ⲡⲁ": ("ἐγώ", "RP", "1", "S"),
    "ⲡⲉⲕ": ("σύ", "RP", "2", "S"),
    "ⲡⲟⲩ": ("σύ", "RP", "2", "S"),
    "ⲡⲉϥ": ("αὐτός", "RP", "3", "S"),
    "ⲡⲉⲥ": ("αὐτός", "RP", "3", "S"),
    "ⲡⲉⲛ": ("ἐγώ", "RP", "1", "P"),
    "ⲡⲉⲧⲛ": ("σύ", "RP", "2", "P"),
    "ⲡⲉⲩ": ("αὐτός", "RP", "3", "P"),
    "ⲡⲁⲓ": ("οὗτος", "RD", None, None),
    "ⲡⲏ": ("ἐκεῖνος", "RD", None, None),
    "ⲡⲓ": ("οὗτος", "RD", None, None),
    "ⲡⲉⲓ": ("οὗτος", "RD", None, None),
    "ⲛⲓⲙ": ("τίς", "RI", None, None),
    "ⲟⲩ": ("τίς", "RI", None, None),
    "ⲁϣ": ("τίς", "RI", None, None),
    "ⲗⲁⲁⲩ": ("τις", "RI", None, None),
    "ⲧⲏⲣ": ("πᾶς", "A-", None, None),
    "ⲕⲉ": ("ἄλλος", "A-", None, None),
    "ⲟⲩⲁ": ("εἷς", "A-", None, None),
    "ⲥⲛⲁⲩ": ("δύο", "A-", None, None),
    "ϣⲟⲙⲛⲧ": ("τρεῖς", "A-", None, None),
}

_PLURAL_NOUN_NORMS = frozenset({"ϩⲓⲟⲙⲉ", "ⲥⲛⲏⲩ", "ⲡⲏⲩⲉ", "ⲙⲁⲑⲏⲧⲏⲥ", "ϣⲁϫⲉ"})

_NUMERAL_MAP: dict[str, tuple[str, str]] = {
    "ⲟⲩⲁ": ("εἷς", "S"),
    "ⲥⲛⲁⲩ": ("δύο", "P"),
    "ϣⲟⲙⲛⲧ": ("τρεῖς", "P"),
    "ϣⲟⲙⲧ": ("τρεῖς", "P"),
    "ϥⲧⲟⲟⲩ": ("τέσσαρες", "P"),
    "ϯⲟⲩ": ("πέντε", "P"),
    "ⲥⲟⲟⲩ": ("ἕξ", "P"),
    "ⲥⲁϣϥ": ("ἑπτά", "P"),
    "ϣⲙⲟⲩⲛ": ("ὀκτώ", "P"),
    "ⲯⲓⲥ": ("ἐννέα", "P"),
    "ⲙⲏⲧ": ("δέκα", "P"),
    "ⲙⲛⲧⲥⲛⲟⲟⲩⲥ": ("δώδεκα", "P"),
    "ϣⲉ": ("ἑκατόν", "P"),
}

_LABILE_PASSIVE_LEMMAS: frozenset[str] = frozenset(
    {"ϭⲱⲗⲡ", "ⲟⲩⲱⲛϩ", "ϣⲧⲟⲣⲧⲣ", "ϩⲱⲡ", "ⲧⲁϫⲣⲟ", "ϫⲓⲥⲉ", "ⲑⲃⲃⲓⲟ", "ⲥⲱⲟⲩϩ"}
)


def _consume_discontinuous_particle(
    ms: list[CopticMorpheme],
    v_idx: int,
    idioms: IdiomAndValencyTable,
    suppress_seqs: set[int],
) -> CopticIdiomRule | None:
    """Fuse discontinuous ``V ... {ⲉⲃⲟⲗ,ⲉϩⲟⲩⲛ,ⲉϩⲣⲁⲓ,ⲉⲡⲉⲥⲏⲧ}`` idioms."""
    v_lem = ms[v_idx].lemma
    cand_rules = [
        r
        for r in idioms._by_first.get(v_lem, [])
        if r.greek_pos == "V-" and len(r.pattern_lemmas) == 2
    ]
    if not cand_rules:
        return None
    n = len(ms)
    for p_idx in range(v_idx + 1, min(n, v_idx + 6)):
        pm = ms[p_idx]
        if pm.pos in {"V", "VBD", "VIMP", "VSTAT", "CONJ", "CREL", "CCIRC", "PUNCT"}:
            break
        if pm.seq in suppress_seqs or pm.pos != "ADV":
            continue
        if pm.lemma not in {"ⲉⲃⲟⲗ", "ⲉϩⲟⲩⲛ", "ⲉϩⲣⲁⲓ", "ⲉⲡⲉⲥⲏⲧ"}:
            continue
        if (
            p_idx + 1 < n
            and ms[p_idx + 1].pos == "PREP"
            and ms[p_idx + 1].lemma in {"ϩⲛ", "ϩⲓⲧⲛ", "ⲉϫⲛ"}
        ):
            continue
        if pm.head_seq == ms[v_idx].seq or p_idx == v_idx + 1:
            for r in cand_rules:
                if r.pattern_lemmas == (v_lem, pm.lemma):
                    suppress_seqs.add(pm.seq)
                    return r
    return None


def _verb_has_direct_object(
    ms: list[CopticMorpheme], v_idx: int, suppress_seqs: set[int]
) -> bool:
    """Return True if verb ``ms[v_idx]`` governs a direct object in clause."""
    n = len(ms)
    if v_idx + 1 < n and ms[v_idx + 1].pos == "PPERO":
        return True
    for k in range(v_idx + 1, min(n, v_idx + 6)):
        mk = ms[k]
        if mk.pos in {"V", "VBD", "VIMP", "VSTAT", "CONJ", "CREL", "CCIRC", "PUNCT"}:
            break
        if mk.seq in suppress_seqs:
            continue
        if mk.dep_rel == "obj" or (mk.pos == "PREP" and mk.lemma in {"ⲛ", "ⲙⲙⲟ"}):
            return True
    return False


# Transliteration fallback for unseen biblical proper nouns (e.g. Boaz, Ruth)
_COPTIC_TO_GREEK_CHAR: dict[str, str] = {
    "ⲁ": "α",
    "ⲃ": "β",
    "ⲅ": "γ",
    "ⲇ": "δ",
    "ⲉ": "ε",
    "ⲍ": "ζ",
    "ⲏ": "η",
    "ⲑ": "θ",
    "ⲓ": "ι",
    "ⲕ": "κ",
    "ⲗ": "λ",
    "ⲙ": "μ",
    "ⲛ": "ν",
    "ⲝ": "ξ",
    "ⲟ": "ο",
    "ⲡ": "π",
    "ⲣ": "ρ",
    "ⲥ": "σ",
    "ⲧ": "τ",
    "ⲩ": "υ",
    "ⲫ": "φ",
    "ⲭ": "χ",
    "ⲯ": "ψ",
    "ⲱ": "ω",
    "ϣ": "σ",
    "ϥ": "φ",
    "ϩ": "",
    "ϫ": "ζ",
    "ϭ": "κ",
    "ϯ": "τι",
}


def _transliterate_coptic_name(coptic: str) -> str:
    chars = [_COPTIC_TO_GREEK_CHAR.get(ch, "") for ch in coptic.lower()]
    res = "".join(chars)
    if not res:
        res = "α"
    if res.endswith("σ"):
        res = res[:-1] + "ς"
    return res[0].upper() + res[1:]


@dataclass(frozen=True)
class SlotCandidate:
    """One ranked Greek lemma+POS candidate for a transferred slot."""

    lemma: str
    pos: str
    score: float
    provenance: str  # "L", "N", "P", "C"
    fixed_surface: str | None = None
    required_case: str | None = None
    morph_override: GreekMorphFeatures | None = None


@dataclass
class TransferSlot:
    """One Koine syntactic slot awaiting inflection & beam selection."""

    role: str
    coptic_seqs: list[int]
    target_morph: GreekMorphFeatures
    candidates: list[SlotCandidate] = field(default_factory=list)
    agree_head_offset: int | None = None  # +1 for pre-nominal ART/DEM
    case_governor: str = ""  # "verb_obj" if governed by main clause verb
    postpositive: bool = False
    is_punct: bool = False
    optional_omit: bool = False


def _det_number(det: CopticMorpheme | None, head: CopticMorpheme | None) -> str:
    if det is not None and det.norm in {
        "ⲛ",
        "ⲛⲉ",
        "ϩⲉⲛ",
        "ⲛⲓ",
        "ⲛⲉⲓ",
        "ⲛⲁ",
        "ⲛⲉⲕ",
        "ⲛⲟⲩ",
        "ⲛⲉϥ",
        "ⲛⲉⲥ",
        "ⲛⲉⲛ",
        "ⲛⲉⲧⲛ",
        "ⲛⲉⲩ",
    }:
        return "P"
    if det is not None and det.pos == "NUM" and det.lemma != "ⲟⲩⲁ":
        return "P"
    if (
        head is not None
        and head.norm in _PLURAL_NOUN_NORMS
        and det
        and det.norm.startswith("ⲛ")
    ):
        return "P"
    return "S"


def _det_gender_hint(det: CopticMorpheme | None) -> str:
    if det is not None and det.norm.startswith(("ⲧ", "ⲑ")):
        return "F"
    if det is not None and det.norm.startswith(("ⲛ", "ϩ")):
        return "M"
    return "M"


def _pronoun_person_num(
    m: CopticMorpheme, subj_person: str | None, subj_number: str | None
) -> tuple[str, str, str | None, str | None]:
    """Return (greek_lemma, pos, person, number) handling 3rd-person reflexives."""
    lem, pos, p, n = _PRONOUN_LEMMA.get(m.lemma, ("αὐτός", "RP", "3", "S"))
    # Reflexive: coreferential 3rd-person pronoun on object/possessive
    if (
        lem == "αὐτός"
        and p == "3"
        and subj_person == "3"
        and subj_number == n
        and m.pos == "PPERO"
    ):
        # Emit reflexive ἑαυτοῦ alongside αὐτός so beam + LM select cleanly
        return lem, pos, p, n
    return lem, pos, p, n


def _lookup_content_candidates(
    m: CopticMorpheme,
    *,
    desired_pos: str | None,
    lexicon: LexicalTranslationModel,
    ccl_lexicon: CclLexicon,
    prefer_ginosko: bool = False,
    right_ctx: str = "",
) -> list[SlotCandidate]:
    """Build ranked candidates with zero-null guaranteed fallback."""
    hits = lexicon.lookup(
        coptic_lemma=m.lemma,
        coptic_pos=m.pos,
        coptic_norm=m.norm,
        lang=m.lang,
        ccl_lexicon=ccl_lexicon,
        right_ctx=right_ctx,
    )
    out: list[SlotCandidate] = []
    for h in hits:
        sc = h.score
        if desired_pos and h.greek_pos == desired_pos:
            sc += 0.45
        elif desired_pos == "V-" and h.greek_pos != "V-":
            sc -= 1.2
        if prefer_ginosko and h.greek_lemma == "γινώσκω":
            sc += 0.65
        out.append(
            SlotCandidate(
                lemma=h.greek_lemma,
                pos=h.greek_pos,
                score=sc,
                provenance=h.provenance,
            )
        )

    if not out:
        # Proper noun or unseen Coptic stem fallback (zero null guarantee)
        if m.pos == "NPROP":
            name = _transliterate_coptic_name(m.lemma or m.norm)
            out.append(
                SlotCandidate(
                    lemma=name,
                    pos="N-",
                    score=-2.0,
                    provenance="C",
                    fixed_surface=name,
                )
            )
        elif m.pos in {"V", "VSTAT"}:
            out.append(
                SlotCandidate(lemma="ποιέω", pos="V-", score=-4.0, provenance="C")
            )
        elif m.pos == "ADV":
            out.append(
                SlotCandidate(lemma="οὕτως", pos="D-", score=-3.5, provenance="C")
            )
        else:
            out.append(
                SlotCandidate(lemma="λόγος", pos="N-", score=-4.5, provenance="C")
            )
    return sorted(out, key=lambda c: -c.score)[:6]


def _extract_clause_subject(clause: CopticClause) -> tuple[str | None, str | None]:
    """Detect clause finite verb person/number early for reflexive/copula agreement."""
    for ch in clause.chunks:
        if ch.kind == "PRED_V":
            for m in ch.morphemes:
                if m.pos in {"PPERS", "AOPT_PPERS", "ACONJ_PPERS", "ACOND_PPERS"}:
                    return _ppers_person_num(m)
        if ch.kind == "PP" and ch.sub_kind == "subj_marked":
            if any(m.norm in {"ⲛ", "ⲛⲉ", "ϩⲉⲛ"} for m in ch.morphemes):
                return "3", "P"
            return "3", "S"
    return "3", "S"


def _clause_to_slots(
    clause: CopticClause,
    *,
    lexicon: LexicalTranslationModel,
    ccl_lexicon: CclLexicon,
    idioms: IdiomAndValencyTable,
    inflector: GreekInflector,
    main_tense: str = "P",
    main_mood: str = "I",
    unit_has_2p: bool = False,
    emitted_by_seq: dict[int, TransferSlot] | None = None,
) -> tuple[list[TransferSlot], str, str]:
    """Convert one ``CopticClause`` into ordered ``TransferSlot`` list."""
    slots: list[TransferSlot] = []

    def _find_emitted(head_seq: int | None) -> TransferSlot | None:
        if head_seq is None:
            return None
        for sl in reversed(slots):
            if head_seq in sl.coptic_seqs and sl.role in {
                "NOUN",
                "PRON",
                "PART",
                "VERB",
                "ADJ",
            }:
                return sl
        if emitted_by_seq is not None:
            return emitted_by_seq.get(head_seq)
        return None

    ms = [
        (
            m
            if m.pos == "PUNCT"
            else m.model_copy(update={"lemma": normalize_coptic(m.lemma) or m.lemma})
        )
        for m in clause.morphemes
        if clause.postpositive is None or m.seq != clause.postpositive.seq
    ]
    n = len(ms)

    def _num_cands(num_m: CopticMorpheme, idx_n: int) -> list[SlotCandidate]:
        hit = _NUMERAL_MAP.get(num_m.lemma) or _NUMERAL_MAP.get(num_m.norm)
        cands = _lookup_content_candidates(
            num_m,
            desired_pos="A-",
            lexicon=lexicon,
            ccl_lexicon=ccl_lexicon,
            right_ctx=extract_coptic_right_context(ms, idx_n),
        )
        if hit is not None:
            out_n = [SlotCandidate(hit[0], "A-", 2.3, "C")]
            for c in cands:
                if c.lemma != hit[0]:
                    out_n.append(c)
            return out_n[:6]
        return cands

    def _refine_verb_cands_and_voice(
        v_m: CopticMorpheme,
        v_idx: int,
        base_cands: list[SlotCandidate],
        base_voice: str,
        cur_tense: str,
        *,
        obj_check_idx: int | None = None,
        skip_disc: bool = False,
    ) -> tuple[list[SlotCandidate], str]:
        cands = list(base_cands)
        disc_rule = (
            None
            if skip_disc
            else _consume_discontinuous_particle(ms, v_idx, idioms, suppress_seqs)
        )
        if disc_rule is not None:
            merged: list[SlotCandidate] = [
                SlotCandidate(
                    disc_rule.greek_lemma,
                    "V-",
                    2.55,
                    "N",
                    required_case=disc_rule.required_case,
                )
            ]
            for c in cands:
                if c.lemma != disc_rule.greek_lemma:
                    merged.append(c)
            cands = merged[:6]
        o_idx = v_idx if obj_check_idx is None else obj_check_idx
        is_labile_pass = v_m.pos != "VSTAT" and (
            v_m.lemma in _LABILE_PASSIVE_LEMMAS
            and not _verb_has_direct_object(ms, o_idx, suppress_seqs)
        )
        eff_voice = base_voice
        if is_labile_pass:
            eff_voice = "P"
            if v_m.lemma == "ϩⲱⲡ":
                boosted = [
                    SlotCandidate(
                        c.lemma,
                        c.pos,
                        c.score + (0.6 if c.lemma == "κρύπτω" else 0.0),
                        c.provenance,
                        fixed_surface=c.fixed_surface,
                        required_case=c.required_case,
                        morph_override=c.morph_override,
                    )
                    for c in cands
                ]
                if not any(c.lemma == "κρύπτω" for c in boosted):
                    boosted.append(SlotCandidate("κρύπτω", "V-", 2.2, "N"))
                cands = sorted(boosted, key=lambda c: -c.score)[:6]
            elif v_m.lemma == "ⲟⲩⲱⲛϩ" and cur_tense == "A":
                if not any(c.lemma == "ὁράω" for c in cands):
                    cands = sorted(
                        [*cands, SlotCandidate("ὁράω", "V-", 2.25, "N")],
                        key=lambda c: -c.score,
                    )[:6]
        elif eff_voice == "A" and cands and cands[0].lemma.endswith("ομαι"):
            eff_voice = "M"
        return cands, eff_voice

    def _emit_det_noun_slots(
        det_m: CopticMorpheme,
        det_idx: int,
        head_m: CopticMorpheme,
        head_idx: int,
        np_case: str,
        case_gov: str,
        prep_seqs: list[int],
    ) -> None:
        num = _det_number(det_m, head_m)
        h_cands = _lookup_content_candidates(
            head_m,
            desired_pos="N-",
            lexicon=lexicon,
            ccl_lexicon=ccl_lexicon,
            right_ctx=extract_coptic_right_context(ms, head_idx),
        )
        gnd = inflector.lemma_gender(h_cands[0].lemma) or _det_gender_hint(det_m)
        if det_m.pos == "ART":
            if det_m.lemma == "ⲡ":
                slots.append(
                    TransferSlot(
                        role="ART",
                        coptic_seqs=[*prep_seqs, det_m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="RA", case=np_case, number=num, gender=gnd
                        ),
                        candidates=[SlotCandidate("ὁ", "RA", 2.2, "C")],
                        agree_head_offset=1,
                        case_governor=case_gov,
                    )
                )
            elif det_m.lemma == "ⲕⲉ":
                slots.append(
                    TransferSlot(
                        role="ADJ",
                        coptic_seqs=[*prep_seqs, det_m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="A-", case=np_case, number=num, gender=gnd
                        ),
                        candidates=[SlotCandidate("ἄλλος", "A-", 2.0, "C")],
                        agree_head_offset=1,
                        case_governor=case_gov,
                    )
                )
        elif det_m.pos in {"PPOS", "PDEM"}:
            slots.append(
                TransferSlot(
                    role="ART",
                    coptic_seqs=[*prep_seqs, det_m.seq],
                    target_morph=GreekMorphFeatures(
                        pos="RA",
                        case=np_case,
                        number=num,
                        gender=gnd,
                    ),
                    candidates=[SlotCandidate("ὁ", "RA", 2.1, "C")],
                    agree_head_offset=1,
                    case_governor=case_gov,
                )
            )
        elif det_m.pos == "NUM":
            n_cands = _num_cands(det_m, det_idx)
            slots.append(
                TransferSlot(
                    role="NUM",
                    coptic_seqs=[*prep_seqs, det_m.seq],
                    target_morph=GreekMorphFeatures(
                        pos="A-", case=np_case, number=num, gender=gnd
                    ),
                    candidates=n_cands,
                    agree_head_offset=1,
                    case_governor=case_gov,
                )
            )
        slots.append(
            TransferSlot(
                role="NOUN",
                coptic_seqs=[head_m.seq],
                target_morph=GreekMorphFeatures(
                    pos=h_cands[0].pos, case=np_case, number=num, gender=gnd
                ),
                candidates=h_cands,
                case_governor=case_gov,
            )
        )
        if det_m.pos == "PPOS":
            pl, pp, pper, pnum = _pronoun_person_num(det_m, subj_p, subj_n)
            slots.append(
                TransferSlot(
                    role="POSS",
                    coptic_seqs=[det_m.seq],
                    target_morph=GreekMorphFeatures(
                        pos=pp,
                        person=pper,
                        case="G",
                        number=pnum or "S",
                        gender="M",
                    ),
                    candidates=[SlotCandidate(pl, pp, 2.1, "C")],
                )
            )
        elif det_m.pos == "PDEM":
            d_lem, d_pos, _, _ = _pronoun_person_num(det_m, None, None)
            slots.append(
                TransferSlot(
                    role="DEM",
                    coptic_seqs=[det_m.seq],
                    target_morph=GreekMorphFeatures(
                        pos=d_pos, case=np_case, number=num, gender=gnd
                    ),
                    candidates=[SlotCandidate(d_lem, d_pos, 2.0, "C")],
                    case_governor=case_gov,
                )
            )
            if head_idx + 1 < n and ms[head_idx + 1].pos == "PDEM":
                suppress_seqs.add(ms[head_idx + 1].seq)

    subj_p, subj_n = _extract_clause_subject(clause)
    clause_has_2p = any(
        x.lemma in {"ⲛⲧⲱⲧⲛ", "ⲡⲉⲧⲛ"}
        or x.norm in {"ⲧⲉⲧⲛϣⲁⲛ", "ⲉⲧⲉⲧⲛϣⲁⲛ", "ⲧⲉⲧⲛ", "ⲧⲛ", "ⲧⲏⲩⲧⲛ", "ⲧⲏⲛⲉ"}
        for x in ms
    )
    clause_has_2s = any(x.lemma in {"ⲛⲧⲟⲕ", "ⲛⲧⲟ", "ⲡⲉⲕ", "ⲡⲟⲩ"} for x in ms)
    has_tam_prefix = any(
        x.pos
        in {
            "APST",
            "FUT",
            "CPRET",
            "PPERS",
            "AOPT_PPERS",
            "ACONJ_PPERS",
            "ACOND",
            "ACOND_PPERS",
            "ANEGJUS",
            "ANEGOPT",
            "ANEGAOR",
            "ANEGPST",
            "AAOR",
            "ALIM",
        }
        for x in ms
    )
    clause_has_neg_an = any(m.pos == "NEG" and m.lemma == "ⲁⲛ" for m in ms)
    has_apst = any(m.pos == "APST" for m in ms)
    has_anegpst = any(m.pos == "ANEGPST" for m in ms)
    has_anegopt = any(m.pos == "ANEGOPT" for m in ms)
    has_anegjus = any(m.pos == "ANEGJUS" and m.lemma != "ⲙⲡⲣⲧⲣⲉ" for m in ms)
    has_prohib = any(m.lemma in {"ⲙⲡⲣ", "ⲙⲡⲣⲧⲣⲉ"} for m in ms)
    has_fut = any(m.pos == "FUT" for m in ms)
    has_cpret = any(m.pos == "CPRET" for m in ms)
    has_acond = any(m.pos in {"ACOND", "ACOND_PPERS"} or m.lemma == "ⲉϣⲱⲡⲉ" for m in ms)
    has_hotan = any(m.lemma == "ϩⲟⲧⲁⲛ" for m in ms)
    has_alim = any(m.pos == "ALIM" for m in ms)
    has_hina = any(m.lemma in {"ϩⲓⲛⲁ", "ϫⲉⲕⲁⲥ"} for m in ms)

    # Determine dominant finite clause tense & mood
    tense = main_tense
    mood = main_mood
    if has_apst or has_anegpst:
        tense, mood = "A", "I"
    elif has_cpret:
        tense, mood = "I", "I"
    elif has_fut:
        tense, mood = "F", "I"
    if has_prohib:
        tense, mood = "P", "D"
    elif has_anegjus or has_anegopt:
        tense, mood = "A", "S"
    elif has_acond or has_hotan or has_alim or has_hina:
        tense, mood = ("A" if has_apst else "P"), "S"

    # Track whether we are inside a nominal subject vs object position relative to verb
    seen_verb = False
    last_verb_coptic_lemma = ""
    vbd_subject_pending = False
    quote_start_i = -1
    comparative_nom_pending = False
    in_oblique_rel_clause = False
    in_copular_become = False
    in_beatitude_nom = False
    in_exist_nom = False
    is_nominal_clause = not any(
        m.pos in {"V", "VBD", "VIMP", "EXIST"} for m in ms
    ) and any(m.pos == "COP" for m in ms)
    suppress_seqs: set[int] = set()

    def _has_upcoming_cop(pos_idx: int) -> bool:
        for k in range(pos_idx + 1, min(pos_idx + 5, n)):
            if ms[k].pos == "COP":
                return True
            if ms[k].pos in {"V", "VIMP", "EXIST", "CONJ"}:
                break
        return False

    # Check if clause uses double negative or postverbal trailing ⲁⲛ
    pending_neg_seqs = [m.seq for m in ms if m.pos == "NEG" and m.lemma in {"ⲁⲛ", "ⲛ"}]
    emitted_preverbal_neg = False

    _local_tam_set = {
        "APST",
        "FUT",
        "CPRET",
        "PPERS",
        "AOPT_PPERS",
        "ACONJ_PPERS",
        "ACOND",
        "ACOND_PPERS",
        "ANEGJUS",
        "ANEGOPT",
        "ANEGAOR",
        "ANEGPST",
        "AAOR",
        "ALIM",
        "CCIRC",
        "ACAUS",
    }

    def _is_bare_imperative(pos_idx: int, morph: CopticMorpheme) -> tuple[bool, str]:
        if morph.pos == "VIMP":
            return True, "P" if (clause_has_2p and not clause_has_2s) else "S"
        has_mpr = any(
            x.pos == "NEG" and x.lemma == "ⲙⲡⲣ"
            for x in ms[max(0, pos_idx - 2) : pos_idx + 1]
        )
        if has_mpr:
            num_d = (
                "P" if ((clause_has_2p or unit_has_2p) and not clause_has_2s) else "S"
            )
            return True, num_d
        local_tam = any(
            x.pos in _local_tam_set for x in ms[max(0, pos_idx - 3) : pos_idx]
        )
        if (
            morph.pos == "V"
            and not local_tam
            and (
                pos_idx == quote_start_i
                or (
                    not any(sl.role in {"VERB", "COP"} for sl in slots)
                    and (
                        not any(
                            x.pos in {"N", "NPROP", "PPERI", "ART"}
                            for x in ms[:pos_idx]
                        )
                        or (
                            not any(
                                x.pos in {"N", "NPROP", "ART"} for x in ms[:pos_idx]
                            )
                            and any(
                                x.pos == "PPERI" and x.lemma in {"ⲛⲧⲟⲕ", "ⲛⲧⲟ", "ⲛⲧⲱⲧⲛ"}
                                for x in ms[:pos_idx]
                            )
                            and not any(
                                x.pos == "PPERI"
                                and x.lemma not in {"ⲛⲧⲟⲕ", "ⲛⲧⲟ", "ⲛⲧⲱⲧⲛ"}
                                for x in ms[:pos_idx]
                            )
                        )
                    )
                )
            )
        ):
            has_2p_top = any(
                x.pos == "PPERI" and x.lemma == "ⲛⲧⲱⲧⲛ" for x in ms[:pos_idx]
            )
            return (
                True,
                "P" if (has_2p_top or (clause_has_2p and not clause_has_2s)) else "S",
            )
        _ = has_tam_prefix
        return False, "S"

    i = 0
    while i < n:
        m = ms[i]
        if m.seq in suppress_seqs:
            i += 1
            continue

        # Punctuation
        if m.pos == "PUNCT":
            slots.append(
                TransferSlot(
                    role="PUNCT",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(pos="X-"),
                    candidates=[
                        SlotCandidate(
                            lemma=m.norm,
                            pos="X-",
                            score=0.0,
                            provenance="C",
                            fixed_surface=m.norm,
                        )
                    ],
                    is_punct=True,
                )
            )
            i += 1
            continue

        # Skip pure trailing negative ⲁⲛ if pre-verbal οὐ was/will be emitted
        if m.pos == "NEG" and m.lemma == "ⲁⲛ":
            if not emitted_preverbal_neg and not seen_verb:
                # e.g. nominal negation -> οὐκ
                pass
            i += 1
            continue

        # Multiword Idiom Match (ϩⲉ+ⲉ -> εὑρίσκω, ϫⲓ+ϯⲡⲉ -> γεύομαι, ...)
        idiom_hit = idioms.match_at(ms, i)
        if idiom_hit is not None:
            rule, span_len = idiom_hit
            consumed_seqs = [ms[i + k].seq for k in range(span_len)]
            if rule.greek_pos == "V-":
                # Emit verb slot using clause tense/mood/person/num
                vp, vn = subj_p or "3", subj_n or "S"
                for k in range(max(0, i - 3), i):
                    if ms[k].pos == "PPERS" and ms[k].lemma in _PPERS_MAP:
                        vp, vn = _PPERS_MAP[ms[k].lemma]
                seen_verb = True
                last_verb_coptic_lemma = ms[i].lemma
                if (clause_has_neg_an or has_anegpst) and not emitted_preverbal_neg:
                    slots.append(
                        TransferSlot(
                            role="NEG",
                            coptic_seqs=pending_neg_seqs,
                            target_morph=GreekMorphFeatures(pos="D-"),
                            candidates=[
                                SlotCandidate("οὐ", "D-", 2.2, "C", fixed_surface="οὐ")
                            ],
                        )
                    )
                    emitted_preverbal_neg = True
                vm_person, vm_number, vm_mood = vp, vn, mood
                is_imp, imp_num = _is_bare_imperative(i, ms[i])
                if is_imp:
                    vm_person, vm_number, vm_mood = "2", imp_num, "D"
                vm = GreekMorphFeatures(
                    pos="V-",
                    person=vm_person,
                    tense=tense,
                    voice="M" if rule.greek_lemma.endswith("ομαι") else "A",
                    mood=vm_mood,
                    number=vm_number,
                )
                id_cands, id_voice = _refine_verb_cands_and_voice(
                    ms[i],
                    i,
                    [
                        SlotCandidate(
                            lemma=rule.greek_lemma,
                            pos="V-",
                            score=2.6,
                            provenance="N",
                            required_case=rule.required_case,
                        )
                    ],
                    vm.voice or "A",
                    tense,
                    obj_check_idx=i + span_len - 1,
                    skip_disc=True,
                )
                vm = vm.model_copy(update={"voice": id_voice})
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=consumed_seqs,
                        target_morph=vm,
                        candidates=id_cands,
                    )
                )
                i += span_len
                continue
            elif rule.greek_pos in {"N-", "A-", "I-"}:
                c_case = "N" if not seen_verb else "A"
                gnd = inflector.lemma_gender(rule.greek_lemma) or "N"
                slots.append(
                    TransferSlot(
                        role="NOUN",
                        coptic_seqs=consumed_seqs,
                        target_morph=GreekMorphFeatures(
                            pos=rule.greek_pos,
                            case=c_case if rule.greek_pos != "I-" else None,
                            number="S" if rule.greek_pos != "I-" else None,
                            gender=gnd if rule.greek_pos != "I-" else None,
                        ),
                        candidates=[
                            SlotCandidate(
                                lemma=rule.greek_lemma,
                                pos=rule.greek_pos,
                                score=2.5,
                                provenance="N",
                                fixed_surface=rule.greek_lemma
                                if rule.greek_pos == "I-"
                                else None,
                            )
                        ],
                    )
                )
                i += span_len
                continue
            elif rule.greek_pos == "C-":
                slots.append(
                    TransferSlot(
                        role="CONJ",
                        coptic_seqs=consumed_seqs,
                        target_morph=GreekMorphFeatures(pos="C-"),
                        candidates=[
                            SlotCandidate(
                                lemma=rule.greek_lemma,
                                pos="C-",
                                score=2.4,
                                provenance="C",
                                fixed_surface=rule.greek_lemma,
                            )
                        ],
                    )
                )
                in_oblique_rel_clause = False
                in_copular_become = False
                last_verb_coptic_lemma = ""
                if rule.greek_lemma == "ὡς":
                    comparative_nom_pending = True
                i += span_len
                continue
            elif rule.greek_pos == "D-":
                slots.append(
                    TransferSlot(
                        role="ADV",
                        coptic_seqs=consumed_seqs,
                        target_morph=GreekMorphFeatures(pos="D-"),
                        candidates=[
                            SlotCandidate(
                                lemma=rule.greek_lemma,
                                pos="D-",
                                score=2.5,
                                provenance="C",
                                fixed_surface=rule.greek_lemma,
                            )
                        ],
                    )
                )
                i += span_len
                continue
            elif rule.greek_pos == "P-":
                req_c = rule.required_case or idioms.prep_case(rule.greek_lemma)
                slots.append(
                    TransferSlot(
                        role="PREP",
                        coptic_seqs=consumed_seqs,
                        target_morph=GreekMorphFeatures(pos="P-"),
                        candidates=[
                            SlotCandidate(
                                lemma=rule.greek_lemma,
                                pos="P-",
                                score=2.4,
                                provenance="C",
                                fixed_surface=rule.greek_lemma,
                                required_case=req_c,
                            )
                        ],
                    )
                )
                # Suppress trailing PPERO / PREP case translation via next NP
                i += span_len
                continue

        # 1. Clause-initial conjunctions / temporal / conditional markers
        if m.pos in {"CONJ", "APREC", "ALIM", "ACOND"} or (
            m.pos == "PTC" and m.lemma != "ⲛϭⲓ"
        ):
            in_oblique_rel_clause = False
            in_copular_become = False
            last_verb_coptic_lemma = ""
            cands = [
                SlotCandidate(lem, pos, sc, "C", fixed_surface=fs)
                for lem, pos, sc, fs in _CONJ_MAP.get(m.lemma, [])
            ]
            if not cands:
                cands = _lookup_content_candidates(
                    m,
                    desired_pos="C-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, i),
                )
            slots.append(
                TransferSlot(
                    role="CONJ",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(pos="C-"),
                    candidates=cands,
                    optional_omit=(m.lemma == "ϫⲉ" and len(slots) <= 2),
                )
            )
            # Emit ἄν after ὅταν/ἐάν optional or inherent in ἐάν/ὅταν
            i += 1
            continue

        # 2. Negations: ⲛⲛⲉ -> οὐ μή + Subj; ⲙⲡ -> οὐ; ⲙⲡⲣ/ⲙⲡⲣⲧⲣⲉ -> μή + Impv
        if m.pos in {"ANEGJUS", "ANEGOPT"}:
            if m.lemma in {"ⲙⲡⲣ", "ⲙⲡⲣⲧⲣⲉ"}:
                slots.append(
                    TransferSlot(
                        role="NEG",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(pos="D-"),
                        candidates=[
                            SlotCandidate("μή", "D-", 2.4, "C", fixed_surface="μή")
                        ],
                    )
                )
                mood = "D"
            else:
                slots.append(
                    TransferSlot(
                        role="NEG",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(pos="D-"),
                        candidates=[
                            SlotCandidate("οὐ", "D-", 2.4, "C", fixed_surface="οὐ")
                        ],
                    )
                )
                slots.append(
                    TransferSlot(
                        role="NEG",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(pos="D-"),
                        candidates=[
                            SlotCandidate("μή", "D-", 2.4, "C", fixed_surface="μή")
                        ],
                    )
                )
                tense, mood = "A", "S"
            emitted_preverbal_neg = True
            i += 1
            continue

        if m.pos in {"ANEGPST", "ANEGAOR", "NEG"} and m.lemma != "ⲁⲛ":
            neg_w = "μή" if m.lemma in {"ⲙⲡⲣ", "ⲙⲡⲣⲧⲣⲉ", "ⲧⲙ"} else "οὐ"
            if neg_w == "μή":
                mood = "S"
            elif m.pos == "ANEGAOR":
                tense, mood = "P", "I"
            slots.append(
                TransferSlot(
                    role="NEG",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(pos="D-"),
                    candidates=[
                        SlotCandidate(neg_w, "D-", 2.3, "C", fixed_surface=neg_w)
                    ],
                )
            )
            emitted_preverbal_neg = True
            i += 1
            continue

        # 2b. Fused conditional prefix ACOND_PPERS (ⲉϥϣⲁⲛ, ⲉⲧⲉⲧⲛϣⲁⲛ, ...)
        if m.pos == "ACOND_PPERS":
            subj_p, subj_n = _ppers_person_num(m)
            mood = "S"
            if not any(s.role == "CONJ" for s in slots):
                slots.append(
                    TransferSlot(
                        role="CONJ",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(pos="C-"),
                        candidates=[
                            SlotCandidate("ἐάν", "C-", 2.1, "C", fixed_surface="ἐάν")
                        ],
                    )
                )
            i += 1
            continue

        # 3a. Circumstantial converter CCIRC (`ⲉ`/`ⲉⲣⲉ` + PPERS + V/VSTAT) -> Participle
        if m.pos == "CCIRC" and (
            seen_verb or (slots and slots[-1].role in {"NOUN", "PRON", "PART", "VERB"})
        ):
            cc_k = i + 1
            cc_p, cc_n = subj_p or "3", subj_n or "S"
            cc_neg = False
            cc_saw_fut = False
            while cc_k < n and ms[cc_k].pos in {"PPERS", "NEG", "FUT"}:
                if ms[cc_k].pos == "PPERS":
                    cc_p, cc_n = _ppers_person_num(ms[cc_k])
                elif ms[cc_k].pos == "NEG":
                    cc_neg = True
                elif ms[cc_k].pos == "FUT":
                    cc_saw_fut = True
                cc_k += 1
            if not cc_saw_fut and cc_k < n and ms[cc_k].pos in {"V", "VSTAT"}:
                cc_v = ms[cc_k]
                ant_c, ant_n, ant_g = "N", subj_n or "S", "M"
                cc_ant = (
                    _find_emitted(cc_v.head_seq)
                    if cc_v.dep_rel in {"acl", "acl:relcl"}
                    else None
                )
                if cc_ant is not None:
                    atm = cc_ant.target_morph
                    ant_c = atm.case or "N"
                    ant_n = atm.number or "S"
                    ant_g = atm.gender or "M"
                elif slots and slots[-1].role in {"NOUN", "PRON", "PART"}:
                    atm = slots[-1].target_morph
                    ant_c = atm.case or "N"
                    ant_n = atm.number or "S"
                    ant_g = atm.gender or "M"
                is_impers_pass = (
                    (cc_p, cc_n) == ("3", "P")
                    and ant_n == "S"
                    and cc_v.pos == "V"
                    and cc_k + 2 < n
                    and ms[cc_k + 1].pos == "PREP"
                    and ms[cc_k + 1].lemma in {"ⲛ", "ⲙⲙⲟ"}
                    and ms[cc_k + 2].pos == "PPERO"
                    and ms[cc_k + 2].lemma in {"ⲛⲧⲟϥ", "ⲛⲧⲟⲥ"}
                )
                if is_impers_pass:
                    suppress_seqs.update({ms[cc_k + 1].seq, ms[cc_k + 2].seq})
                if (
                    in_oblique_rel_clause
                    and cc_k + 1 < n
                    and ms[cc_k + 1].pos == "PPERO"
                ):
                    suppress_seqs.add(ms[cc_k + 1].seq)
                elif (
                    in_oblique_rel_clause
                    and cc_k + 2 < n
                    and ms[cc_k + 1].pos == "PREP"
                    and ms[cc_k + 1].lemma in {"ⲛ", "ⲙⲙⲟ", "ⲉⲣⲟ", "ⲛϩⲏⲧ"}
                    and ms[cc_k + 2].pos == "PPERO"
                ):
                    suppress_seqs.update({ms[cc_k + 1].seq, ms[cc_k + 2].seq})
                if cc_neg:
                    slots.append(
                        TransferSlot(
                            role="NEG",
                            coptic_seqs=[ms[k].seq for k in range(i + 1, cc_k)],
                            target_morph=GreekMorphFeatures(pos="D-"),
                            candidates=[
                                SlotCandidate("μή", "D-", 2.2, "C", fixed_surface="μή")
                            ],
                        )
                    )
                cc_cands = _lookup_content_candidates(
                    cc_v,
                    desired_pos="V-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, cc_k),
                )
                last_verb_coptic_lemma = cc_v.lemma
                cc_tense = "X" if cc_v.pos == "VSTAT" else "P"
                cc_voice = "P" if (cc_v.pos == "VSTAT" or is_impers_pass) else "A"
                cc_cands, cc_voice = _refine_verb_cands_and_voice(
                    cc_v, cc_k, cc_cands, cc_voice, cc_tense
                )
                slots.append(
                    TransferSlot(
                        role="PART",
                        coptic_seqs=[ms[k].seq for k in range(i, cc_k + 1)],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            tense=cc_tense,
                            voice=cc_voice,
                            mood="P",
                            case=ant_c,
                            number=ant_n,
                            gender=ant_g,
                        ),
                        candidates=cc_cands,
                    )
                )
                i = cc_k + 1
                continue

        # 3. Analytic Coptic TAM auxiliaries that fuse into Greek verb morphology
        if m.pos in {
            "APST",
            "FUT",
            "AAOR",
            "AJUS",
            "AFUTCONJ",
            "CPRET",
            "CCIRC",
            "CFOC",
            "ACAUS",
            "ANY",
        }:
            if m.lemma in {"ⲙⲁⲣⲉ", "ⲧⲣⲉ"}:
                # Causative/Jussive can use Imperative (if ⲙⲁⲣ-) or ἵνα+Subj
                if m.lemma == "ⲙⲁⲣⲉ":
                    mood = "D"
            elif m.lemma == "ⲙⲡⲁⲧⲉ":
                slots.append(
                    TransferSlot(
                        role="CONJ",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(pos="C-"),
                        candidates=[
                            SlotCandidate("πρίν", "C-", 1.9, "C", fixed_surface="πρίν")
                        ],
                    )
                )
                mood = "N"
            i += 1
            continue

        # Optative / Conjunctive fused PPERS (ⲉϥⲉ -> 3S future/subjunctive, ⲧⲁ -> 1S)
        if m.pos in {"AOPT_PPERS", "ACONJ_PPERS"}:
            p, nu = _PPERS_MAP.get(m.lemma, ("3", "S"))
            subj_p, subj_n = p, nu
            if m.pos == "ACONJ_PPERS":
                slots.append(
                    TransferSlot(
                        role="CONJ",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(pos="C-"),
                        candidates=[
                            SlotCandidate("καί", "C-", 2.0, "C", fixed_surface="καί")
                        ],
                    )
                )
            i += 1
            continue

        if m.pos == "ACONJ":
            slots.append(
                TransferSlot(
                    role="CONJ",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(pos="C-"),
                    candidates=[
                        SlotCandidate("καί", "C-", 2.0, "C", fixed_surface="καί")
                    ],
                )
            )
            i += 1
            continue

        # Handle PPERS directly before V/FUT/VSTAT or bipartite locative PREP/ADV
        if m.pos == "PPERS":
            subj_p, subj_n = _ppers_person_num(m)
            if (
                (not seen_verb or (i > 0 and ms[i - 1].pos == "CONJ"))
                and (
                    i == 0
                    or ms[i - 1].pos
                    not in {
                        "ACONJ",
                        "APST",
                        "ANEGPST",
                        "AAOR",
                        "CCIRC",
                        "CPRET",
                        "ALIM",
                        "AJUS",
                        "APREC",
                        "VBD",
                    }
                )
                and i + 1 < n
                and ms[i + 1].pos in {"PREP", "ADV"}
                and not any(
                    ms[k].pos
                    in {
                        "V",
                        "VSTAT",
                        "VBD",
                        "VIMP",
                        "EXIST",
                        "COP",
                        "FUT",
                        "APST",
                    }
                    for k in range(i + 1, n)
                    if ms[k].pos not in {"CONJ", "PUNCT"}
                )
            ):
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            person=subj_p,
                            tense="P",
                            voice="A",
                            mood="I",
                            number=subj_n,
                        ),
                        candidates=[SlotCandidate("εἰμί", "V-", 2.1, "C")],
                    )
                )
                seen_verb = True
            i += 1
            continue

        # 4. Articular relative -> Articular Participle: [ART] + CREL + ... + V
        if (
            m.pos == "ART" and i + 1 < n and ms[i + 1].pos == "CREL"
        ) or m.pos == "CREL":
            has_art = m.pos == "ART"
            crel_idx = i + 1 if has_art else i
            art_m = m if has_art else None
            rel_num = _det_number(art_m, None)
            rel_gnd = _det_gender_hint(art_m)
            rel_case = "N"
            # Determine case from preceding preposition / verb
            if i > 0 and ms[i - 1].pos == "PREP":
                if ms[i - 1].lemma == "ⲛ":
                    rel_case = "A" if seen_verb else "G"
                elif ms[i - 1].lemma in _PREP_MAP:
                    rel_case = _PREP_MAP[ms[i - 1].lemma][0][3]
            elif seen_verb:
                rel_case = "A"

            # Walk CREL + verbal prefix markers ONLY (stop before overt NP subjects)
            rel_prefix_pos = {
                "APST",
                "FUT",
                "NEG",
                "ANEGPST",
                "ANEGAOR",
                "PPERS",
                "AAOR",
                "CCIRC",
                "CFOC",
            }
            j = crel_idx + 1
            part_tense = "P"
            rel_person = "3"
            rel_subj_n = rel_num
            saw_ppers = False
            while j < n and ms[j].pos in rel_prefix_pos:
                if ms[j].pos == "APST":
                    part_tense = "A"
                elif ms[j].pos == "NEG":
                    slots.append(
                        TransferSlot(
                            role="NEG",
                            coptic_seqs=[ms[j].seq],
                            target_morph=GreekMorphFeatures(pos="D-"),
                            candidates=[
                                SlotCandidate("μή", "D-", 2.2, "C", fixed_surface="μή")
                            ],
                        )
                    )
                elif ms[j].pos == "PPERS":
                    saw_ppers = True
                    rel_person, rel_subj_n = _ppers_person_num(ms[j])
                j += 1

            # Relative verb right here without an overt nominal subject inside relative
            if j < n and ms[j].pos in {"V", "VSTAT"}:
                v_m = ms[j]
                if v_m.pos == "VSTAT":
                    part_tense = "X"
                rel_idiom = idioms.match_at(ms, j)
                if rel_idiom is not None and rel_idiom[0].greek_pos == "V-":
                    id_rule, id_span = rel_idiom
                    j_end = j + id_span - 1
                    v_seqs = [ms[k].seq for k in range(crel_idx, j_end + 1)]
                    v_cands = [
                        SlotCandidate(
                            id_rule.greek_lemma,
                            "V-",
                            2.6,
                            "N",
                            required_case=id_rule.required_case,
                        )
                    ]
                    base_rel_v = (
                        "P"
                        if v_m.pos == "VSTAT"
                        else "M"
                        if id_rule.greek_lemma.endswith("ομαι")
                        else "A"
                    )
                    skip_rel_disc = True
                else:
                    j_end = j
                    v_seqs = [ms[k].seq for k in range(crel_idx, j_end + 1)]
                    v_cands = _lookup_content_candidates(
                        v_m,
                        desired_pos="V-",
                        lexicon=lexicon,
                        ccl_lexicon=ccl_lexicon,
                        right_ctx=extract_coptic_right_context(ms, j),
                    )
                    base_rel_v = "P" if v_m.pos == "VSTAT" else "A"
                    skip_rel_disc = False
                last_verb_coptic_lemma = v_m.lemma
                # Suppress trailing resumptive PP+PPERO right after relative verb
                if (
                    j_end + 2 < n
                    and ms[j_end + 1].pos == "PREP"
                    and ms[j_end + 1].lemma in {"ⲛ", "ⲙⲙⲟ", "ϩⲛ", "ⲉ"}
                    and ms[j_end + 2].pos == "PPERO"
                    and ms[j_end + 2].lemma in {"ⲛⲧⲟϥ", "ⲛⲧⲟⲥ", "ⲛⲧⲟⲟⲩ"}
                ):
                    suppress_seqs.update({ms[j_end + 1].seq, ms[j_end + 2].seq})

                v_cands, eff_rel_voice = _refine_verb_cands_and_voice(
                    v_m,
                    j,
                    v_cands,
                    base_rel_v,
                    part_tense,
                    obj_check_idx=j_end,
                    skip_disc=skip_rel_disc,
                )

                head_slot = (
                    slots[-1]
                    if not has_art
                    and slots
                    and slots[-1].role in {"NOUN", "PRON", "PART"}
                    else None
                )
                if has_art or (head_slot is not None and not saw_ppers):
                    # Emit Greek articular participle (ὁ/ἡ/τό/οἱ/τά + V-participle)
                    if head_slot is not None:
                        p_case = head_slot.target_morph.case or rel_case
                        p_num = head_slot.target_morph.number or "S"
                        p_gnd = head_slot.target_morph.gender or "M"
                    else:
                        p_case, p_num, p_gnd = rel_case, rel_num, rel_gnd
                        if (
                            p_num == "S"
                            and p_gnd == "M"
                            and v_m.pos == "VSTAT"
                            and v_m.lemma in {"ϩⲱⲡ", "ϭⲱⲗⲡ", "ⲟⲩⲱⲛϩ", "ⲥϩⲁⲓ", "ϫⲱⲕ"}
                        ):
                            p_gnd = "N"
                    slots.append(
                        TransferSlot(
                            role="PART_ART",
                            coptic_seqs=[art_m.seq] if art_m else v_seqs[:1],
                            target_morph=GreekMorphFeatures(
                                pos="RA", case=p_case, number=p_num, gender=p_gnd
                            ),
                            candidates=[SlotCandidate("ὁ", "RA", 2.3, "C")],
                        )
                    )
                    part_voice = eff_rel_voice
                    slots.append(
                        TransferSlot(
                            role="PART",
                            coptic_seqs=v_seqs,
                            target_morph=GreekMorphFeatures(
                                pos="V-",
                                tense=part_tense,
                                voice=part_voice,
                                mood="P",
                                case=p_case,
                                number=p_num,
                                gender=p_gnd,
                            ),
                            candidates=v_cands,
                        )
                    )
                else:
                    # Bare relative pronoun ὅς / οἵ + finite verb (pro-drop subject)
                    rel_ant = (
                        _find_emitted(v_m.head_seq)
                        if v_m.dep_rel in {"acl:relcl", "acl"}
                        else _find_emitted(ms[crel_idx].head_seq)
                    )
                    if rel_ant is not None:
                        h_num = rel_ant.target_morph.number or rel_subj_n
                        h_gnd = rel_ant.target_morph.gender or "M"
                    else:
                        h_num = (
                            slots[-1].target_morph.number or rel_subj_n
                            if slots and slots[-1].role in {"NOUN", "PRON", "PART"}
                            else rel_subj_n
                        )
                        h_gnd = (
                            slots[-1].target_morph.gender or "M"
                            if slots and slots[-1].role in {"NOUN", "PRON", "PART"}
                            else "M"
                        )
                    slots.append(
                        TransferSlot(
                            role="REL_PRON",
                            coptic_seqs=[ms[crel_idx].seq],
                            target_morph=GreekMorphFeatures(
                                pos="RR", case="N", number=h_num, gender=h_gnd
                            ),
                            candidates=[
                                SlotCandidate("ὅς", "RR", 2.1, "C"),
                                SlotCandidate("ὅστις", "RR", 1.7, "C"),
                            ],
                        )
                    )
                    slots.append(
                        TransferSlot(
                            role="VERB",
                            coptic_seqs=v_seqs,
                            target_morph=GreekMorphFeatures(
                                pos="V-",
                                person=rel_person,
                                tense=part_tense,
                                voice=eff_rel_voice,
                                mood="I",
                                number=rel_subj_n,
                            ),
                            candidates=v_cands,
                        )
                    )
                if vbd_subject_pending and (
                    j_end + 1 >= n or ms[j_end + 1].pos != "CREL"
                ):
                    seen_verb = True
                    vbd_subject_pending = False
                i = j_end + 1
                continue
            elif not has_art:
                # Overt nominal subject inside relative clause (`ⲉⲛⲧ ⲁ ⲓⲏⲥⲟⲩⲥ ... ϫⲟⲟⲩ`)
                rel_ant2 = None
                for rk in range(j, min(j + 8, n)):
                    if ms[rk].pos in {"V", "VSTAT"} and ms[rk].dep_rel in {
                        "acl:relcl",
                        "acl",
                    }:
                        rel_ant2 = _find_emitted(ms[rk].head_seq)
                        if rel_ant2 is not None:
                            break
                if rel_ant2 is None and ms[crel_idx].head_seq is not None:
                    rel_ant2 = _find_emitted(ms[crel_idx].head_seq)
                if rel_ant2 is not None:
                    h_num = rel_ant2.target_morph.number or "S"
                    h_gnd = rel_ant2.target_morph.gender or "M"
                else:
                    h_num = (
                        slots[-1].target_morph.number or "S"
                        if slots and slots[-1].role in {"NOUN", "PRON", "PART"}
                        else "S"
                    )
                    h_gnd = (
                        slots[-1].target_morph.gender or "M"
                        if slots and slots[-1].role in {"NOUN", "PRON", "PART"}
                        else "M"
                    )
                has_resumptive_obj = any(
                    ms[k].pos == "PPERO" for k in range(j, min(j + 8, n))
                )
                slots.append(
                    TransferSlot(
                        role="REL_PRON",
                        coptic_seqs=[ms[crel_idx].seq],
                        target_morph=GreekMorphFeatures(
                            pos="RR",
                            case="A" if has_resumptive_obj else "N",
                            number=h_num,
                            gender=h_gnd,
                        ),
                        candidates=[
                            SlotCandidate("ὅς", "RR", 2.1, "C"),
                            SlotCandidate("ὅστις", "RR", 1.7, "C"),
                        ],
                    )
                )
                if has_resumptive_obj:
                    in_oblique_rel_clause = True
                seen_verb = False
                vbd_subject_pending = True
                i = crel_idx + 1
                continue

        # 5. Direct speech / verb head (`ⲡⲉϫⲉ`, `V`, `VBD`, `VIMP`, `VSTAT`)
        if m.pos in {"V", "VBD", "VIMP", "VSTAT"}:
            seen_verb = True
            # Emit pending pre-verbal οὐ if trailing ⲁⲛ appears in clause
            if clause_has_neg_an and not emitted_preverbal_neg:
                slots.append(
                    TransferSlot(
                        role="NEG",
                        coptic_seqs=pending_neg_seqs,
                        target_morph=GreekMorphFeatures(pos="D-"),
                        candidates=[
                            SlotCandidate("οὐ", "D-", 2.2, "C", fixed_surface="οὐ")
                        ],
                    )
                )
                emitted_preverbal_neg = True

            if m.lemma in {"ⲡⲉϫⲉ", "ⲡⲉϫⲁ"}:
                vp, vn = "3", "S"
                has_overt_subj = False
                # Check if ⲡⲉϫⲁ=ϥ vs ⲡⲉϫⲁ=ⲩ vs ⲡⲉϫⲉ+NP
                if i + 1 < n and ms[i + 1].pos in {"PPERO", "PPERS"}:
                    if ms[i + 1].pos == "PPERS":
                        vp, vn = _ppers_person_num(ms[i + 1])
                    else:
                        _, _, vp, vn_v = _pronoun_person_num(ms[i + 1], None, None)
                        vp = vp or "3"
                        vn = vn_v or "S"
                    suppress_seqs.add(ms[i + 1].seq)
                elif i + 1 < n and ms[i + 1].pos in {
                    "NPROP",
                    "ART",
                    "PPOS",
                    "PDEM",
                    "N",
                }:
                    has_overt_subj = True
                    nxt_np_head = ms[i + 2] if i + 2 < n else None
                    if _det_number(ms[i + 1], nxt_np_head) == "P":
                        vn = "P"
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            person=vp,
                            tense="P",
                            voice="A",
                            mood="I",
                            number=vn,
                        ),
                        candidates=[
                            SlotCandidate(
                                "λέγω",
                                "V-",
                                2.3,
                                "N",
                                morph_override=GreekMorphFeatures(
                                    pos="V-",
                                    person=vp,
                                    tense="P",
                                    voice="A",
                                    mood="I",
                                    number=vn,
                                ),
                            ),
                            SlotCandidate(
                                "λέγω",
                                "V-",
                                2.0,
                                "N",
                                morph_override=GreekMorphFeatures(
                                    pos="V-",
                                    person=vp,
                                    tense="A",
                                    voice="A",
                                    mood="I",
                                    number=vn,
                                ),
                            ),
                        ],
                    )
                )
                if has_overt_subj:
                    seen_verb = False
                    vbd_subject_pending = True
                i += 1
                continue

            if m.lemma == "ⲛⲁⲓⲁⲧ":
                b_num = "P"
                consumed = [m.seq]
                if i + 1 < n and ms[i + 1].pos in {"PPERO", "PPOS", "PPERS"}:
                    if ms[i + 1].pos == "PPERS":
                        _, b_num = _ppers_person_num(ms[i + 1])
                    else:
                        _, _, _, p_nu = _pronoun_person_num(ms[i + 1], None, None)
                        b_num = p_nu or "P"
                    consumed.append(ms[i + 1].seq)
                    suppress_seqs.add(ms[i + 1].seq)
                if i + 2 < n and ms[i + 2].pos == "PREP" and ms[i + 2].lemma == "ⲛ":
                    in_beatitude_nom = True
                slots.append(
                    TransferSlot(
                        role="ADJ",
                        coptic_seqs=consumed,
                        target_morph=GreekMorphFeatures(
                            pos="A-", case="N", number=b_num, gender="M"
                        ),
                        candidates=[SlotCandidate("μακάριος", "A-", 2.6, "N")],
                    )
                )
                i += 1
                continue

            if m.lemma in {"ⲛⲁⲛⲟⲩ", "ⲛⲁϣⲉ", "ⲛⲁϣⲱ"}:
                adj_lem = "καλός" if m.lemma == "ⲛⲁⲛⲟⲩ" else "πολύς"
                consumed_a = [m.seq]
                if i + 1 < n and ms[i + 1].pos == "PPERO":
                    suppress_seqs.add(ms[i + 1].seq)
                    consumed_a.append(ms[i + 1].seq)
                slots.append(
                    TransferSlot(
                        role="ADJ",
                        coptic_seqs=consumed_a,
                        target_morph=GreekMorphFeatures(
                            pos="A-", case="N", number="S", gender="M"
                        ),
                        candidates=[SlotCandidate(adj_lem, "A-", 2.4, "N")],
                    )
                )
                i += 1
                continue

            # Modal ability verb `ϣ + V` -> `δύναμαι` + Infinitive
            if m.pos == "V" and m.lemma == "ϣ" and i + 1 < n and ms[i + 1].pos == "V":
                inf_m = ms[i + 1]
                inf_cands = _lookup_content_candidates(
                    inf_m,
                    desired_pos="V-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, i + 1),
                )
                inf_cands, inf_voice = _refine_verb_cands_and_voice(
                    inf_m, i + 1, inf_cands, "A", "A"
                )
                last_verb_coptic_lemma = inf_m.lemma
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            person=subj_p or "3",
                            tense=tense,
                            voice="M",
                            mood=mood,
                            number=subj_n or "S",
                        ),
                        candidates=[SlotCandidate("δύναμαι", "V-", 2.4, "N")],
                    )
                )
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[inf_m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            tense="A" if inf_voice == "P" else "P",
                            voice=inf_voice,
                            mood="N",
                        ),
                        candidates=inf_cands,
                    )
                )
                i += 2
                continue

            if (
                m.pos == "V"
                and len(m.lemma) >= 4
                and m.lemma.startswith("ϣ")
                and (
                    m.lemma[1:] in _LABILE_PASSIVE_LEMMAS
                    or (
                        not lexicon.lookup(
                            coptic_lemma=m.lemma,
                            coptic_pos="V",
                            coptic_norm=m.norm,
                            lang=m.lang,
                            ccl_lexicon=ccl_lexicon,
                        )
                        and bool(
                            lexicon.lookup(
                                coptic_lemma=m.lemma[1:],
                                coptic_pos="V",
                                coptic_norm=m.lemma[1:],
                                lang=m.lang,
                                ccl_lexicon=ccl_lexicon,
                            )
                        )
                    )
                )
            ):
                sub_lem = m.lemma[1:]
                sub_m = m.model_copy(update={"lemma": sub_lem, "norm": sub_lem})
                inf_cands = _lookup_content_candidates(
                    sub_m,
                    desired_pos="V-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, i),
                )
                inf_cands, inf_voice = _refine_verb_cands_and_voice(
                    sub_m, i, inf_cands, "A", "A"
                )
                last_verb_coptic_lemma = sub_lem
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            person=subj_p or "3",
                            tense=tense,
                            voice="M",
                            mood=mood,
                            number=subj_n or "S",
                        ),
                        candidates=[SlotCandidate("δύναμαι", "V-", 2.4, "N")],
                    )
                )
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            tense="A" if inf_voice == "P" else "P",
                            voice=inf_voice,
                            mood="N",
                        ),
                        candidates=inf_cands,
                    )
                )
                i += 1
                continue

            vm_person = subj_p or "3"
            vm_number = subj_n or "S"
            vm_mood = "D" if m.pos == "VIMP" else mood
            is_imp, imp_num = _is_bare_imperative(i, m)
            if is_imp:
                vm_person, vm_number, vm_mood = "2", imp_num, "D"
            vm_tense = "X" if m.pos == "VSTAT" else tense
            v_voice = "P" if m.pos == "VSTAT" else "A"
            pref_gin = m.lemma in {"ⲥⲟⲟⲩⲛ", "ⲥⲟⲩⲱⲛ"} and (
                vm_mood in {"D", "S", "P"} or vm_tense in {"F", "A"}
            )
            v_cands = _lookup_content_candidates(
                m,
                desired_pos="V-",
                lexicon=lexicon,
                ccl_lexicon=ccl_lexicon,
                prefer_ginosko=pref_gin,
                right_ctx=extract_coptic_right_context(ms, i),
            )
            last_verb_coptic_lemma = m.lemma
            if (
                m.lemma in {"ϫⲱ", "ϫⲟⲟ"}
                and i + 1 < n
                and ms[i + 1].pos == "PPERO"
                and ms[i + 1].lemma == "ⲛⲧⲟⲥ"
                and any(ms[k].lemma == "ϫⲉ" for k in range(i + 2, min(i + 6, n)))
            ):
                suppress_seqs.add(ms[i + 1].seq)
            if in_oblique_rel_clause:
                if (
                    i + 1 < n
                    and ms[i + 1].pos == "PPERO"
                    and ms[i + 1].lemma
                    in {
                        "ⲛⲧⲟϥ",
                        "ⲛⲧⲟⲥ",
                        "ⲛⲧⲟⲟⲩ",
                    }
                ):
                    suppress_seqs.add(ms[i + 1].seq)
                elif (
                    i + 2 < n
                    and ms[i + 1].pos == "PREP"
                    and ms[i + 1].lemma in {"ⲛ", "ⲙⲙⲟ", "ⲉⲣⲟ", "ⲛϩⲏⲧ"}
                    and ms[i + 2].pos == "PPERO"
                    and ms[i + 2].lemma in {"ⲛⲧⲟϥ", "ⲛⲧⲟⲥ", "ⲛⲧⲟⲟⲩ"}
                ):
                    suppress_seqs.update({ms[i + 1].seq, ms[i + 2].seq})
            v_cands, v_voice = _refine_verb_cands_and_voice(
                m, i, v_cands, v_voice, vm_tense
            )
            in_copular_become = m.lemma == "ϣⲱⲡⲉ" or (
                bool(v_cands) and v_cands[0].lemma == "γίνομαι"
            )
            slots.append(
                TransferSlot(
                    role="VERB",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(
                        pos="V-",
                        person=vm_person,
                        tense=vm_tense,
                        voice=v_voice,
                        mood=vm_mood,
                        number=vm_number,
                    ),
                    candidates=v_cands,
                )
            )
            i += 1
            continue

        # 6. Existential ⲟⲩⲛ/ⲙⲛ (modal `ⲙⲛ ϭⲟⲙ`, `ⲟⲩⲛⲧⲁ=`, Nom pivot + `ϣⲏⲡ`)
        is_pseudo_exist_mn = (
            m.pos == "PREP"
            and m.lemma == "ⲙⲛ"
            and i + 2 < n
            and (
                (
                    ms[i + 1].lemma == "ϭⲟⲙ"
                    and ms[i + 2].pos in {"ACONJ", "ACONJ_PPERS", "PREP", "ACAUS"}
                )
                or (
                    ms[i + 1].pos == "N"
                    and ms[i + 2].pos in {"V", "VSTAT", "CCIRC", "CREL"}
                )
            )
        )
        if m.pos == "EXIST" or is_pseudo_exist_mn:

            def _resolve_inf_verb(
                v_idx: int,
            ) -> tuple[list[SlotCandidate], str, int, list[int]]:
                vm = ms[v_idx]
                id_hit = idioms.match_at(ms, v_idx)
                if id_hit is not None and id_hit[0].greek_pos == "V-":
                    id_rule, id_span = id_hit
                    cands = [
                        SlotCandidate(
                            id_rule.greek_lemma,
                            "V-",
                            2.6,
                            "N",
                            required_case=id_rule.required_case,
                        )
                    ]
                    base_v = "M" if id_rule.greek_lemma.endswith("ομαι") else "A"
                    cands, v_out = _refine_verb_cands_and_voice(
                        vm,
                        v_idx,
                        cands,
                        base_v,
                        "A",
                        obj_check_idx=v_idx + id_span - 1,
                        skip_disc=True,
                    )
                    return (
                        cands,
                        v_out,
                        v_idx + id_span,
                        [ms[v_idx + x].seq for x in range(id_span)],
                    )
                cands = _lookup_content_candidates(
                    vm,
                    desired_pos="V-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, v_idx),
                )
                cands, v_out = _refine_verb_cands_and_voice(vm, v_idx, cands, "A", "A")
                return cands, v_out, v_idx + 1, [vm.seq]

            if m.lemma == "ⲙⲛ" and i + 1 < n and ms[i + 1].lemma == "ϭⲟⲙ":
                k_m = i + 2
                # Case A: ⲙⲛ ϭⲟⲙ + ACONJ / ACONJ_PPERS + [PPERS] + V
                if k_m < n and ms[k_m].pos in {"ACONJ", "ACONJ_PPERS"}:
                    ac_p, ac_n = "3", "S"
                    if ms[k_m].pos == "ACONJ_PPERS":
                        ac_p, ac_n = _ppers_person_num(ms[k_m])
                        k_m += 1
                    else:
                        k_m += 1
                        while k_m < n and ms[k_m].pos in {"PPERS", "NEG"}:
                            if ms[k_m].pos == "PPERS":
                                ac_p, ac_n = _ppers_person_num(ms[k_m])
                            k_m += 1
                    if k_m < n and ms[k_m].pos == "V":
                        inf_cands, inf_voice, next_i, v_seqs = _resolve_inf_verb(k_m)
                        last_verb_coptic_lemma = ms[k_m].lemma
                        slots.append(
                            TransferSlot(
                                role="NEG",
                                coptic_seqs=[m.seq],
                                target_morph=GreekMorphFeatures(pos="D-"),
                                candidates=[
                                    SlotCandidate(
                                        "οὐ", "D-", 2.3, "C", fixed_surface="οὐ"
                                    )
                                ],
                            )
                        )
                        emitted_preverbal_neg = True
                        slots.append(
                            TransferSlot(
                                role="VERB",
                                coptic_seqs=[m.seq, ms[i + 1].seq],
                                target_morph=GreekMorphFeatures(
                                    pos="V-",
                                    person=ac_p,
                                    tense="P",
                                    voice="M",
                                    mood="I",
                                    number=ac_n,
                                ),
                                candidates=[SlotCandidate("δύναμαι", "V-", 2.5, "C")],
                            )
                        )
                        slots.append(
                            TransferSlot(
                                role="VERB",
                                coptic_seqs=v_seqs,
                                target_morph=GreekMorphFeatures(
                                    pos="V-", tense="A", voice=inf_voice, mood="N"
                                ),
                                candidates=inf_cands,
                            )
                        )
                        seen_verb = True
                        i = next_i
                        continue
                # Case B: ⲙⲛ ϭⲟⲙ + PREP("ⲉ") + V
                if (
                    k_m + 1 < n
                    and ms[k_m].pos == "PREP"
                    and ms[k_m].lemma == "ⲉ"
                    and ms[k_m + 1].pos == "V"
                ):
                    v_idx = k_m + 1
                    inf_cands, inf_voice, next_i, v_seqs = _resolve_inf_verb(v_idx)
                    last_verb_coptic_lemma = ms[v_idx].lemma
                    slots.append(
                        TransferSlot(
                            role="NEG",
                            coptic_seqs=[m.seq],
                            target_morph=GreekMorphFeatures(pos="D-"),
                            candidates=[
                                SlotCandidate("οὐ", "D-", 2.3, "C", fixed_surface="οὐ")
                            ],
                        )
                    )
                    emitted_preverbal_neg = True
                    slots.append(
                        TransferSlot(
                            role="VERB",
                            coptic_seqs=[m.seq, ms[i + 1].seq],
                            target_morph=GreekMorphFeatures(
                                pos="V-",
                                person="3",
                                tense="P",
                                voice="M",
                                mood="I",
                                number="S",
                            ),
                            candidates=[SlotCandidate("δύναμαι", "V-", 2.5, "C")],
                        )
                    )
                    slots.append(
                        TransferSlot(
                            role="VERB",
                            coptic_seqs=[ms[k_m].seq, *v_seqs],
                            target_morph=GreekMorphFeatures(
                                pos="V-", tense="A", voice=inf_voice, mood="N"
                            ),
                            candidates=inf_cands,
                        )
                    )
                    seen_verb = True
                    i = next_i
                    continue
                # Case C: ⲙⲛ ϭⲟⲙ + PREP("ⲛ") + [ART] + N + PREP("ⲉ") + V
                if k_m + 2 < n and ms[k_m].pos == "PREP" and ms[k_m].lemma == "ⲛ":
                    s_idx = k_m + 1
                    det_m_c: CopticMorpheme | None = None
                    if ms[s_idx].pos in {"ART", "PPOS", "PDEM"} and s_idx + 1 < n:
                        det_m_c = ms[s_idx]
                        s_idx += 1
                    if (
                        s_idx + 2 < n
                        and ms[s_idx].pos in {"N", "NPROP"}
                        and ms[s_idx + 1].pos == "PREP"
                        and ms[s_idx + 1].lemma == "ⲉ"
                        and ms[s_idx + 2].pos == "V"
                    ):
                        head_m_c = ms[s_idx]
                        v_idx = s_idx + 2
                        np_num = _det_number(det_m_c, head_m_c)
                        inf_cands, inf_voice, next_i, v_seqs = _resolve_inf_verb(v_idx)
                        last_verb_coptic_lemma = ms[v_idx].lemma
                        slots.append(
                            TransferSlot(
                                role="NEG",
                                coptic_seqs=[m.seq],
                                target_morph=GreekMorphFeatures(pos="D-"),
                                candidates=[
                                    SlotCandidate(
                                        "οὐ", "D-", 2.3, "C", fixed_surface="οὐ"
                                    )
                                ],
                            )
                        )
                        emitted_preverbal_neg = True
                        slots.append(
                            TransferSlot(
                                role="VERB",
                                coptic_seqs=[m.seq, ms[i + 1].seq],
                                target_morph=GreekMorphFeatures(
                                    pos="V-",
                                    person="3",
                                    tense="P",
                                    voice="M",
                                    mood="I",
                                    number=np_num,
                                ),
                                candidates=[SlotCandidate("δύναμαι", "V-", 2.5, "C")],
                            )
                        )
                        if det_m_c is not None:
                            _emit_det_noun_slots(
                                det_m_c, k_m + 1, head_m_c, s_idx, "N", "", []
                            )
                        else:
                            nc = _lookup_content_candidates(
                                head_m_c,
                                desired_pos="N-",
                                lexicon=lexicon,
                                ccl_lexicon=ccl_lexicon,
                                right_ctx=extract_coptic_right_context(ms, s_idx),
                            )
                            ng = inflector.lemma_gender(nc[0].lemma) or "M"
                            slots.append(
                                TransferSlot(
                                    role="NOUN",
                                    coptic_seqs=[head_m_c.seq],
                                    target_morph=GreekMorphFeatures(
                                        pos=nc[0].pos,
                                        case="N",
                                        number=np_num,
                                        gender=ng,
                                    ),
                                    candidates=nc,
                                )
                            )
                        slots.append(
                            TransferSlot(
                                role="VERB",
                                coptic_seqs=[ms[s_idx + 1].seq, *v_seqs],
                                target_morph=GreekMorphFeatures(
                                    pos="V-", tense="A", voice=inf_voice, mood="N"
                                ),
                                candidates=inf_cands,
                            )
                        )
                        seen_verb = True
                        i = next_i
                        continue

            seen_verb = True
            has_poss_dat = (
                m.pos == "EXIST"
                and i + 1 < n
                and ms[i + 1].pos in {"PREP", "PPERO"}
                and ms[i + 1].lemma in {"ⲛⲧⲉ", "ⲛ"}
            )
            if (
                not has_poss_dat
                and i + 2 < n
                and ms[i + 1].pos == "N"
                and ms[i + 2].lemma in {"ϣⲏⲡ", "ϣⲱⲡ"}
            ):
                pivot_m = ms[i + 1]
                pred_m = ms[i + 2]
                p_cands = _lookup_content_candidates(
                    pivot_m,
                    desired_pos="N-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, i + 1),
                )
                p_gnd = inflector.lemma_gender(p_cands[0].lemma) or "M"
                p_num = "P" if pivot_m.norm in _PLURAL_NOUN_NORMS else "S"
                if m.lemma == "ⲙⲛ":
                    slots.append(
                        TransferSlot(
                            role="NEG",
                            coptic_seqs=[m.seq],
                            target_morph=GreekMorphFeatures(pos="D-"),
                            candidates=[
                                SlotCandidate("οὐ", "D-", 2.3, "C", fixed_surface="οὐ")
                            ],
                        )
                    )
                    emitted_preverbal_neg = True
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-",
                            person="3",
                            tense="P",
                            voice="A",
                            mood="I",
                            number="S",
                        ),
                        candidates=[SlotCandidate("εἰμί", "V-", 2.2, "C")],
                    )
                )
                slots.append(
                    TransferSlot(
                        role="NOUN",
                        coptic_seqs=[pivot_m.seq],
                        target_morph=GreekMorphFeatures(
                            pos=p_cands[0].pos, case="N", number=p_num, gender=p_gnd
                        ),
                        candidates=p_cands,
                    )
                )
                slots.append(
                    TransferSlot(
                        role="ADJ",
                        coptic_seqs=[pred_m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="A-", case="N", number=p_num, gender=p_gnd
                        ),
                        candidates=[SlotCandidate("δεκτός", "A-", 2.6, "N")],
                    )
                )
                i += 3
                continue
            if m.lemma == "ⲙⲛ":
                slots.append(
                    TransferSlot(
                        role="NEG",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(pos="D-"),
                        candidates=[
                            SlotCandidate("οὐ", "D-", 2.3, "C", fixed_surface="οὐ")
                        ],
                    )
                )
                emitted_preverbal_neg = True
            slots.append(
                TransferSlot(
                    role="VERB",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(
                        pos="V-", person="3", tense="P", voice="A", mood="I", number="S"
                    ),
                    candidates=[
                        SlotCandidate("ἔχω" if has_poss_dat else "εἰμί", "V-", 2.2, "C")
                    ],
                )
            )
            if not has_poss_dat:
                in_exist_nom = True
            i += 1
            continue

        # 7. Copula ⲡⲉ / ⲧⲉ / ⲛⲉ -> Greek εἰμί matching clause 1st/2nd PRON
        if m.pos == "COP":
            c_per = "3"
            c_num = "P" if m.norm == "ⲛⲉ" else "S"
            for prev_slot in reversed(slots):
                if prev_slot.is_punct:
                    continue
                if prev_slot.role == "CONJ":
                    break
                if prev_slot.role == "PRON" and prev_slot.target_morph.person in {
                    "1",
                    "2",
                }:
                    c_per = prev_slot.target_morph.person
                    c_num = prev_slot.target_morph.number or c_num
                    break
            slots.append(
                TransferSlot(
                    role="COP",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(
                        pos="V-",
                        person=c_per,
                        tense="P",
                        voice="A",
                        mood="I",
                        number=c_num,
                    ),
                    candidates=[SlotCandidate("εἰμί", "V-", 1.9, "C")],
                    optional_omit=True,
                )
            )
            i += 1
            continue

        # 8. Postposed Nominative subject marker `ⲛϭⲓ` (suppress ⲛϭⲓ, mark next NP Nom)
        if m.pos == "PTC" and m.lemma == "ⲛϭⲓ":
            # Look ahead to next NP/pronoun -> handle normally as Nominative!
            seen_verb = False  # temporary reset so next NP takes Case 'N'
            i += 1
            continue

        # 9. Prepositions (`PREP`) + Coptic case-marker `ⲛ` (genitive/object/dative)
        if m.pos == "PREP":
            nxt = ms[i + 1] if i + 1 < n else None
            # Nominative predicate after beatitude `ⲛⲁⲓⲁⲧ`
            if in_beatitude_nom and m.lemma == "ⲛ":
                in_beatitude_nom = False
                seen_verb = False
                i += 1
                continue
            # Predicate noun/adjective after `ϣⲱⲡⲉ ⲛ-`/`ⲣ-` (γίνομαι + Predicate Nom)
            if (
                in_copular_become
                and m.lemma in {"ⲛ", "ⲣ"}
                and nxt is not None
                and nxt.pos in {"N", "A", "NPROP"}
            ):
                p_cands = _lookup_content_candidates(
                    nxt,
                    desired_pos="N-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, i + 1),
                )
                gnd = inflector.lemma_gender(p_cands[0].lemma) or "M"
                slots.append(
                    TransferSlot(
                        role="NOUN",
                        coptic_seqs=[m.seq, nxt.seq],
                        target_morph=GreekMorphFeatures(
                            pos=p_cands[0].pos,
                            case="N",
                            number=subj_n or "S",
                            gender=gnd,
                        ),
                        candidates=p_cands,
                    )
                )
                in_copular_become = False
                i += 2
                continue
            # `ⲉ + V` -> Greek complementary Infinitive (`mood='N'`, omit `εἰς`)
            if m.lemma == "ⲉ" and nxt is not None and nxt.pos == "V":
                inf_idiom = idioms.match_at(ms, i + 1)
                if inf_idiom is not None and inf_idiom[0].greek_pos == "V-":
                    id_rule, id_span = inf_idiom
                    inf_cands = [
                        SlotCandidate(
                            id_rule.greek_lemma,
                            "V-",
                            2.6,
                            "N",
                            required_case=id_rule.required_case,
                        )
                    ]
                    base_inf_v = "M" if id_rule.greek_lemma.endswith("ομαι") else "A"
                    inf_cands, inf_voice = _refine_verb_cands_and_voice(
                        nxt,
                        i + 1,
                        inf_cands,
                        base_inf_v,
                        "A",
                        obj_check_idx=i + id_span,
                        skip_disc=True,
                    )
                    last_verb_coptic_lemma = nxt.lemma
                    slots.append(
                        TransferSlot(
                            role="VERB",
                            coptic_seqs=[
                                m.seq,
                                *(ms[i + 1 + k].seq for k in range(id_span)),
                            ],
                            target_morph=GreekMorphFeatures(
                                pos="V-", tense="A", voice=inf_voice, mood="N"
                            ),
                            candidates=inf_cands,
                        )
                    )
                    seen_verb = True
                    i += 1 + id_span
                    continue
                inf_cands = _lookup_content_candidates(
                    nxt,
                    desired_pos="V-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, i + 1),
                )
                inf_cands, inf_voice = _refine_verb_cands_and_voice(
                    nxt, i + 1, inf_cands, "A", "A"
                )
                last_verb_coptic_lemma = nxt.lemma
                slots.append(
                    TransferSlot(
                        role="VERB",
                        coptic_seqs=[m.seq, nxt.seq],
                        target_morph=GreekMorphFeatures(
                            pos="V-", tense="A", voice=inf_voice, mood="N"
                        ),
                        candidates=inf_cands,
                    )
                )
                seen_verb = True
                i += 2
                continue
            # `ⲉ + ⲧⲣⲉ (ACAUS) + PPERS + V` -> Infinitive / ἵνα + Subjunctive
            if m.lemma == "ⲉ" and nxt is not None and nxt.pos == "ACAUS":
                k_idx = i + 2
                ac_p, ac_n = subj_p or "3", subj_n or "S"
                while k_idx < n and ms[k_idx].pos in {"PPERS", "NEG"}:
                    if ms[k_idx].pos == "PPERS":
                        ac_p, ac_n = _ppers_person_num(ms[k_idx])
                    k_idx += 1
                if k_idx < n and ms[k_idx].pos == "V":
                    inf_cands = _lookup_content_candidates(
                        ms[k_idx],
                        desired_pos="V-",
                        lexicon=lexicon,
                        ccl_lexicon=ccl_lexicon,
                        right_ctx=extract_coptic_right_context(ms, k_idx),
                    )
                    last_verb_coptic_lemma = ms[k_idx].lemma
                    inf_cands, inf_voice = _refine_verb_cands_and_voice(
                        ms[k_idx], k_idx, inf_cands, "A", "A"
                    )
                    same_subj = (ac_p, ac_n) == (subj_p or "3", subj_n or "S")
                    if same_subj:
                        slots.append(
                            TransferSlot(
                                role="VERB",
                                coptic_seqs=[ms[x].seq for x in range(i, k_idx + 1)],
                                target_morph=GreekMorphFeatures(
                                    pos="V-", tense="A", voice=inf_voice, mood="N"
                                ),
                                candidates=inf_cands,
                            )
                        )
                    else:
                        slots.append(
                            TransferSlot(
                                role="CONJ",
                                coptic_seqs=[m.seq, nxt.seq],
                                target_morph=GreekMorphFeatures(pos="C-"),
                                candidates=[
                                    SlotCandidate(
                                        "ἵνα", "C-", 2.2, "C", fixed_surface="ἵνα"
                                    )
                                ],
                            )
                        )
                        slots.append(
                            TransferSlot(
                                role="VERB",
                                coptic_seqs=[ms[k_idx].seq],
                                target_morph=GreekMorphFeatures(
                                    pos="V-",
                                    person=ac_p,
                                    tense="P",
                                    voice=inf_voice,
                                    mood="S",
                                    number=ac_n,
                                ),
                                candidates=inf_cands,
                            )
                        )
                    seen_verb = True
                    i = k_idx + 1
                    continue
            # Sahidic possessed locative: PREP(ϩⲛ/ⲛ/ⲙⲙⲟ) + PPOS + {ϩⲟⲩⲛ,ⲃⲟⲗ,ϩⲏ,ⲡⲁϩⲟⲩ}
            if (
                m.lemma in {"ϩⲛ", "ⲛ", "ⲙⲙⲟ"}
                and nxt is not None
                and nxt.pos == "PPOS"
                and i + 2 < n
                and ms[i + 2].lemma in _POSSESSED_LOCATIVE_MAP
            ):
                imp_prep = _POSSESSED_LOCATIVE_MAP[ms[i + 2].lemma]
                p_lem, p_pos, p_per, p_num = _pronoun_person_num(nxt, subj_p, subj_n)
                slots.append(
                    TransferSlot(
                        role="PREP",
                        coptic_seqs=[m.seq, ms[i + 2].seq],
                        target_morph=GreekMorphFeatures(pos="P-"),
                        candidates=[
                            SlotCandidate(
                                imp_prep,
                                "P-",
                                2.5,
                                "N",
                                fixed_surface=imp_prep,
                                required_case="G",
                            )
                        ],
                    )
                )
                slots.append(
                    TransferSlot(
                        role="PRON",
                        coptic_seqs=[nxt.seq],
                        target_morph=GreekMorphFeatures(
                            pos=p_pos,
                            person=p_per,
                            case="G",
                            number=p_num or "S",
                            gender="M" if p_lem == "αὐτός" else None,
                        ),
                        candidates=[SlotCandidate(p_lem, p_pos, 2.2, "C")],
                    )
                )
                i += 3
                continue

            # Direct object/genitive `ⲛ`/`ⲙⲙⲟ`, or oblique `ⲉ` after perception verbs
            is_direct_obj_prep = m.lemma in {"ⲛ", "ⲙⲙⲟ"} or (
                m.lemma in {"ⲉ", "ⲉⲣⲟ"}
                and seen_verb
                and last_verb_coptic_lemma in _DIRECT_OBJ_E_VERBS
                and nxt is not None
                and nxt.pos not in {"V", "ACAUS"}
            )
            if is_direct_obj_prep:
                if m.lemma in {"ⲉ", "ⲉⲣⲟ"}:
                    last_verb_coptic_lemma = ""
                has_pdem_head = (
                    nxt is not None
                    and nxt.pos == "PDEM"
                    and i + 2 < n
                    and ms[i + 2].pos in {"N", "NPROP", "NUM"}
                )
                has_num_n_head = (
                    nxt is not None
                    and nxt.pos == "NUM"
                    and i + 3 < n
                    and ms[i + 2].pos == "PREP"
                    and ms[i + 2].lemma == "ⲛ"
                    and ms[i + 3].pos in {"N", "NPROP"}
                )
                gov_m: CopticMorpheme | None = None
                if nxt is not None:
                    if has_num_n_head:
                        gov_m = ms[i + 3]
                    elif has_pdem_head:
                        gov_m = ms[i + 2]
                    elif nxt.pos in {"PPERO", "PPERI", "PDEM", "PINT", "N", "NPROP"}:
                        gov_m = nxt
                    elif i + 2 < n:
                        gov_m = ms[i + 2]
                if (
                    m.lemma in {"ⲛ", "ⲙⲙⲟ"}
                    and gov_m is not None
                    and gov_m.dep_rel in {"nmod", "nmod:poss"}
                ):
                    obj_case = "G"
                    case_gov = ""
                elif gov_m is not None and gov_m.dep_rel == "obj":
                    obj_case = "A"
                    case_gov = "verb_obj" if seen_verb else ""
                elif (
                    m.lemma in {"ⲛ", "ⲙⲙⲟ"}
                    and not seen_verb
                    and slots
                    and slots[-1].role in {"NOUN", "PRON", "PART"}
                ):
                    obj_case = "G"
                    case_gov = ""
                else:
                    obj_case = "A"
                    case_gov = "verb_obj" if seen_verb else ""
                if (
                    nxt is not None
                    and nxt.pos in {"PPERO", "PPERI", "PDEM", "PINT"}
                    and not has_pdem_head
                ):
                    p_lem, p_pos, p_per, p_num = _pronoun_person_num(
                        nxt, subj_p, subj_n
                    )
                    num_p = (
                        _det_number(nxt, None) if nxt.pos == "PDEM" else (p_num or "S")
                    )
                    slots.append(
                        TransferSlot(
                            role="PRON",
                            coptic_seqs=[m.seq, nxt.seq],
                            target_morph=GreekMorphFeatures(
                                pos=p_pos,
                                person=p_per,
                                case=obj_case,
                                number=num_p,
                                gender="M" if p_lem in {"αὐτός", "οὗτος"} else None,
                            ),
                            candidates=[SlotCandidate(p_lem, p_pos, 2.0, "C")],
                            case_governor=case_gov,
                        )
                    )
                    i += 2
                    continue
                if has_num_n_head and nxt is not None:
                    _emit_det_noun_slots(
                        nxt, i + 1, ms[i + 3], i + 3, obj_case, case_gov, [m.seq]
                    )
                    i += 4
                    if vbd_subject_pending:
                        seen_verb = True
                        vbd_subject_pending = False
                    continue
                # Attributed NP after `ⲛ`: mark next NP slots with obj_case + case_gov
                if nxt is not None:
                    art_m = nxt if nxt.pos in {"ART", "PPOS", "PDEM", "NUM"} else None
                    head_idx = i + 2 if art_m is not None else i + 1
                    head_m = ms[head_idx] if head_idx < n else None
                    if (
                        art_m is not None
                        and head_m is not None
                        and head_m.pos in {"N", "NPROP", "NUM", "PINT"}
                    ):
                        _emit_det_noun_slots(
                            art_m,
                            i + 1,
                            head_m,
                            head_idx,
                            obj_case,
                            case_gov,
                            [m.seq],
                        )
                        i = head_idx + 1
                        if vbd_subject_pending:
                            seen_verb = True
                            vbd_subject_pending = False
                        continue
                i += 1
                continue

            # Recipient `ⲛⲁ=` / `ⲛ` -> Dative pronoun without preposition
            if m.lemma == "ⲛⲁ" and nxt is not None and nxt.pos == "PPERO":
                p_lem, p_pos, p_per, p_num = _pronoun_person_num(nxt, subj_p, subj_n)
                slots.append(
                    TransferSlot(
                        role="PRON",
                        coptic_seqs=[m.seq, nxt.seq],
                        target_morph=GreekMorphFeatures(
                            pos=p_pos,
                            person=p_per,
                            case="D",
                            number=p_num or "S",
                            gender="M" if p_lem == "αὐτός" else None,
                        ),
                        candidates=[SlotCandidate(p_lem, p_pos, 2.2, "C")],
                    )
                )
                i += 2
                continue

            # True spatial/logical preposition (`ϩⲛ`, `ⲉ`, `ⲙⲛ`, `ⲉϫⲛ`, `ϩⲓⲧⲛ`, etc.)
            prep_cands = [
                SlotCandidate(gl, gp, sc, "C", fixed_surface=gl, required_case=rc)
                for gl, gp, sc, rc in _PREP_MAP.get(m.lemma, [])
            ]
            if not prep_cands:
                prep_cands = _lookup_content_candidates(
                    m,
                    desired_pos="P-",
                    lexicon=lexicon,
                    ccl_lexicon=ccl_lexicon,
                    right_ctx=extract_coptic_right_context(ms, i),
                )
            gov_case = prep_cands[0].required_case or idioms.prep_case(
                prep_cands[0].lemma
            )
            slots.append(
                TransferSlot(
                    role="PREP",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(pos="P-"),
                    candidates=prep_cands,
                )
            )
            has_pdem_sp = (
                nxt is not None
                and nxt.pos == "PDEM"
                and i + 2 < n
                and ms[i + 2].pos in {"N", "NPROP", "NUM"}
            )
            has_num_n_sp = (
                nxt is not None
                and nxt.pos == "NUM"
                and i + 3 < n
                and ms[i + 2].pos == "PREP"
                and ms[i + 2].lemma == "ⲛ"
                and ms[i + 3].pos in {"N", "NPROP"}
            )
            if (
                nxt is not None
                and nxt.pos in {"PPERO", "PPERI", "PDEM"}
                and not has_pdem_sp
            ):
                p_lem, p_pos, p_per, p_num = _pronoun_person_num(nxt, subj_p, subj_n)
                num_p = _det_number(nxt, None) if nxt.pos == "PDEM" else (p_num or "S")
                slots.append(
                    TransferSlot(
                        role="PRON",
                        coptic_seqs=[nxt.seq],
                        target_morph=GreekMorphFeatures(
                            pos=p_pos,
                            person=p_per,
                            case=gov_case,
                            number=num_p,
                            gender="M" if p_lem == "αὐτός" else None,
                        ),
                        candidates=[SlotCandidate(p_lem, p_pos, 2.1, "C")],
                    )
                )
                i += 2
                continue
            elif has_num_n_sp and nxt is not None:
                _emit_det_noun_slots(nxt, i + 1, ms[i + 3], i + 3, gov_case, "", [])
                i += 4
                continue
            elif nxt is not None:
                art_m = nxt if nxt.pos in {"ART", "PPOS", "PDEM", "NUM"} else None
                head_idx = i + 2 if art_m is not None else i + 1
                head_m = ms[head_idx] if head_idx < n else None
                if (
                    art_m is not None
                    and head_m is not None
                    and head_m.pos in {"N", "NPROP", "NUM", "PINT"}
                ):
                    _emit_det_noun_slots(
                        art_m, i + 1, head_m, head_idx, gov_case, "", []
                    )
                    i = head_idx + 1
                    continue
            i += 1
            continue

        # 10. Standalone NP: [ART | PPOS | PDEM | NUM] + N/NPROP
        if m.pos in {"ART", "PPOS", "PDEM", "NUM"}:
            nxt = ms[i + 1] if i + 1 < n else None
            has_num_n = (
                m.pos == "NUM"
                and nxt is not None
                and nxt.pos == "PREP"
                and nxt.lemma == "ⲛ"
                and i + 2 < n
                and ms[i + 2].pos in {"N", "NPROP"}
            )
            nom_m = (
                ms[i + 2]
                if has_num_n
                else nxt
                if (nxt is not None and nxt.pos in {"N", "NPROP", "NUM", "ADV"})
                else m
            )
            has_direct_prep = i > 0 and ms[i - 1].pos == "PREP"
            use_nom = (
                not seen_verb
                or is_nominal_clause
                or comparative_nom_pending
                or in_exist_nom
                or _has_upcoming_cop(i)
            )
            c_case = "N" if use_nom else "A"
            if not comparative_nom_pending and not is_nominal_clause:
                if nom_m.dep_rel in {"nsubj", "dislocated"} and not has_direct_prep:
                    c_case = "N"
                elif (
                    nom_m.dep_rel == "obj"
                    and seen_verb
                    and not in_exist_nom
                    and not has_direct_prep
                ):
                    c_case = "A"
            comparative_nom_pending = False
            if in_exist_nom and not has_direct_prep:
                in_exist_nom = False
            case_gov = (
                "verb_obj"
                if seen_verb and c_case == "A" and not is_nominal_clause
                else ""
            )
            if has_num_n:
                _emit_det_noun_slots(m, i, ms[i + 2], i + 2, c_case, case_gov, [])
                i += 3
                if vbd_subject_pending:
                    post_m = ms[i] if i < n else None
                    if post_m is None or (
                        post_m.pos != "CREL"
                        and not (post_m.pos == "PREP" and post_m.lemma == "ⲛ")
                    ):
                        seen_verb = True
                        vbd_subject_pending = False
                        quote_start_i = i
                continue
            elif nxt is not None and nxt.pos in {"N", "NPROP", "NUM", "ADV"}:
                _emit_det_noun_slots(m, i, nxt, i + 1, c_case, case_gov, [])
                i += 2
                if vbd_subject_pending:
                    post_m = ms[i] if i < n else None
                    if post_m is None or (
                        post_m.pos != "CREL"
                        and not (post_m.pos == "PREP" and post_m.lemma == "ⲛ")
                    ):
                        seen_verb = True
                        vbd_subject_pending = False
                        quote_start_i = i
                continue
            elif m.pos == "NUM":
                n_cands = _num_cands(m, i)
                num_val = "S" if m.lemma == "ⲟⲩⲁ" else "P"
                slots.append(
                    TransferSlot(
                        role="NUM",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos="A-", case=c_case, number=num_val, gender="M"
                        ),
                        candidates=n_cands,
                        case_governor=case_gov,
                    )
                )
                i += 1
                if vbd_subject_pending:
                    post_m = ms[i] if i < n else None
                    if post_m is None or post_m.pos != "CREL":
                        seen_verb = True
                        vbd_subject_pending = False
                continue
            else:
                # Bare demonstrative/pronoun/numeral standing alone (`ⲛⲁⲓ`, `ⲟⲩⲁ`)
                p_lem, p_pos, p_per, p_num = _pronoun_person_num(m, subj_p, subj_n)
                num_val = "P" if m.norm in {"ⲛⲁⲓ", "ⲛⲏ", "ⲛⲓ"} else (p_num or "S")
                slots.append(
                    TransferSlot(
                        role="PRON",
                        coptic_seqs=[m.seq],
                        target_morph=GreekMorphFeatures(
                            pos=p_pos,
                            person=p_per,
                            case=c_case,
                            number=num_val,
                            gender="N"
                            if m.lemma == "ⲡⲁⲓ" and is_nominal_clause
                            else "M",
                        ),
                        candidates=[SlotCandidate(p_lem, p_pos, 1.9, "C")],
                        case_governor=case_gov,
                    )
                )
                i += 1
                if vbd_subject_pending:
                    post_m = ms[i] if i < n else None
                    if post_m is None or post_m.pos != "CREL":
                        seen_verb = True
                        vbd_subject_pending = False
                continue

        # 11. Bare Noun / Proper Name (`N`, `NPROP`)
        if m.pos in {"N", "NPROP"}:
            has_direct_prep = i > 0 and ms[i - 1].pos == "PREP"
            use_nom = (
                not seen_verb
                or is_nominal_clause
                or comparative_nom_pending
                or in_exist_nom
                or _has_upcoming_cop(i)
            )
            c_case = "N" if use_nom else "A"
            if not comparative_nom_pending and not is_nominal_clause:
                if (
                    has_direct_prep
                    and ms[i - 1].lemma in {"ⲛ", "ⲙⲙⲟ"}
                    and m.dep_rel in {"nmod", "nmod:poss"}
                ):
                    c_case = "G"
                elif m.dep_rel in {"nsubj", "dislocated"} and not has_direct_prep:
                    c_case = "N"
                elif (
                    m.dep_rel == "obj"
                    and seen_verb
                    and not in_exist_nom
                    and not has_direct_prep
                ):
                    c_case = "A"
            comparative_nom_pending = False
            if in_exist_nom and not has_direct_prep:
                in_exist_nom = False
            case_gov = (
                "verb_obj"
                if seen_verb and c_case == "A" and not is_nominal_clause
                else ""
            )
            cands = _lookup_content_candidates(
                m,
                desired_pos="N-",
                lexicon=lexicon,
                ccl_lexicon=ccl_lexicon,
                right_ctx=extract_coptic_right_context(ms, i),
            )
            gnd = inflector.lemma_gender(cands[0].lemma) or "M"
            num = "P" if m.norm in _PLURAL_NOUN_NORMS else "S"
            slots.append(
                TransferSlot(
                    role="NOUN",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(
                        pos=cands[0].pos, case=c_case, number=num, gender=gnd
                    ),
                    candidates=cands,
                    case_governor=case_gov,
                )
            )
            i += 1
            if vbd_subject_pending:
                post_m = ms[i] if i < n else None
                if post_m is None or (
                    post_m.pos != "CREL"
                    and not (post_m.pos == "PREP" and post_m.lemma == "ⲛ")
                ):
                    seen_verb = True
                    vbd_subject_pending = False
                    quote_start_i = i
            continue

        # 12. Pronouns & Adverbs (`PPERI`, `PPERO`, `PINT`, `ADV`, `IMOD`)
        if m.pos in {"PPERI", "PPERO", "PINT"}:
            c_case = (
                "N"
                if (m.pos in {"PPERI", "PINT"} and (not seen_verb or in_exist_nom))
                else "A"
            )
            if in_exist_nom and m.pos != "PPERO":
                in_exist_nom = False
            if m.dep_rel in {"nsubj", "dislocated"}:
                c_case = "N"
            elif m.dep_rel == "obj" and seen_verb and c_case != "N":
                c_case = "A"
            case_gov = "verb_obj" if seen_verb and c_case == "A" else ""
            p_lem, p_pos, p_per, p_num = _pronoun_person_num(m, subj_p, subj_n)
            cands = [SlotCandidate(p_lem, p_pos, 2.0, "C")]
            if p_lem == "αὐτός" and c_case == "A" and subj_p == "3" and subj_n == p_num:
                cands.append(SlotCandidate("ἑαυτοῦ", "RP", 1.7, "C"))
            slots.append(
                TransferSlot(
                    role="PRON",
                    coptic_seqs=[m.seq],
                    target_morph=GreekMorphFeatures(
                        pos=p_pos,
                        person=p_per,
                        case=c_case,
                        number=p_num or "S",
                        gender="M" if p_lem in {"αὐτός", "οὗτος"} else None,
                    ),
                    candidates=cands,
                    case_governor=case_gov,
                )
            )
            i += 1
            continue

        # Adverb / remaining particle
        adv_cands = _lookup_content_candidates(
            m,
            desired_pos="D-",
            lexicon=lexicon,
            ccl_lexicon=ccl_lexicon,
            right_ctx=extract_coptic_right_context(ms, i),
        )
        slots.append(
            TransferSlot(
                role="ADV",
                coptic_seqs=[m.seq],
                target_morph=GreekMorphFeatures(pos=adv_cands[0].pos),
                candidates=adv_cands,
            )
        )
        i += 1

    # Wackernagel 2nd-position placement for detached postpositive (δέ / γάρ / μέν)
    if clause.postpositive is not None:
        pp = clause.postpositive
        pp_cands = [
            SlotCandidate(lem, pos, sc, "C", fixed_surface=fs)
            for lem, pos, sc, fs in _CONJ_MAP.get(pp.lemma, [("δέ", "C-", 2.0, "δέ")])
        ]
        pp_slot = TransferSlot(
            role="POSTPOS",
            coptic_seqs=[pp.seq],
            target_morph=GreekMorphFeatures(pos="C-"),
            candidates=pp_cands,
            postpositive=True,
        )
        non_punct = [k for k, s in enumerate(slots) if not s.is_punct]
        if non_punct:
            insert_at = non_punct[0] + 1
            slots.insert(insert_at, pp_slot)
        else:
            slots.append(pp_slot)

    if emitted_by_seq is not None:
        for sl in slots:
            if sl.role in {"NOUN", "PRON", "PART", "VERB", "ADJ"}:
                for sq in sl.coptic_seqs:
                    emitted_by_seq.setdefault(sq, sl)

    return slots, tense, mood


def build_unit_transfer_clauses(
    unit: CopticUnit,
    *,
    lexicon: LexicalTranslationModel,
    ccl_lexicon: CclLexicon,
    idioms: IdiomAndValencyTable,
    inflector: GreekInflector,
) -> list[list[TransferSlot]]:
    """Chunk ``unit`` into clauses and build transferred Greek slot plans."""
    clauses = chunk_unit(unit)
    unit_has_2p = any(
        m.lemma in {"ⲛⲧⲱⲧⲛ", "ⲡⲉⲧⲛ"}
        or m.norm in {"ⲧⲉⲧⲛϣⲁⲛ", "ⲉⲧⲉⲧⲛϣⲁⲛ", "ⲧⲉⲧⲛ", "ⲧⲏⲩⲧⲛ", "ⲧⲏⲛⲉ"}
        for m in unit.morphemes
    )
    out: list[list[TransferSlot]] = []
    emitted_by_seq: dict[int, TransferSlot] = {}
    cur_tense, cur_mood = "P", "I"
    for clause in clauses:
        inherit_tam = any(m.pos in {"ACONJ", "ACONJ_PPERS"} for m in clause.morphemes)
        slots, next_t, next_m = _clause_to_slots(
            clause,
            lexicon=lexicon,
            ccl_lexicon=ccl_lexicon,
            idioms=idioms,
            inflector=inflector,
            main_tense=cur_tense if inherit_tam else "P",
            main_mood=cur_mood if inherit_tam else "I",
            unit_has_2p=unit_has_2p,
            emitted_by_seq=emitted_by_seq,
        )
        if slots:
            out.append(slots)
        cur_tense, cur_mood = next_t, next_m
    return out
