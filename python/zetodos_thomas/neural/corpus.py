# SPDX-License-Identifier: Apache-2.0
"""Zero-leakage parallel/monolingual corpus builder & canonical serialization.

Strictly uses NT TRAIN (`nt-split-v1.json`), Sahidic OT x Swete LXX, Swete LXX
monolingual verses, and CCL/DDGLC loanwords. Never touches NT holdout, Dilley
Thomas, or P.Oxy. fragment witnesses during training.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from zetodos_thomas.ingest.apophthegmata import load_apophthegmata_corpus
from zetodos_thomas.ingest.download import STRONGS_XML_FILENAME
from zetodos_thomas.ingest.holdouts import (
    assert_clean_training_unit,
    clean_nt_train_ids,
    greek_token_ngrams,
    load_nt_split,
    shares_greek_token_ngram,
)
from zetodos_thomas.ingest.koine_patristics import load_all_patristic_koine_sentences
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt, pair_horner_morphgnt
from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.ingest.ot_swete import load_ot_swete_pairs, parse_swete_file
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.lexicon.gloss_bridge import build_ccl_strongs_bridge
from zetodos_thomas.morphology.coptic_analyzer import CopticAnalyzer
from zetodos_thomas.neural.accent_restorer import strip_greek_to_bare
from zetodos_thomas.reconstruct.chunker import chunk_unit
from zetodos_thomas.schemas.holdout import PassageRole
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit, GreekUnit
from zetodos_thomas.schemas.source import UnitKind

KNOWN_COPTIC_POS: tuple[str, ...] = (
    "ART",
    "PPERS",
    "PPERO",
    "PPOS",
    "PPERI",
    "PDEM",
    "PINT",
    "N",
    "NPROP",
    "V",
    "VSTAT",
    "VBD",
    "VIMP",
    "PREP",
    "CONJ",
    "PTC",
    "ADV",
    "NEG",
    "EXIST",
    "COP",
    "CREL",
    "CCIRC",
    "CFOC",
    "APST",
    "FUT",
    "CPRET",
    "AAOR",
    "ACONJ",
    "ACONJ_PPERS",
    "ACOND",
    "ACOND_PPERS",
    "AOPT",
    "AOPT_PPERS",
    "ACAUS",
    "AJUS",
    "ALIM",
    "APREC",
    "AFUTCONJ",
    "ANEGPST",
    "ANEGAOR",
    "ANEGJUS",
    "ANEGOPT",
    "ANY",
    "NUM",
    "IMOD",
    "PUNCT",
    "OTHER",
)

_KNOWN_POS_SET = frozenset(KNOWN_COPTIC_POS)
_CONTENT_LEM_POS = frozenset({"N", "NPROP", "V", "VSTAT", "VBD", "VIMP", "ADV"})
_EXPLICIT_POS_TAGS = frozenset({"VSTAT", "VIMP", "NPROP"})
_LOAN_HINT_POS = frozenset({"N", "NPROP", "V", "VSTAT", "VIMP", "ADV"})
_CLAUSE_SPLIT_NORMS = frozenset(
    {"ⲁⲩⲱ", "ϫⲉ", "ϩⲟⲧⲁⲛ", "ⲉϣⲱⲡⲉ", "ⲁⲗⲗⲁ", "ⲉⲓⲉ", "ⲧⲟⲧⲉ", "ⲇⲉ"}
)
_GREEK_STRIP_RE = re.compile(r"[\[\]<>«»\"“”„†‡*]+")
_GREEK_PUNCT_RE = re.compile(r"[.,;;·:!?—\-]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_SMOOTH_APOSTROPHE = "\u2019"

SPECIAL_USER_SYMBOLS: tuple[str, ...] = (
    "<DOM_NT>",
    "<DOM_OT>",
    "<DOM_BT>",
    "<DOM_LEX>",
    "<REV_COPTIC>",
    "<V1>",
    "<LEM>",
    "<LOAN>",
    "<PUNCT>",
    "<MASK>",
    *(f"<POS_{p}>" for p in KNOWN_COPTIC_POS),
)

SPECIAL_USER_SYMBOLS_V3: tuple[str, ...] = (
    *SPECIAL_USER_SYMBOLS,
    "<DOM_PAT>",
    "<DOM_APOP>",
    "<TGT_GRC>",
    "<TGT_ENG>",
    "<GLOSS>",
)

# Deterministic Enis & Megalaa script unification: map the 24 Greek-derived Coptic
# letters to lowercase Greek Unicode while keeping the 7 Demotic letters
# (ϣ, ϥ, ϧ, ϩ, ϫ, ϭ, ϯ) intact so subword embeddings share Greek loanword stems.
_COPTIC_TO_GREEK_SCRIPT = str.maketrans(
    {
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
    }
)


def greekify_coptic(text: str) -> str:
    """Map the 24 Greek-derived Coptic letters to bare lowercase Greek (final ς)."""
    translated = text.translate(_COPTIC_TO_GREEK_SCRIPT)
    if " " not in translated:
        if len(translated) > 1 and translated.endswith("σ"):
            return translated[:-1] + "ς"
        return translated
    words: list[str] = []
    for tok in translated.split():
        if len(tok) > 1 and tok.endswith("σ") and not tok.startswith("<"):
            tok = tok[:-1] + "ς"
        words.append(tok)
    return " ".join(words)


def normalize_greek_for_model(text: str, *, unaccented: bool = False) -> str:
    """Normalize Koine Greek target to acute-only NFC or unaccented bare lowercase."""
    norm = normalize_greek_target(text)
    return strip_greek_to_bare(norm) if unaccented else norm


def normalize_greek_target(text: str) -> str:
    """Normalize Koine Greek to acute-only NFC without editorial brackets/movable-nu."""
    cleaned = text.replace("(", "").replace(")", "")
    cleaned = _GREEK_STRIP_RE.sub(" ", cleaned)
    cleaned = (
        cleaned.replace("\u02bc", _SMOOTH_APOSTROPHE)
        .replace("\u02bd", _SMOOTH_APOSTROPHE)
        .replace("'", _SMOOTH_APOSTROPHE)
        .replace("\u1fbf", _SMOOTH_APOSTROPHE)
    )
    cleaned = _GREEK_PUNCT_RE.sub(" ", cleaned)
    nfd = unicodedata.normalize("NFD", cleaned)
    # Unify grave accents (\u0300) to acute (\u0301) to match MorphGNT normalized
    nfd = nfd.replace("\u0300", "\u0301")
    nfc = unicodedata.normalize("NFC", nfd)
    tokens: list[str] = []
    for raw in _MULTI_SPACE_RE.split(nfc.strip()):
        if not raw:
            continue
        if len(raw) > 1 and raw.endswith("σ"):
            raw = raw[:-1] + "ς"
        tokens.append(raw)
    return " ".join(tokens)


def normalize_greek_unit_target(unit: GreekUnit) -> str:
    """Extract normalized space-separated Koine Greek surface string from unit."""
    words: list[str] = []
    for tok in unit.tokens:
        if tok.morph.pos == "X-":
            continue
        raw = tok.normalized or tok.surface
        if not raw:
            continue
        norm = normalize_greek_target(raw)
        if norm:
            words.append(norm)
    return " ".join(words)


def _pos_tag_token(pos: str) -> str:
    return f"<POS_{pos}>" if pos in _KNOWN_POS_SET else "<POS_OTHER>"


def resolve_morpheme_loan_hint(
    m: CopticMorpheme, ccl_lexicon: CclLexicon | None
) -> str | None:
    """Resolve normalized Greek lemma hint for a Coptic Greek-loanword morpheme."""
    if ccl_lexicon is None or m.pos not in _LOAN_HINT_POS:
        return None
    norm_s = normalize_coptic(m.norm).replace("\u2cff", "")
    lem_s = normalize_coptic(m.lemma).replace("\u2cff", "")
    head = norm_s or lem_s
    if not head:
        return None
    if m.lang != "Greek" and len(lem_s or head) < 4:
        return None
    hits = ccl_lexicon.resolve(
        norm=norm_s or head,
        lemma=lem_s or head,
        pos=m.pos,
        lang=m.lang,
    )
    if hits:
        base = lem_s or head
        stripped = (
            base[2:]
            if base.startswith("ⲉⲣ") and len(base) > 5
            else base[1:]
            if base[:1] in {"ⲣ", "ϯ"} and len(base) > 4
            else base
        )
        if m.lang == "Greek" or hits[0].coptic_lemma in {base, stripped}:
            g_hint = normalize_greek_target(hits[0].greek_lemma)
            return g_hint or None
    return None


def serialize_coptic_morphemes(
    morphemes: Sequence[CopticMorpheme],
    *,
    ccl_lexicon: CclLexicon | None = None,
    domain_tag: str = "<DOM_NT>",
    v1_draft: str | None = None,
    loan_drop_p: float = 0.0,
    rng: random.Random | None = None,
    morpheme_drop_p: float = 0.0,
    greekify: bool = False,
    unaccented_hints: bool = False,
    native_gloss_hints: Mapping[str, str] | None = None,
    target_tag: str | None = None,
) -> tuple[str, list[tuple[int, int, int]]]:
    """Serialize Coptic morphemes compactly + optional `<V1>` draft + char spans.

    Returns ``(serialized_text, [(m.seq, char_start, char_end), ...])``.
    Char spans cover strictly the Coptic morpheme prefix (excluding `<V1>`).
    """
    parts: list[str] = [domain_tag]
    cursor = len(domain_tag)
    if target_tag:
        parts.append(target_tag)
        cursor += 1 + len(target_tag)
    spans: list[tuple[int, int, int]] = []

    for m in morphemes:
        if morpheme_drop_p > 0.0 and rng is not None and rng.random() < morpheme_drop_p:
            continue
        if m.pos == "PUNCT":
            seg = "<PUNCT>"
        else:
            norm_s = normalize_coptic(m.norm).replace("\u2cff", "")
            lem_s = normalize_coptic(m.lemma).replace("\u2cff", "")
            head = norm_s or lem_s
            if not head:
                continue
            disp_head = greekify_coptic(head) if greekify else head
            disp_lem = greekify_coptic(lem_s) if (greekify and lem_s) else lem_s
            toks = [disp_head]
            if m.pos in _CONTENT_LEM_POS and disp_lem and disp_lem != disp_head:
                toks.append("<LEM>")
                toks.append(disp_lem)
            if m.pos in _EXPLICIT_POS_TAGS:
                toks.append(_pos_tag_token(m.pos))
            loan_added = False
            if loan_drop_p <= 0.0 or rng is None or rng.random() >= loan_drop_p:
                g_hint = resolve_morpheme_loan_hint(m, ccl_lexicon)
                if g_hint:
                    if unaccented_hints:
                        g_hint = strip_greek_to_bare(g_hint)
                    if g_hint:
                        toks.append("<LOAN>")
                        toks.append(g_hint)
                        loan_added = True
            if (
                not loan_added
                and native_gloss_hints
                and m.pos in _CONTENT_LEM_POS
                and (loan_drop_p <= 0.0 or rng is None or rng.random() >= loan_drop_p)
            ):
                gl_hint = native_gloss_hints.get(lem_s or head)
                if gl_hint:
                    if unaccented_hints:
                        gl_hint = strip_greek_to_bare(gl_hint)
                    if gl_hint:
                        toks.append("<GLOSS>")
                        toks.append(gl_hint)
            seg = " ".join(toks)
        # Account for single separating space
        start = cursor + 1
        end = start + len(seg)
        parts.append(seg)
        spans.append((m.seq, start, end))
        cursor = end

    if v1_draft:
        clean_draft = normalize_greek_for_model(v1_draft, unaccented=unaccented_hints)
        if clean_draft:
            parts.append("<V1>")
            parts.append(clean_draft)

    return " ".join(parts), spans


def make_sub_coptic_unit(
    morphemes: Sequence[CopticMorpheme],
    *,
    unit_id: str = "chunk",
    source_id: str = "chunk",
) -> CopticUnit:
    """Build a valid ``CopticUnit`` from a morpheme slice with contiguous ``seq``."""
    old_to_new = {m.seq: idx for idx, m in enumerate(morphemes)}
    reindexed: list[CopticMorpheme] = []
    for idx, m in enumerate(morphemes):
        new_head = (
            old_to_new[m.head_seq]
            if m.head_seq is not None and m.head_seq in old_to_new
            else None
        )
        reindexed.append(m.model_copy(update={"seq": idx, "head_seq": new_head}))
    return CopticUnit(
        unit_id=unit_id,
        source_id=source_id,
        book="chunk",
        chapter=1,
        verse=1,
        kind=UnitKind.VERSE,
        original_text="",
        normalized_text=" ".join(m.norm for m in reindexed if m.norm),
        morphemes=reindexed,
    )


def _split_long_clause(
    ms: list[CopticMorpheme], max_morphemes: int
) -> list[list[CopticMorpheme]]:
    """Split an over-long unpunctuated Coptic clause near natural conjunctions."""
    out: list[list[CopticMorpheme]] = []
    start = 0
    n = len(ms)
    while start < n:
        if n - start <= max_morphemes:
            out.append(ms[start:])
            break
        lo = start + max(8, max_morphemes // 2)
        hi = min(n - 4, start + max_morphemes)
        split_at = hi
        # Prefer splitting immediately before a clause conjunction near the budget
        for idx in range(hi, lo - 1, -1):
            norm_i = normalize_coptic(ms[idx].norm).replace("\u2cff", "")
            if ms[idx].pos in {"CONJ", "PTC"} or norm_i in _CLAUSE_SPLIT_NORMS:
                split_at = idx
                break
        if split_at <= start:
            split_at = min(n, start + max_morphemes)
        out.append(ms[start:split_at])
        start = split_at
    return out


def split_coptic_unit_for_neural(
    unit: CopticUnit, *, max_morphemes: int = 36
) -> list[list[CopticMorpheme]]:
    """Keep verse-length units intact; split long logia to match NT/OT lengths."""
    ms = list(unit.morphemes)
    if len(ms) <= max_morphemes:
        return [ms] if ms else []

    # 1. Split on punctuation first, packing consecutive short sentences up to limit
    clauses = chunk_unit(unit)
    chunks: list[list[CopticMorpheme]] = []
    buf: list[CopticMorpheme] = []
    for cl in clauses:
        cl_ms = list(cl.morphemes)
        if not cl_ms:
            continue
        if buf and len(buf) + len(cl_ms) > max_morphemes:
            chunks.append(buf)
            buf = []
        if len(cl_ms) > max_morphemes:
            chunks.extend(_split_long_clause(cl_ms, max_morphemes))
        else:
            buf.extend(cl_ms)
    if buf:
        chunks.append(buf)
    return chunks


@dataclass(frozen=True)
class ParallelExample:
    """One clean Coptic <-> Koine parallel training/dev pair."""

    unit_id: str
    domain_tag: str
    coptic_morphemes: tuple[CopticMorpheme, ...]
    greek_target: str


@dataclass(frozen=True)
class ParallelCorpusBundle:
    """Complete zero-leakage datasets for training and dev validation."""

    nt_train: list[ParallelExample]
    nt_dev: list[ParallelExample]
    nt_dev_units: list[tuple[CopticUnit, GreekUnit]]
    ot_train: list[ParallelExample]
    clause_aug: list[ParallelExample]
    lex_aug: list[tuple[str, str]]
    swete_unaligned_greek: list[str]
    all_clean_greek: list[str]
    all_clean_coptic_units: list[CopticUnit]
    # model-v3 optional non-biblical & lexicon-bridge datasets
    apop_coptic_units: list[CopticUnit] = field(default_factory=list)
    apop_eng_pairs: list[tuple[CopticUnit, str]] = field(default_factory=list)
    patristic_greek: list[str] = field(default_factory=list)
    grc_eng_pairs: list[tuple[str, str]] = field(default_factory=list)
    native_gloss_hints: dict[str, str] = field(default_factory=dict)


def _extract_clause_subpairs(
    c_unit: CopticUnit,
    g_unit: GreekUnit,
    domain_tag: str,
    *,
    unaccented: bool = False,
) -> list[ParallelExample]:
    """Extract high-precision sub-verse clause pairs when boundaries align."""

    def _split_coptic(split_conj: bool) -> list[list[CopticMorpheme]]:
        segs: list[list[CopticMorpheme]] = []
        cur: list[CopticMorpheme] = []
        for m in c_unit.morphemes:
            if m.pos == "PUNCT":
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            if (
                split_conj
                and len(cur) >= 4
                and normalize_coptic(m.norm).replace("\u2cff", "") == "ⲁⲩⲱ"
            ):
                segs.append(cur)
                cur = [m]
                continue
            cur.append(m)
        if cur:
            segs.append(cur)
        return segs

    def _split_greek(include_comma: bool, split_kai: bool) -> list[list[str]]:
        puncts = (
            (".", "·", ";", ";", "?", ",")
            if include_comma
            else (".", "·", ";", ";", "?")
        )
        segs: list[list[str]] = []
        cur: list[str] = []
        for tok in g_unit.tokens:
            raw_surf = tok.surface or ""
            norm_w = normalize_greek_for_model(
                tok.normalized or raw_surf, unaccented=unaccented
            )
            if tok.morph.pos == "X-":
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            if split_kai and len(cur) >= 3 and norm_w.lower() in {"καί", "και"}:
                segs.append(cur)
                cur = [norm_w]
                continue
            if norm_w:
                cur.append(norm_w)
            if any(ch in raw_surf for ch in puncts):
                if cur:
                    segs.append(cur)
                    cur = []
        if cur:
            segs.append(cur)
        return segs

    c_segs: list[list[CopticMorpheme]] = []
    g_segs: list[list[str]] = []
    for split_conj, inc_comma, split_kai in (
        (False, False, False),
        (False, True, False),
        (True, False, True),
    ):
        cs_try = _split_coptic(split_conj)
        gs_try = _split_greek(inc_comma, split_kai)
        if 2 <= len(cs_try) == len(gs_try) <= 4:
            c_segs, g_segs = cs_try, gs_try
            break

    if not c_segs:
        return []

    out: list[ParallelExample] = []
    for idx, (cs, gs) in enumerate(zip(c_segs, g_segs, strict=True)):
        if len(cs) < 3 or len(gs) < 2:
            continue
        ratio = len(gs) / max(len(cs), 1)
        if ratio < 0.25 or ratio > 2.2:
            continue
        out.append(
            ParallelExample(
                unit_id=f"{c_unit.unit_id}#cl{idx}",
                domain_tag=domain_tag,
                coptic_morphemes=tuple(cs),
                greek_target=" ".join(gs),
            )
        )
    return out


def select_patristic_training_sentences(
    sentences: Iterable[tuple[str, str]],
    *,
    seen_exact: Collection[str],
    holdout_verses: Iterable[str],
    unaccented: bool = False,
) -> list[str]:
    """Return patristic Greek kept for the language model and back-translation.

    Drops an empty line, a line under five tokens, an exact duplicate of other
    clean Greek, and any sentence that shares a whitespace token 5-gram with an
    NT holdout verse. Holdout text is not in the exact-duplicate pool.
    Comparison is lowercase after model normalization, so MorphGNT movable-nu
    parentheses and a sentence-initial capital still count as the same tokens.
    """
    blocked: set[str] = set()
    for verse in holdout_verses:
        normalized = normalize_greek_for_model(verse, unaccented=unaccented)
        if normalized:
            blocked.update(greek_token_ngrams(normalized.lower()))
    seen = set(seen_exact)
    kept: list[str] = []
    for _uid, raw in sentences:
        greek = normalize_greek_for_model(raw, unaccented=unaccented)
        if not greek or len(greek.split()) < 5 or greek in seen:
            continue
        if shares_greek_token_ngram(greek.lower(), blocked):
            continue
        seen.add(greek)
        kept.append(greek)
    return kept


def build_parallel_corpus_bundle(
    *,
    downloads_dir: Path,
    split_path: Path,
    coptic_analyzer: CopticAnalyzer,
    ccl_lexicon: CclLexicon,
    exclude_synoptic_thomas: bool = False,
    recover_ot_versification: bool = False,
    greekify: bool = False,
    unaccented_target: bool = False,
    include_apophthegmata: bool = False,
    include_patristics: bool = False,
    include_gloss_bridge: bool = False,
    target_tag: str | None = None,
) -> ParallelCorpusBundle:
    """Load leak-free NT train/dev, OT x Swete, Swete mono, CCL and v3 corpora.

    Patristic sentences that share a Greek 5-gram with an NT holdout verse are
    omitted. Exact-string dedupe cannot see those verses.
    """
    split = load_nt_split(split_path)
    train_ids = clean_nt_train_ids(
        split, exclude_synoptic_thomas=exclude_synoptic_thomas
    )
    dev_ids = {
        uid for p in split.passages if p.role is PassageRole.DEV for uid in p.unit_ids
    }
    morphgnt_dir = downloads_dir / "morphgnt-sblgnt"
    raw_nt = pair_horner_morphgnt(
        sword_root=downloads_dir / "sword",
        morphgnt_dir=morphgnt_dir,
    )
    train_raw = [(c, g) for c, g in raw_nt if g.unit_id in train_ids]
    dev_raw = [(c, g) for c, g in raw_nt if g.unit_id in dev_ids]

    analyzed_train_c = coptic_analyzer.analyze_units([c for c, _ in train_raw])
    analyzed_dev_c = coptic_analyzer.analyze_units([c for c, _ in dev_raw])

    nt_train: list[ParallelExample] = []
    clause_aug: list[ParallelExample] = []
    all_clean_greek: list[str] = []
    all_clean_coptic_units: list[CopticUnit] = []
    coptic_lemma_freq: Counter[str] = Counter()

    def _fmt_g(u: GreekUnit) -> str:
        txt = normalize_greek_unit_target(u)
        return strip_greek_to_bare(txt) if unaccented_target else txt

    for c_u, (_, g_u) in zip(analyzed_train_c, train_raw, strict=True):
        assert_clean_training_unit(c_u.unit_id, source_id=c_u.source_id)
        assert_clean_training_unit(g_u.unit_id, source_id=g_u.source_id)
        g_txt = _fmt_g(g_u)
        if not c_u.morphemes or not g_txt:
            continue
        nt_train.append(
            ParallelExample(
                unit_id=g_u.unit_id,
                domain_tag="<DOM_NT>",
                coptic_morphemes=tuple(c_u.morphemes),
                greek_target=g_txt,
            )
        )
        all_clean_greek.append(g_txt)
        all_clean_coptic_units.append(c_u)
        for m in c_u.morphemes:
            lem = normalize_coptic(m.lemma or m.norm).replace("\u2cff", "")
            if lem:
                coptic_lemma_freq[lem] += 1
        clause_aug.extend(
            _extract_clause_subpairs(c_u, g_u, "<DOM_NT>", unaccented=unaccented_target)
        )

    nt_dev: list[ParallelExample] = []
    nt_dev_units: list[tuple[CopticUnit, GreekUnit]] = []
    for c_u, (_, g_u) in zip(analyzed_dev_c, dev_raw, strict=True):
        g_txt = _fmt_g(g_u)
        if c_u.morphemes and g_txt:
            nt_dev.append(
                ParallelExample(
                    unit_id=g_u.unit_id,
                    domain_tag="<DOM_NT>",
                    coptic_morphemes=tuple(c_u.morphemes),
                    greek_target=g_txt,
                )
            )
            nt_dev_units.append((c_u, g_u))

    # Also include MorphGNT TRAIN verses missing from Horner in monolingual Greek LM
    morphgnt_all = load_morphgnt_nt(morphgnt_dir)
    seen_nt_ids = {ex.unit_id for ex in nt_train}
    for uid, g_u in sorted(morphgnt_all.items()):
        if uid in train_ids and uid not in seen_nt_ids:
            g_txt = _fmt_g(g_u)
            if g_txt:
                all_clean_greek.append(g_txt)

    # Sahidic OT x Swete LXX (all clean, zero Thomas/NT overlap)
    ot_zip = next((downloads_dir / "sahidic-ot").rglob("*.zip"))
    swete_dir = downloads_dir / "lxx-swete"
    ot_pairs = load_ot_swete_pairs(
        tt_zip=ot_zip,
        swete_dir=swete_dir,
        recover_versification=recover_ot_versification,
    )
    ot_train: list[ParallelExample] = []
    aligned_ot_ids: set[str] = set()
    for c_u, g_u in ot_pairs:
        assert_clean_training_unit(c_u.unit_id, source_id=c_u.source_id)
        assert_clean_training_unit(g_u.unit_id, source_id=g_u.source_id)
        g_txt = _fmt_g(g_u)
        if not c_u.morphemes or not g_txt:
            continue
        aligned_ot_ids.add(g_u.unit_id)
        ot_train.append(
            ParallelExample(
                unit_id=g_u.unit_id,
                domain_tag="<DOM_OT>",
                coptic_morphemes=tuple(c_u.morphemes),
                greek_target=g_txt,
            )
        )
        all_clean_coptic_units.append(c_u)
        for m in c_u.morphemes:
            lem = normalize_coptic(m.lemma or m.norm).replace("\u2cff", "")
            if lem:
                coptic_lemma_freq[lem] += 1
        clause_aug.extend(
            _extract_clause_subpairs(c_u, g_u, "<DOM_OT>", unaccented=unaccented_target)
        )

    # All Swete LXX verses (aligned -> clean_greek, unaligned -> back-translation pool)
    sw_data = swete_dir / "data" if (swete_dir / "data").is_dir() else swete_dir
    swete_unaligned_greek: list[str] = []
    seen_swete: set[str] = set()
    for p in sorted(sw_data.glob("*.txt")):
        parsed = parse_swete_file(p.read_text(encoding="utf-8"))
        for v_key, words in parsed.items():
            g_txt = normalize_greek_for_model(
                " ".join(words), unaccented=unaccented_target
            )
            if not g_txt or len(g_txt.split()) < 3:
                continue
            all_clean_greek.append(g_txt)
            if g_txt not in seen_swete and v_key not in aligned_ot_ids:
                seen_swete.add(g_txt)
                swete_unaligned_greek.append(g_txt)

    # Deduplicate unaligned Swete verses against aligned OT Greek targets
    aligned_g_set = {ex.greek_target for ex in ot_train}
    swete_unaligned_greek = [g for g in swete_unaligned_greek if g not in aligned_g_set]

    # Optional v3: Patristic & 1st-c. Koine prose (Apostolic Fathers, Epictetus, Philo).
    # Exact-string dedupe never sees holdout Greek. Drop shared token 5-grams.
    patristic_greek: list[str] = []
    if include_patristics:
        holdout_verses: list[str] = []
        for passage in split.passages:
            if passage.role is not PassageRole.HOLDOUT:
                continue
            for uid in passage.unit_ids:
                unit = morphgnt_all.get(uid)
                if unit is None:
                    continue
                text = _fmt_g(unit)
                if text:
                    holdout_verses.append(text)
        patristic_greek = select_patristic_training_sentences(
            load_all_patristic_koine_sentences(downloads_dir),
            seen_exact=all_clean_greek,
            holdout_verses=holdout_verses,
            unaccented=unaccented_target,
        )
        all_clean_greek.extend(patristic_greek)

    # Optional v3: Sahidic Apophthegmata Patrum sayings + English translations
    apop_coptic_units: list[CopticUnit] = []
    apop_eng_pairs: list[tuple[CopticUnit, str]] = []
    ap_root = downloads_dir / "apophthegmata-patrum"
    if include_apophthegmata and ap_root.is_dir():
        for ap_u in load_apophthegmata_corpus(ap_root):
            if not ap_u.coptic.morphemes:
                continue
            apop_coptic_units.append(ap_u.coptic)
            all_clean_coptic_units.append(ap_u.coptic)
            if ap_u.english:
                apop_eng_pairs.append((ap_u.coptic, ap_u.english))

    # CCL/DDGLC short lexical training examples (guarantees loanword vocab coverage)
    lex_aug: list[tuple[str, str]] = []
    seen_lex: set[tuple[str, str]] = set()
    tgt_prefix = f"{target_tag} " if target_tag else ""
    for c_key, entries in sorted(ccl_lexicon._loanwords.items()):
        if not entries:
            continue
        ent = entries[0]
        c_norm = normalize_coptic(c_key)
        c_disp = greekify_coptic(c_norm) if greekify else c_norm
        g_norm = normalize_greek_for_model(
            ent.greek_lemma, unaccented=unaccented_target
        )
        if not c_disp or not g_norm or (c_disp, g_norm) in seen_lex:
            continue
        seen_lex.add((c_disp, g_norm))
        pos_tag = (
            "<POS_V>"
            if ent.greek_pos == "V-"
            else "<POS_ADV>"
            if ent.greek_pos == "D-"
            else "<POS_N>"
        )
        lex_aug.append(
            (f"<DOM_LEX> {tgt_prefix}{c_disp} {pos_tag} <LOAN> {g_norm}", g_norm)
        )

    # Optional v3: Native Egyptian CCL x Strong's Greek gloss bridge
    native_gloss_hints: dict[str, str] = {}
    grc_eng_pairs: list[tuple[str, str]] = []
    strongs_xml = downloads_dir / "strongs-greek" / STRONGS_XML_FILENAME
    ccl_xml_cand = (
        downloads_dir / "ccl-ddglc" / "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
    )
    if include_gloss_bridge and strongs_xml.is_file() and ccl_xml_cand.is_file():
        bridge = build_ccl_strongs_bridge(ccl_xml=ccl_xml_cand, strongs_xml=strongs_xml)
        for gl, en_gloss in bridge.grc_eng_pairs:
            g_norm = normalize_greek_for_model(gl, unaccented=unaccented_target)
            if g_norm and en_gloss:
                grc_eng_pairs.append((g_norm, en_gloss))
        for m in bridge.native_pairs:
            c_norm = normalize_coptic(m.coptic_lemma)
            c_disp = greekify_coptic(c_norm) if greekify else c_norm
            g_norm = normalize_greek_for_model(
                m.greek_lemma, unaccented=unaccented_target
            )
            if not c_disp or not g_norm:
                continue
            # Rare native Coptic lemmas (< 10 hits in NT+OT) receive <GLOSS> hint
            if (
                m.score >= 0.45
                and coptic_lemma_freq.get(c_norm, 0) < 10
                and c_norm not in native_gloss_hints
            ):
                native_gloss_hints[c_norm] = g_norm
            if (c_disp, g_norm) not in seen_lex:
                seen_lex.add((c_disp, g_norm))
                pos_tag = (
                    "<POS_V>"
                    if m.greek_pos == "V-"
                    else "<POS_ADV>"
                    if m.greek_pos == "D-"
                    else "<POS_N>"
                )
                lex_aug.append(
                    (
                        f"<DOM_LEX> {tgt_prefix}{c_disp} {pos_tag} <GLOSS> {g_norm}",
                        g_norm,
                    )
                )

    return ParallelCorpusBundle(
        nt_train=nt_train,
        nt_dev=nt_dev,
        nt_dev_units=nt_dev_units,
        ot_train=ot_train,
        clause_aug=clause_aug,
        lex_aug=lex_aug,
        swete_unaligned_greek=swete_unaligned_greek,
        all_clean_greek=all_clean_greek,
        all_clean_coptic_units=all_clean_coptic_units,
        apop_coptic_units=apop_coptic_units,
        apop_eng_pairs=apop_eng_pairs,
        patristic_greek=patristic_greek,
        grc_eng_pairs=grc_eng_pairs,
        native_gloss_hints=native_gloss_hints,
    )
