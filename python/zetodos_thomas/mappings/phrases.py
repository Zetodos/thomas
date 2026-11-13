# SPDX-License-Identifier: Apache-2.0
"""Extract minimal statistical phrase/construction mappings (JAK-38).

Train-split only. Dual Phase-2 aligners (eflomal + fast_align) stay active.
Thomas-7 / P.Oxy. 654 never feed observations. All rows are labeled
``evidence_class=statistical`` — never independently verified.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zetodos_thomas.align.bitext import build_bitext
from zetodos_thomas.align.segment_coptic import SEGMENTED_ID_SCHEME_VERSION
from zetodos_thomas.align.splits import PilotSplits, SplitRole
from zetodos_thomas.integrations.aligners.types import AlignmentLinks, BitextSentence
from zetodos_thomas.lexicon.candidates import (
    DEFAULT_PHASE2_ALIGNERS,
    FORBIDDEN_UNIT_MARKERS,
    load_compact_alignment_artifact,
)
from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION, parallel_id, parse_token_id
from zetodos_thomas.schemas.lexicon import SplitMembership
from zetodos_thomas.schemas.mappings import (
    ConstructionExample,
    ConstructionFamily,
    EvidenceClass,
    SpanRef,
    StatisticalConstruction,
    StatisticalConstructionTable,
)
from zetodos_thomas.schemas.source import PilotCorpus

MAX_EXAMPLES = 3
MIN_DUAL_UNITS = 2
# Dual-aligner coincidence on a single verse is not enough to promote a
# decoder-facing construction — especially for verbal morphology, where
# bound-group peels invent false aux+verb pairs (Jake review on JAK-38).
MIN_DUAL_UNITS_BY_FAMILY = {
    ConstructionFamily.NEGATION: 1,
    ConstructionFamily.VERBAL: 2,
    ConstructionFamily.ARTICLE: 2,
    ConstructionFamily.PRONOUN: 2,
}
MAX_PER_FAMILY = 12
# Verses whose dual-aligner "verbal" hits are known false morphology and must
# never be exported as reusable verbal constructions.
VERBAL_FALSE_POSITIVE_UNITS = frozenset(
    {
        "matthew.5.40",  # ⲛⲧⲉ + ⲕϣⲧⲏⲛ ← possessive garment, not aux+verb
        "matthew.5.48",  # ⲛⲧⲉ + ⲗⲓⲟⲥ ← split adjective ⲛⲧⲉⲗⲓⲟⲥ
        "john.1.23",  # ⲛⲧⲉ + ϩⲓ ← not a credible verbal boundary
    }
)
# Conservative whole-lexeme allowlist for DET + N promotions. Dual-aligner
# agreement alone does not prove the Coptic second token is a noun stem —
# fused peels (ⲡⲉⲡⲣⲟⲫⲏⲧⲏⲥ → ⲡ+ⲉⲡ; ⲧⲡⲉⲧⲣⲁ → ⲧ+ⲡⲉⲧ) and non-article
# constructions (ⲟⲩⲟⲛ … ⲛⲓⲙ → ⲛ+ⲓⲙ) must not become family=article rows.
ARTICLE_NOUN_ALLOWLIST = frozenset(
    {
        "ⲟⲩⲟⲓⲛ",  # light
        "ϣⲁϫⲉ",  # word
        "ⲛⲟⲩⲧⲉ",  # God
        "ⲭⲥ",  # Χριστός nomen sacrum
        "ⲣⲱⲙⲉ",  # man (protected stem)
        "ⲥϩⲓⲙⲉ",  # woman (protected stem)
    }
)
# Interrogative / indefinite remnants that are never DET+N heads.
ARTICLE_NON_NOUN_SECONDS = frozenset(
    {
        "ⲓⲙ",  # from ⲛⲓⲙ 'who/everyone'
        "ⲁϣ",
        "ⲟⲩ",
    }
)
GREEK_PARTICLES = frozenset(
    {
        "δε",
        "γαρ",
        "μεν",
        "αλλα",
        "και",
        "τε",
        "ουν",
        "γε",
        "δη",
        "αν",
    }
)
GREEK_ART = frozenset(
    {
        "ο",
        "η",
        "το",
        "τον",
        "την",
        "του",
        "της",
        "τω",
        "τη",
        "οι",
        "αι",
        "τα",
        "των",
        "τοις",
        "ταις",
        "τους",
        "τας",
    }
)
GREEK_NEG = frozenset({"ου", "ουκ", "ουχ", "μη", "μηδε", "ουδε", "ουτε", "μητε"})
GREEK_PRON_STEMS = (
    "εγω",
    "συ",
    "σοι",
    "σε",
    "μοι",
    "με",
    "ημεις",
    "υμεις",
    "ημιν",
    "υμιν",
    "υμας",
    "αυτ",
    "ουτ",
    "ταυτ",
    "εκειν",
)
COPTIC_DET = frozenset({"ⲡ", "ⲧ", "ⲛ", "ⲡⲉ", "ⲧⲉ", "ⲛⲉ"})
COPTIC_INDEF = frozenset({"ⲟⲩ", "ϩⲉⲛ"})
# Bound-group splits of Greek particles must not look like ⲟⲩ + noun.
COPTIC_PARTICLE_LOANS = frozenset(
    {"ⲇⲉ", "ⲅⲁⲣ", "ⲙⲉⲛ", "ⲁⲗⲗⲁ", "ⲟⲩⲇⲉ", "ⲕⲁⲓ", "ⲟⲩⲛ", "ⲅⲉ"}
)
COPTIC_PRON = frozenset(
    {
        "ⲁⲛⲟⲕ",
        "ⲛⲧⲟⲕ",
        "ⲛⲧⲟ",
        "ⲛⲧⲟϥ",
        "ⲛⲧⲟⲥ",
        "ⲁⲛⲟⲛ",
        "ⲛⲧⲱⲧⲛ",
        "ⲛⲧⲟⲟⲩ",
        "ⲡⲁⲓ",
        "ⲧⲁⲓ",
        "ⲛⲁⲓ",
    }
)
COPTIC_PRON_CLITIC = frozenset({"ⲕ", "ϥ", "ⲥ", "ⲟⲩ", "ⲧⲛ"})
# Pilot sayings motivate priorities only — never as training Greek.
PILOT_PRIORITY = {
    ConstructionFamily.NEGATION: (
        "Pilot 17 uses ⲙⲡⲉ… negation; prioritize NT-learned NEG templates only "
        "(no P.Oxy. 654 / Thomas-7 Greek)."
    ),
    ConstructionFamily.ARTICLE: (
        "Pilot 7/30/77 use definite articles (ⲡ…); article+noun from NT train."
    ),
    ConstructionFamily.PRONOUN: (
        "Pilot 30/77 use ⲁⲛⲟⲕ; independent pronouns from NT train."
    ),
    ConstructionFamily.VERBAL: (
        "Pilot 7/17/77 use future ⲛⲁ; verbal aux patterns from NT train."
    ),
}


@dataclass
class _Hit:
    """One aligned construction observation."""

    family: ConstructionFamily
    coptic: tuple[str, ...]
    greek: tuple[str, ...]
    unit_id: str
    aligner: str
    coptic_token_ids: tuple[str, ...]
    greek_token_ids: tuple[str, ...]


@dataclass
class _Accum:
    """Aggregate one statistical construction pattern."""

    family: ConstructionFamily
    per_aligner: dict[str, set[tuple[str, tuple[str, ...], tuple[str, ...]]]] = field(
        default_factory=dict
    )
    units: set[str] = field(default_factory=set)
    examples: list[_Hit] = field(default_factory=list)


def default_mappings_dir(repo_root: Path) -> Path:
    """``data/sources/mappings`` under the checkout."""
    return repo_root / "data" / "sources" / "mappings"


def default_statistical_constructions_path(repo_root: Path) -> Path:
    """Committed statistical construction table path."""
    return default_mappings_dir(repo_root) / "statistical-phrase-constructions-v1.json"


def _clean_greek(token: str) -> str:
    return token.strip(".,;:·̈\"'()[]")


def _is_greek_pron(token: str) -> bool:
    t = _clean_greek(token)
    return any(t == stem or t.startswith(stem) for stem in GREEK_PRON_STEMS)


def _looks_greek_verbal(tokens: Sequence[str]) -> bool:
    """Heuristic: at least one non-function Greek token that could be verbal."""
    for raw in tokens:
        g = _clean_greek(raw)
        if not g:
            continue
        if (
            g in GREEK_ART
            or g in GREEK_NEG
            or g in GREEK_PARTICLES
            or _is_greek_pron(g)
        ):
            continue
        # Infinitives / common finite endings, or a contentful stem.
        if g.endswith(("ειν", "σθαι", "ναι", "σαι", "αι")):
            return True
        if len(g) >= 5:
            return True
    return False


def _coptic_second_looks_nonverbal(aux: str, second: str) -> bool:
    """Reject peeled noun/adjective fragments after a false ⲛⲧⲉ/ϣⲁ split."""
    if aux == "ⲛⲧⲉ" and (
        second.startswith(("ⲕ", "ϥ", "ⲥ", "ⲧⲛ"))  # possessive clitic peels
        or second in {"ϩⲓ", "ⲗⲓⲟⲥ", "ⲕϣⲧⲏⲛ"}
        or (len(second) <= 4 and not second.startswith(("ⲣ", "ⲥⲱ", "ϫ", "ϯ")))
    ):
        return True
    if aux == "ϣⲁ" and len(second) <= 2:
        return True
    return False


def _normalize_coptic_lexeme(token: str) -> str:
    """NFC, strip combining marks and trailing punctuation for allowlist checks."""
    nfd = unicodedata.normalize("NFD", token.strip())
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped).strip(".,;:·̈\"'()[]")


def _coptic_article_noun_ok(second: str) -> bool:
    """True only when the DET-following token is a whole allowlisted noun lexeme.

    Mis-segmented fragments (ⲉⲡ, ⲡⲉⲧ) and non-nouns (ⲓⲙ) fail even when both
    aligners agree on the Greek article+content span.
    """
    stem = _normalize_coptic_lexeme(second)
    if not stem or stem in ARTICLE_NON_NOUN_SECONDS:
        return False
    return stem in ARTICLE_NOUN_ALLOWLIST


def _assert_safe_unit(
    unit_id: str,
    *,
    allowed_unit_ids: set[str] | None = None,
) -> None:
    lowered = unit_id.lower()
    for marker in FORBIDDEN_UNIT_MARKERS:
        if marker in lowered:
            msg = f"forbidden unit id in construction observations: {unit_id}"
            raise ValueError(msg)
    if allowed_unit_ids is not None and unit_id not in allowed_unit_ids:
        msg = f"unit {unit_id!r} is not in authoritative train membership"
        raise ValueError(msg)


def _slug_tokens(tokens: Sequence[str], *, limit: int = 4) -> str:
    parts: list[str] = []
    for tok in tokens[:limit]:
        cleaned = re.sub(r"[^0-9A-Za-zⲀ-⳿α-ωΑ-Ω]+", "", tok)
        parts.append(cleaned or "x")
    return "-".join(parts) or "span"


def _mint_span(
    token_ids: Sequence[str],
    surfaces: Sequence[str],
    *,
    local_slug: str,
) -> SpanRef:
    if not token_ids:
        msg = "span requires token_ids"
        raise ValueError(msg)
    parsed = parse_token_id(token_ids[0])
    scheme = parsed.scheme_version
    # Span scheme follows the token layer (v1 whitespace vs v2 bound-group).
    span = f"v{scheme}:{parsed.source_id}/{parsed.unit_id}/s/{local_slug}"
    return SpanRef(
        span_id=span,
        source_id=parsed.source_id,
        unit_id=parsed.unit_id,
        token_ids=list(token_ids),
        surfaces=list(surfaces),
    )


def _add_hit(
    buckets: dict[tuple[str, tuple[str, ...], tuple[str, ...]], _Accum],
    hit: _Hit,
) -> None:
    key = (hit.family.value, hit.coptic, hit.greek)
    bucket = buckets.setdefault(
        key,
        _Accum(family=hit.family),
    )
    occ = (hit.unit_id, hit.coptic_token_ids, hit.greek_token_ids)
    bucket.per_aligner.setdefault(hit.aligner, set()).add(occ)
    bucket.units.add(hit.unit_id)
    if len(bucket.examples) < MAX_EXAMPLES * 2:
        for existing in bucket.examples:
            if (
                existing.unit_id,
                existing.coptic_token_ids,
                existing.greek_token_ids,
            ) == (hit.unit_id, hit.coptic_token_ids, hit.greek_token_ids):
                return
        bucket.examples.append(hit)


def _src_trg_maps(
    pairs: Sequence[tuple[int, int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    src_to_trg: dict[int, list[int]] = defaultdict(list)
    trg_to_src: dict[int, list[int]] = defaultdict(list)
    for i, j in pairs:
        if i < 0 or j < 0:
            continue
        src_to_trg[i].append(j)
        trg_to_src[j].append(i)
    return src_to_trg, trg_to_src


def _extract_from_sentence(
    *,
    aligner: str,
    sentence: BitextSentence,
    pairs: Sequence[tuple[int, int]],
) -> Iterable[_Hit]:
    src = sentence.source_tokens
    trg = sentence.target_tokens
    src_ids = sentence.source_token_ids
    trg_ids = sentence.target_token_ids
    src_to_trg, _trg_to_src = _src_trg_maps(pairs)

    def aligned(idxs: Sequence[int]) -> bool:
        return all(src_to_trg.get(i) for i in idxs)

    def greek_span(idxs: Sequence[int]) -> tuple[int, int]:
        js = sorted({j for i in idxs for j in src_to_trg[i]})
        return min(js), max(js)

    # Articles: DET + whole-lexeme N with Greek article in target span.
    # Dual-aligner agreement is not proof of a valid article+noun construction.
    for i in range(len(src) - 1):
        if src[i] not in COPTIC_DET | COPTIC_INDEF:
            continue
        if src[i + 1] in COPTIC_PARTICLE_LOANS | COPTIC_DET | COPTIC_INDEF:
            continue
        if not _coptic_article_noun_ok(src[i + 1]):
            continue
        if not aligned((i, i + 1)):
            continue
        lo, hi = greek_span((i, i + 1))
        if hi - lo > 3:
            continue
        g_words = [_clean_greek(trg[j]) for j in range(lo, hi + 1)]
        if src[i] in COPTIC_DET and not any(g in GREEK_ART for g in g_words):
            continue
        if src[i] in COPTIC_INDEF:
            # Indefinite often lacks a Greek article; reject particle/neg/wh noise.
            if (hi - lo) > 2:
                continue
            if any(
                g in GREEK_NEG or g in {"δε", "γαρ", "μεν", "αλλα", "τι", "τις"}
                for g in g_words
            ):
                continue
            if any(g in GREEK_ART for g in g_words):
                # ⲟⲩ + N rarely aligns to a Greek article phrase.
                continue
        yield _Hit(
            family=ConstructionFamily.ARTICLE,
            coptic=tuple(src[i : i + 2]),
            greek=tuple(trg[lo : hi + 1]),
            unit_id=sentence.unit_id,
            aligner=aligner,
            coptic_token_ids=tuple(src_ids[i : i + 2]),
            greek_token_ids=tuple(trg_ids[lo : hi + 1]),
        )

    # Negation: ⲁⲛ ↔ Greek neg; ⲙⲡⲉ/ⲧⲙ + verb; ⲛ…ⲁⲛ circumfix.
    for i, tok in enumerate(src):
        if tok == "ⲁⲛ" and src_to_trg.get(i):
            js = sorted(src_to_trg[i])
            g_words = [_clean_greek(trg[j]) for j in js]
            if any(g in GREEK_NEG for g in g_words):
                yield _Hit(
                    family=ConstructionFamily.NEGATION,
                    coptic=("ⲁⲛ",),
                    greek=tuple(trg[j] for j in js),
                    unit_id=sentence.unit_id,
                    aligner=aligner,
                    coptic_token_ids=(src_ids[i],),
                    greek_token_ids=tuple(trg_ids[j] for j in js),
                )
                if i > 0 and src_to_trg.get(i - 1):
                    lo, hi = greek_span((i - 1, i))
                    yield _Hit(
                        family=ConstructionFamily.NEGATION,
                        coptic=tuple(src[i - 1 : i + 1]),
                        greek=tuple(trg[lo : hi + 1]),
                        unit_id=sentence.unit_id,
                        aligner=aligner,
                        coptic_token_ids=tuple(src_ids[i - 1 : i + 1]),
                        greek_token_ids=tuple(trg_ids[lo : hi + 1]),
                    )
        if tok in {"ⲙⲡⲉ", "ⲧⲙ"} and i + 1 < len(src) and aligned((i, i + 1)):
            lo, hi = greek_span((i, i + 1))
            g_words = [_clean_greek(trg[j]) for j in range(lo, hi + 1)]
            if hi - lo <= 3 and any(g in GREEK_NEG for g in g_words):
                yield _Hit(
                    family=ConstructionFamily.NEGATION,
                    coptic=tuple(src[i : i + 2]),
                    greek=tuple(trg[lo : hi + 1]),
                    unit_id=sentence.unit_id,
                    aligner=aligner,
                    coptic_token_ids=tuple(src_ids[i : i + 2]),
                    greek_token_ids=tuple(trg_ids[lo : hi + 1]),
                )
        if tok == "ⲛ":
            for k in range(i + 1, min(i + 5, len(src))):
                if src[k] != "ⲁⲛ":
                    continue
                if not aligned((i, k)):
                    break
                lo, hi = greek_span((i, k))
                g_words = [_clean_greek(trg[j]) for j in range(lo, hi + 1)]
                if any(g in GREEK_NEG for g in g_words):
                    yield _Hit(
                        family=ConstructionFamily.NEGATION,
                        coptic=tuple(src[i : k + 1]),
                        greek=tuple(trg[lo : hi + 1]),
                        unit_id=sentence.unit_id,
                        aligner=aligner,
                        coptic_token_ids=tuple(src_ids[i : k + 1]),
                        greek_token_ids=tuple(trg_ids[lo : hi + 1]),
                    )
                break

    # Pronouns (incl. segmented ⲛⲧⲟ + clitic).
    for i, tok in enumerate(src):
        if tok not in COPTIC_PRON:
            continue
        end = i
        if tok == "ⲛⲧⲟ" and i + 1 < len(src) and src[i + 1] in COPTIC_PRON_CLITIC:
            end = i + 1
        idxs = list(range(i, end + 1))
        if not aligned(idxs):
            continue
        lo, hi = greek_span(idxs)
        g_words = [_clean_greek(trg[j]) for j in range(lo, hi + 1)]
        if not any(_is_greek_pron(g) for g in g_words):
            continue
        yield _Hit(
            family=ConstructionFamily.PRONOUN,
            coptic=tuple(src[i : end + 1]),
            greek=tuple(trg[lo : hi + 1]),
            unit_id=sentence.unit_id,
            aligner=aligner,
            coptic_token_ids=tuple(src_ids[i : end + 1]),
            greek_token_ids=tuple(trg_ids[lo : hi + 1]),
        )

    # Verbal auxiliaries: ⲛⲁ / ϣⲁ / ⲛⲧⲉⲣⲉ / ⲛⲧⲉ + verb (bare ⲁ is too noisy).
    # Dual-aligner agreement alone does not validate grammatical categories.
    if sentence.unit_id in VERBAL_FALSE_POSITIVE_UNITS:
        pass  # never emit verbal hits for known false-morphology verses
    else:
        for i, tok in enumerate(src):
            if tok not in {"ⲛⲁ", "ϣⲁ", "ⲛⲧⲉⲣⲉ", "ⲛⲧⲉ"} or i + 1 >= len(src):
                continue
            second = src[i + 1]
            if _coptic_second_looks_nonverbal(tok, second):
                continue
            if (
                second
                in COPTIC_DET | COPTIC_INDEF | COPTIC_PARTICLE_LOANS | COPTIC_PRON
            ):
                continue
            if not aligned((i, i + 1)):
                continue
            lo, hi = greek_span((i, i + 1))
            if hi - lo > 3:
                continue
            g_tuple = tuple(trg[lo : hi + 1])
            g_words = [_clean_greek(t) for t in g_tuple]
            if tok == "ⲛⲧⲉⲣⲉ" and "οτε" not in g_words:
                continue
            if not _looks_greek_verbal(g_tuple):
                continue
            # Possessive/article Greek dominating the span → not aux+verb.
            content = [
                g
                for g in g_words
                if g
                and g not in GREEK_ART
                and g not in GREEK_NEG
                and g not in GREEK_PARTICLES
                and not _is_greek_pron(g)
            ]
            if not content:
                continue
            yield _Hit(
                family=ConstructionFamily.VERBAL,
                coptic=tuple(src[i : i + 2]),
                greek=g_tuple,
                unit_id=sentence.unit_id,
                aligner=aligner,
                coptic_token_ids=tuple(src_ids[i : i + 2]),
                greek_token_ids=tuple(trg_ids[lo : hi + 1]),
            )


def accumulate_construction_observations(
    aligner_links: Mapping[str, AlignmentLinks],
    *,
    allowed_unit_ids: set[str] | None = None,
) -> dict[tuple[str, tuple[str, ...], tuple[str, ...]], _Accum]:
    """Aggregate template-matched constructions across Phase-2 aligners."""
    buckets: dict[tuple[str, tuple[str, ...], tuple[str, ...]], _Accum] = {}
    for aligner, links in aligner_links.items():
        by_pairs = {a.unit_id: a.pairs for a in links.symmetrized}
        for sentence in links.sentences:
            if sentence.split_role != SplitRole.TRAIN.value:
                msg = (
                    "statistical constructions require train-only links; "
                    f"got split_role={sentence.split_role!r} for {sentence.unit_id}"
                )
                raise ValueError(msg)
            _assert_safe_unit(sentence.unit_id, allowed_unit_ids=allowed_unit_ids)
            pairs = tuple(dict.fromkeys(by_pairs.get(sentence.unit_id, ())))
            for hit in _extract_from_sentence(
                aligner=aligner, sentence=sentence, pairs=pairs
            ):
                _add_hit(buckets, hit)
    return buckets


def _construction_id(
    family: ConstructionFamily,
    coptic: Sequence[str],
    greek: Sequence[str],
) -> str:
    return f"stat-{family.value}-{_slug_tokens(coptic)}--{_slug_tokens(greek)}"


def build_statistical_construction_table(
    aligner_links: Mapping[str, AlignmentLinks],
    *,
    provenance: Mapping[str, Any] | None = None,
    allowed_unit_ids: set[str] | None = None,
    max_per_family: int = MAX_PER_FAMILY,
    min_dual_units: int = MIN_DUAL_UNITS,
) -> StatisticalConstructionTable:
    """Select a minimal dual-aligner construction set, labeled statistical."""
    required = {a.value for a in DEFAULT_PHASE2_ALIGNERS}
    present = set(aligner_links)
    if present != required:
        msg = (
            "JAK-38 keep-both requires exactly "
            f"{sorted(required)}; got {sorted(present)}"
        )
        raise ValueError(msg)

    buckets = accumulate_construction_observations(
        aligner_links, allowed_unit_ids=allowed_unit_ids
    )
    by_family: dict[ConstructionFamily, list[tuple[int, int, _Accum, tuple]]] = {
        f: [] for f in ConstructionFamily if f is not ConstructionFamily.OTHER
    }
    for key, accum in buckets.items():
        family_s, coptic, greek = key
        family = ConstructionFamily(family_s)
        aligners = [a for a, occ in accum.per_aligner.items() if occ]
        if set(aligners) != required:
            continue
        family_min = MIN_DUAL_UNITS_BY_FAMILY.get(family, min_dual_units)
        if len(accum.units) < family_min:
            continue
        # Union observation count across aligner occurrence keys.
        union: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
        for occ in accum.per_aligner.values():
            union |= occ
        by_family[family].append((len(accum.units), len(union), accum, key))

    constructions: list[StatisticalConstruction] = []
    for family, rows in by_family.items():
        rows.sort(key=lambda r: (-r[0], -r[1], r[3][1], r[3][2]))
        for _n_units, _n_obs, accum, key in rows[:max_per_family]:
            _family_s, coptic, greek = key
            per_counts = {
                name: len(occ) for name, occ in sorted(accum.per_aligner.items())
            }
            union: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
            for occ in accum.per_aligner.values():
                union |= occ
            cid = _construction_id(family, coptic, greek)
            examples: list[ConstructionExample] = []
            seen_coords: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
            # Prefer examples attested by both aligners when possible.
            for hit in accum.examples:
                coord = (hit.unit_id, hit.coptic_token_ids, hit.greek_token_ids)
                if coord in seen_coords:
                    continue
                # Which aligners saw this coordinate?
                aligners = [
                    name for name, occs in accum.per_aligner.items() if coord in occs
                ]
                if not aligners:
                    aligners = [hit.aligner]
                local = (
                    f"stat-{family.value}-{_slug_tokens(coptic)}-"
                    f"{hit.unit_id.replace('.', '-')}-{len(examples)}"
                )
                c_span = _mint_span(
                    hit.coptic_token_ids,
                    hit.coptic,
                    local_slug=local + "-cop",
                )
                g_span = _mint_span(
                    hit.greek_token_ids,
                    hit.greek,
                    local_slug=local + "-grk",
                )
                examples.append(
                    ConstructionExample(
                        unit_id=hit.unit_id,
                        split_role=SplitMembership.TRAIN,
                        aligners=sorted(aligners),
                        coptic_span=c_span,
                        greek_span=g_span,
                        parallel_id=parallel_id(local),
                    )
                )
                seen_coords.add(coord)
                if len(examples) >= MAX_EXAMPLES:
                    break
            constructions.append(
                StatisticalConstruction(
                    evidence_class=EvidenceClass.STATISTICAL,
                    construction_id=cid,
                    family=family,
                    coptic_pattern=list(coptic),
                    greek_pattern=list(greek),
                    observation_count=len(union),
                    per_aligner_observation_counts=per_counts,
                    contributing_aligners=sorted(per_counts),
                    source_unit_ids=sorted(accum.units),
                    split_membership=SplitMembership.TRAIN,
                    examples=examples,
                    pilot_priority_notes=PILOT_PRIORITY.get(family, ""),
                    notes=(
                        "Statistical correspondence from train symmetrized "
                        "alignments; not independently verified lexicon evidence."
                    ),
                )
            )

    family_counts = {
        f.value: sum(1 for c in constructions if c.family is f)
        for f in ConstructionFamily
        if f is not ConstructionFamily.OTHER
    }
    return StatisticalConstructionTable(
        id="statistical-phrase-constructions-v1",
        title=(
            "Minimal statistical Coptic↔Greek phrase/construction mappings "
            "(train-split; JAK-38)"
        ),
        owned_by="JAK-38",
        schema_version="1",
        evidence_class=EvidenceClass.STATISTICAL,
        id_scheme_version=ID_SCHEME_VERSION,
        segmented_id_scheme_version=SEGMENTED_ID_SCHEME_VERSION,
        constructions=constructions,
        provenance={
            "owned_by": "JAK-38",
            "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "phase2_active_aligners": [a.value for a in DEFAULT_PHASE2_ALIGNERS],
            "adopted_aligner": None,
            "jake_decision": (
                "Keep both eflomal and fast_align active for Phase 2; "
                "separate statistical correspondences from independently "
                "verified lexical evidence."
            ),
            **dict(provenance or {}),
        },
        summary={
            "construction_count": len(constructions),
            "per_family_counts": family_counts,
            "phase2_active_aligners": [a.value for a in DEFAULT_PHASE2_ALIGNERS],
            "adopted_aligner": None,
            "selection": {
                "min_dual_aligner_unit_support_default": min_dual_units,
                "min_dual_aligner_unit_support_by_family": {
                    f.value: n for f, n in MIN_DUAL_UNITS_BY_FAMILY.items()
                },
                "max_per_family": max_per_family,
                "require_both_aligners": True,
            },
        },
        notes=[
            "All rows are evidence_class=statistical (aligner-derived).",
            "Do not treat these as independently verified loanword facts.",
            "Train-split only; Thomas-7 / P.Oxy. 654 blocked.",
            "Span IDs follow JAK-32 grammar; Coptic bound-group tokens use v2.",
            "Pilot sayings 7/17/30/77 may prioritize constructions but never "
            "supply hidden Greek for training/verification shortcuts.",
            "Usable by JAK-39 decoder via construction_id + span/parallel ids.",
            "Verbal family requires ≥2 dual-aligner train units plus verbal "
            "morphology guards; one-off aux peels (e.g. Matt 5:40/5:48, "
            "John 1:23) are omitted rather than promoted as reusable mappings. "
            "If no verbal rows survive, reliable patterns are unrecovered yet.",
            "Article family requires a conservative whole-lexeme Coptic noun "
            "allowlist after DET (not mere DET+next-token+Greek-article); "
            "fused peels (John 1:23/25 ⲡ+ⲉⲡ; Matt 7:24/25 ⲧ+ⲡⲉⲧ) and "
            "non-article ⲛⲓⲙ (Luke 11:10 / Matt 7:8) are omitted.",
        ],
    )


def generate_statistical_constructions(
    *,
    corpus: PilotCorpus,
    splits: PilotSplits,
    repo_root: Path,
    link_paths: Mapping[str, Path] | None = None,
) -> StatisticalConstructionTable:
    """Load train bitext + dual aligner links → statistical construction table."""
    bitext = build_bitext(corpus, splits)
    train = bitext.sentences_for(SplitRole.TRAIN)
    if not train:
        msg = "no train bitext sentences for statistical constructions"
        raise ValueError(msg)
    allowed = set(splits.unit_ids_for(SplitRole.TRAIN))
    paths = dict(link_paths or {})
    if not paths:
        lex = repo_root / "data" / "sources" / "lexicon"
        paths = {
            "eflomal": lex / "eflomal-train-links-v1.json",
            "fast_align": lex / "fast_align-train-links-v1.json",
        }
    expected_prov = {
        "pilot_corpus_id": bitext.pilot_corpus_id,
        "split_id": bitext.split_id,
        "bitext_checksum_sha256": bitext.checksum_sha256,
        "coptic_segmentation": bitext.coptic_segmentation,
        "greek_tokenization": bitext.greek_tokenization,
    }
    aligner_links: dict[str, AlignmentLinks] = {}
    for name in ("eflomal", "fast_align"):
        path = paths.get(name)
        if path is None or not path.is_file():
            msg = f"missing train-links artifact for {name}: {path}"
            raise FileNotFoundError(msg)
        aligner_links[name] = load_compact_alignment_artifact(
            path,
            train,
            expected_aligner=name,
            expected_provenance=expected_prov,
        )
    return build_statistical_construction_table(
        aligner_links,
        provenance={
            "pilot_corpus_id": bitext.pilot_corpus_id,
            "split_id": bitext.split_id,
            "greek_source_id": bitext.greek_source_id,
            "sahidic_source_id": bitext.sahidic_source_id,
            "coptic_segmentation": bitext.coptic_segmentation,
            "greek_tokenization": bitext.greek_tokenization,
            "bitext_checksum_sha256": bitext.checksum_sha256,
            "excluded_from_training_stack": bitext.excluded_from_training_stack,
            "train_link_artifacts": {k: str(v) for k, v in paths.items()},
        },
        allowed_unit_ids=allowed,
    )


def write_statistical_construction_table(
    table: StatisticalConstructionTable,
    path: Path,
    *,
    compact: bool = True,
) -> None:
    """Serialize the statistical construction table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = table.model_dump(mode="json")
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )
    path.write_text(text + "\n", encoding="utf-8")


def load_statistical_construction_table(
    path: Path,
) -> StatisticalConstructionTable:
    """Load and validate a statistical construction JSON artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StatisticalConstructionTable.model_validate(payload)
