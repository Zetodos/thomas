# SPDX-License-Identifier: Apache-2.0
"""Lexically-constrained banked beam search, coverage scoring & span alignment.

Implements deterministic faithfulness guarantees for Coptic -> Koine decoding:
1. Dynamic Banked Lexically-Constrained Beam Search (Hokamp & Liu / Post & Vilar
   DBA-style) over SentencePiece BPE prefix tries of valid Koine inflected forms
   for explicit Greek loanwords (<LOAN>) and proper nouns (<POS_NPROP>).
2. Exact maximum bipartite matching (Kuhn) between source content Coptic
   morphemes and candidate Greek content lemmas (multiset- & synonym-aware).
3. Wu et al. (Google NMT) source-content cross-attention coverage penalty.
4. Bound-group multi-morpheme ``coptic_seqs`` attribution.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from zetodos_thomas.align.lemma_aligner import LexicalTranslationModel
from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.neural.accent_restorer import strip_greek_to_bare
from zetodos_thomas.neural.corpus import (
    normalize_greek_for_model,
    resolve_morpheme_loan_hint,
)
from zetodos_thomas.neural.tokenizer import BilingualTokenizer
from zetodos_thomas.neural.transformer import KoineSeq2SeqTransformer
from zetodos_thomas.schemas.linguistic import CopticMorpheme

CONTENT_COPTIC_POS: frozenset[str] = frozenset(
    {"N", "NPROP", "V", "VSTAT", "VBD", "VIMP", "ADV", "NUM"}
)
_HARD_LOAN_POS: frozenset[str] = frozenset({"N", "NPROP", "V", "VSTAT", "VIMP"})
# Decode-time structural constraint surfaces matching model-v1 `_clause_to_slots`:
# 1) Coptic copula (`COP` ⲡⲉ/ⲧⲉ/ⲛⲉ) -> present indicative forms of εἰμί (tense="P")
# 2) Object-resumptive relative (`CREL` without ART + PPERO) -> accusative forms of ὅς
_COPULA_EIMI_SURFACES: tuple[str, ...] = (
    "ἐστίν",
    "ἐστί",
    "εἰσίν",
    "εἰσί",
    "εἰμί",
    "εἶ",
    "ἐσμέν",
    "ἐστέ",
)
_RESUMPTIVE_HOS_ACC_SURFACES: tuple[str, ...] = (
    "ὅν",
    "ἥν",
    "ὅ",
    "οὕς",
    "ἅς",
    "ἅ",
)
# Coptic body-part nouns grammaticalized into compound prepositions when NOT
# preceded by a determiner (e.g. ⲛ-ϩⲏⲧ-ⲥ "in it", ⲥⲱⲕ ϩⲏⲧ-ϥ "lead before him",
# ⲛ-ⲧⲟⲟⲧ-ϥ "by him", ϩⲁ-ⲣⲁⲧ-ϥ "under him") — never lexical καρδία/χείρ/πούς.
COMPOUND_PREP_NOUNS: frozenset[str] = frozenset(
    {"ϩⲏⲧ", "ⲧⲟⲟⲧ", "ⲣⲁⲧ", "ϩⲣⲁ", "ϩⲏ", "ⲙⲧⲟ", "ⲣⲱ"}
)
_DETERMINER_POS: frozenset[str] = frozenset({"ART", "PPOS", "PDEM"})


def is_grammaticalized_prep_noun(morphemes: Sequence[CopticMorpheme], idx: int) -> bool:
    """Return True if ``morphemes[idx]`` is a bare compound-preposition noun."""
    m = morphemes[idx]
    if m.pos != "N":
        return False
    lem_s = normalize_coptic(m.lemma).replace("\u2cff", "")
    norm_s = normalize_coptic(m.norm).replace("\u2cff", "")
    key = lem_s or norm_s
    if key not in COMPOUND_PREP_NOUNS:
        return False
    prev_pos: str | None = None
    for j in range(idx - 1, -1, -1):
        if morphemes[j].pos != "PUNCT":
            prev_pos = morphemes[j].pos
            break
    return prev_pos not in _DETERMINER_POS


def _strip_bare(word: str) -> str:
    nfd = unicodedata.normalize("NFD", word)
    return (
        "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        .lower()
        .replace("ς", "σ")
    )


_DEMOTIC_CHARS: frozenset[str] = frozenset("ϣϥϧϩϫϭϯⳉ")
_UNACCENTED_ALLOWED_CLITICS: frozenset[str] = frozenset(
    {
        "ο",
        "η",
        "οι",
        "αι",
        "εν",
        "εκ",
        "εξ",
        "εισ",
        "ει",
        "ωσ",
        "ου",
        "ουκ",
        "ουχ",
        "τε",
        "γε",
        "μου",
        "μοι",
        "με",
        "σου",
        "σοι",
        "σε",
        "τισ",
        "τι",
        "τινοσ",
        "τινι",
        "τινα",
        "τινεσ",
        "τινων",
        "τισι",
        "τισιν",
        "τινασ",
        "πωσ",
        "που",
        "ποτε",
        "ειμι",
        "εστι",
        "εστιν",
        "εσμεν",
        "εστε",
        "εισι",
        "εισιν",
        "φησι",
        "φησιν",
    }
)


def is_bare_coptic_copy_word(word: str) -> bool:
    """Return True if a Greek token (>=3 letters) has zero polytonic diacritics."""
    nfd = unicodedata.normalize("NFD", word)
    alpha = [ch for ch in nfd if ch.isalpha()]
    if len(alpha) < 3:
        return False
    has_diac = any(unicodedata.category(ch) == "Mn" for ch in nfd)
    if has_diac:
        return False
    return _strip_bare(word) not in _UNACCENTED_ALLOWED_CLITICS


def has_local_loop_repeat(seq: list[int], next_id: int) -> bool:
    """Block immediate stuttering loops without forbidding parallel clause repeats."""
    n = len(seq)
    if n >= 2 and seq[-1] == next_id and seq[-2] == next_id:
        return True
    for span in range(2, 7):
        if n >= 2 * span - 1 and tuple(seq[-(2 * span - 1) : -(span - 1)]) == (
            *seq[-(span - 1) :],
            next_id,
        ):
            return True
    if n >= 15:
        eight = (*seq[-7:], next_id)
        for i in range(n - 7):
            if tuple(seq[i : i + 8]) == eight:
                return True
    return False


def count_spurious_adjacent_word_repeats(
    greek_words: Sequence[str],
    morphemes: Sequence[CopticMorpheme],
) -> int:
    """Count consecutive identical Greek words unsupported by source repeats."""
    if len(greek_words) < 2:
        return 0
    # Allow adjacent repeats when Coptic itself repeats a lemma within 3 morphs
    # (e.g. ⲥⲁⲃⲃⲁⲧⲟⲛ ⲛ ⲥⲁⲃⲃⲁⲧⲟⲛ, ϩⲁⲙⲏⲛ ϩⲁⲙⲏⲛ, ⲡϫⲟⲉⲓⲥ ⲡϫⲟⲉⲓⲥ).
    src_has_adj_repeat = False
    non_punct = [
        normalize_coptic(m.lemma or m.norm).replace("\u2cff", "")
        for m in morphemes
        if m.pos != "PUNCT"
    ]
    for i, lem in enumerate(non_punct):
        if len(lem) >= 2 and any(
            non_punct[j] == lem for j in range(i + 1, min(i + 4, len(non_punct)))
        ):
            src_has_adj_repeat = True
            break
    if src_has_adj_repeat:
        return 0
    dup = 0
    for i in range(1, len(greek_words)):
        w_cur = _strip_bare(greek_words[i])
        w_prev = _strip_bare(greek_words[i - 1])
        if len(w_cur) >= 3 and w_cur == w_prev:
            dup += 1
    return dup


def _has_trigram_repeat(seq: list[int], next_id: int) -> bool:
    return has_local_loop_repeat(seq, next_id)


@dataclass(frozen=True)
class ConstraintTrie:
    """Compact BPE token prefix trie for one mandatory Greek lexical constraint."""

    label: str
    lemma_bare: str
    edges: dict[int, dict[int, int]]
    terminals: frozenset[int]
    token_tuples: frozenset[tuple[int, ...]]
    bare_surfaces: frozenset[str]
    source_seq: int
    rel_pos: float = 0.5
    is_repair: bool = False
    min_step: int = 0
    max_step: int | None = None
    min_count: int = 1

    @property
    def root_children(self) -> dict[int, int]:
        return self.edges.get(0, {})

    @classmethod
    def from_surfaces(
        cls,
        *,
        label: str,
        lemma: str,
        surfaces: Sequence[str],
        tokenizer: BilingualTokenizer,
        source_seq: int,
        rel_pos: float = 0.5,
        unaccented: bool = False,
        is_repair: bool = False,
        min_step: int = 0,
        max_step: int | None = None,
        min_count: int = 1,
    ) -> ConstraintTrie | None:
        edges: dict[int, dict[int, int]] = {0: {}}
        terminals: set[int] = set()
        token_tuples: set[tuple[int, ...]] = set()
        bare_surfaces: set[str] = set()
        next_node = 1

        for raw in surfaces:
            norm = normalize_greek_for_model(raw, unaccented=unaccented)
            if not norm or " " in norm:
                continue
            ids = tokenizer.encode(norm, add_bos=False, add_eos=False)
            if not ids or len(ids) > 8:
                continue
            if any(tid == tokenizer.unk_id for tid in ids):
                continue
            token_tuples.add(tuple(ids))
            bare_surfaces.add(_strip_bare(norm))
            node = 0
            for tid in ids:
                ch = edges.setdefault(node, {})
                nxt = ch.get(tid)
                if nxt is None:
                    nxt = next_node
                    next_node += 1
                    ch[tid] = nxt
                    edges[nxt] = {}
                node = nxt
            terminals.add(node)

        if not token_tuples:
            return None
        return cls(
            label=label,
            lemma_bare=_strip_bare(lemma),
            edges=edges,
            terminals=frozenset(terminals),
            token_tuples=frozenset(token_tuples),
            bare_surfaces=frozenset(bare_surfaces),
            source_seq=source_seq,
            rel_pos=rel_pos,
            is_repair=is_repair,
            min_step=min_step,
            max_step=max_step,
            min_count=min_count,
        )


def _cop_is_nominal_predicate(morphemes: Sequence[CopticMorpheme], idx: int) -> bool:
    """Return True if ``COP`` (ⲡⲉ/ⲧⲉ/ⲛⲉ) heads a nominal clause (not imperfect)."""
    if morphemes[idx].pos != "COP":
        return False
    # Walk backward inside the same clause: if an imperfect converter (CPRET) or
    # finite non-VBD verb precedes COP, it is a verbal/cleft auxiliary, not εἰμί.
    for k in range(idx - 1, max(-1, idx - 8), -1):
        pos_k = morphemes[k].pos
        if pos_k in {"PUNCT", "CONJ", "VBD"}:
            break
        if pos_k in {"CPRET", "V", "VIMP"}:
            return False
    return True


_THIRD_PERSON_PPERO_LEMMAS: frozenset[str] = frozenset(
    {"ⲛⲧⲟϥ", "ⲛⲧⲟⲥ", "ⲛⲧⲟⲟⲩ", "ϥ", "ⲥ", "ⲟⲩ"}
)


def _crel_has_resumptive_obj(morphemes: Sequence[CopticMorpheme], idx: int) -> bool:
    """Return True if bare ``CREL`` has a 3rd-person verbal resumptive ``PPERO``."""
    if morphemes[idx].pos != "CREL":
        return False
    # Articular relatives (`ⲡⲉⲧ-`) and clefts (`ⲡⲉ ⲛⲧ-`) are not oblique ὅν clauses.
    if idx > 0 and morphemes[idx - 1].pos in {"ART", "COP"}:
        return False
    if idx + 1 >= len(morphemes):
        return False
    # Non-verbal/adjectival/demonstrative relative heads (`ⲉⲧ ⲟⲛϩ`, `ⲉⲧ ⲙⲙⲁⲩ`,
    # `ⲉⲧ ϩⲛ`, `ⲉⲧ ⲛⲁⲛⲟⲩϥ`, `ⲉⲧⲉⲩⲛⲧⲁϥ`) do not take a verbal resumptive object.
    if morphemes[idx + 1].pos in {"VSTAT", "ADV", "PREP", "VBD", "EXIST"}:
        return False
    limit = min(idx + 9, len(morphemes))
    seen_subj = False
    k = idx + 1
    while k < limit:
        pos_k = morphemes[k].pos
        if pos_k in {
            "PUNCT",
            "CONJ",
            "VBD",
            "EXIST",
            "ANEGPST",
            "CCIRC",
            "AJUS",
            "ACONJ",
            "ACOND",
            "APREC",
        }:
            return False
        if pos_k == "CREL":
            # Allow an embedded attributive `CREL + VSTAT` inside the subject NP
            # (e.g., `ⲓⲏⲥⲟⲩⲥ ⲉⲧ ⲟⲛϩ`), otherwise a new relative clause started.
            if k + 1 < limit and morphemes[k + 1].pos == "VSTAT":
                k += 2
                continue
            return False
        if pos_k in {"PPERS", "N", "NPROP", "PPERI", "PDEM"}:
            seen_subj = True
        if pos_k == "V":
            # Stop at the clause's first finite verb so we never cross into a
            # following clause; require an overt clause subject and a 3rd-person
            # direct-object suffix pronoun resuming the antecedent.
            if not seen_subj or k + 1 >= len(morphemes):
                return False
            nxt = morphemes[k + 1]
            if nxt.pos != "PPERO":
                return False
            p_lem = normalize_coptic(nxt.lemma or nxt.norm).replace("\u2cff", "")
            return p_lem in _THIRD_PERSON_PPERO_LEMMAS
        k += 1
    return False


def build_chunk_lexical_constraints(
    morphemes: Sequence[CopticMorpheme],
    *,
    ccl_lexicon: CclLexicon,
    inflector: GreekInflector,
    tokenizer: BilingualTokenizer,
    unaccented: bool = False,
    max_constraints: int | None = None,
) -> list[ConstraintTrie]:
    """Build BPE prefix tries for mandatory chunk loanwords, COP, & resumptive CREL.

    Covers:
    - Unambiguous content Greek loanwords (``N``, ``NPROP``, ``V``, ``VSTAT``,
      ``VIMP``) whose CCL Greek lemma is >= 3 chars.
    - Coptic copula ``COP`` (``ⲡⲉ``/``ⲧⲉ``/``ⲛⲉ``) -> finite ``εἰμί`` forms.
    - Bare object-resumptive ``CREL`` (``has_resumptive_obj``) -> accusative ``ὅς``.
    """
    constraints: list[ConstraintTrie] = []
    n_morphs = max(len(morphemes), 1)
    n_np = max(sum(1 for x in morphemes if x.pos != "PUNCT"), 1)
    est_tgt = max(int(n_np * 0.85), 8)
    prec_content = 0
    for m_idx, m in enumerate(morphemes):
        cur_prec_content = prec_content
        if m.pos in CONTENT_COPTIC_POS and not is_grammaticalized_prep_noun(
            morphemes, m_idx
        ):
            prec_content += 1

        rel_p = (m_idx + 0.5) / n_morphs
        min_st = (
            0
            if cur_prec_content == 0
            else max(cur_prec_content, int((rel_p - 0.16) * est_tgt))
        )

        # Decode-time structural slots from model-v1 `_clause_to_slots`:
        # nominal COP -> present εἰμί; object-resumptive CREL -> accusative ὅς.
        if m.pos == "COP" and _cop_is_nominal_predicate(morphemes, m_idx):
            trie = ConstraintTrie.from_surfaces(
                label="εἰμί",
                lemma="εἰμί",
                surfaces=_COPULA_EIMI_SURFACES,
                tokenizer=tokenizer,
                source_seq=m.seq,
                rel_pos=rel_p,
                unaccented=unaccented,
                min_step=min_st,
            )
            if trie is not None:
                constraints.append(trie)
                if max_constraints is not None and len(constraints) >= max_constraints:
                    break
            continue

        if m.pos == "CREL" and _crel_has_resumptive_obj(morphemes, m_idx):
            trie = ConstraintTrie.from_surfaces(
                label="ὅς",
                lemma="ὅς",
                surfaces=_RESUMPTIVE_HOS_ACC_SURFACES,
                tokenizer=tokenizer,
                source_seq=m.seq,
                rel_pos=rel_p,
                unaccented=unaccented,
                min_step=min_st,
            )
            if trie is not None:
                constraints.append(trie)
                if max_constraints is not None and len(constraints) >= max_constraints:
                    break
            continue

        if m.pos not in _HARD_LOAN_POS:
            continue
        g_hint = resolve_morpheme_loan_hint(m, ccl_lexicon)
        if not g_hint:
            continue
        bare_h = _strip_bare(g_hint)
        if len(bare_h) < 3:
            continue
        # Map Coptic POS to coarse Greek POS for paradigm expansion
        g_pos = (
            "V-"
            if m.pos in {"V", "VSTAT", "VIMP"}
            else "NPROP"
            if m.pos == "NPROP"
            else None
        )
        # Check CCL entry for explicit Greek POS (e.g., A- vs N- vs V-)
        c_hits = ccl_lexicon.resolve(
            norm=normalize_coptic(m.norm).replace("\u2cff", "") or m.norm,
            lemma=normalize_coptic(m.lemma).replace("\u2cff", "") or m.lemma,
            pos=m.pos,
            lang=m.lang,
        )
        if c_hits and c_hits[0].greek_pos in {"N-", "A-", "V-", "NPROP"}:
            g_pos = c_hits[0].greek_pos

        surfaces = inflector.all_inflected_forms(g_hint, pos=g_pos, max_forms=48)
        # Also include lowercase variant if lemma was capitalized (e.g. Φαρισαῖος)
        if g_hint[0].isupper():
            surfaces.extend(
                inflector.all_inflected_forms(g_hint.lower(), pos=g_pos, max_forms=16)
            )
        # Also include capitalized variant if Coptic POS is NPROP
        if m.pos == "NPROP" and g_hint[0].islower():
            cap = unicodedata.normalize("NFC", g_hint[0].upper() + g_hint[1:])
            surfaces.extend(
                inflector.all_inflected_forms(cap, pos="NPROP", max_forms=16)
            )

        trie = ConstraintTrie.from_surfaces(
            label=g_hint,
            lemma=g_hint,
            surfaces=surfaces,
            tokenizer=tokenizer,
            source_seq=m.seq,
            rel_pos=rel_p,
            unaccented=unaccented,
            min_step=min_st,
        )
        if trie is not None:
            constraints.append(trie)
            if max_constraints is not None and len(constraints) >= max_constraints:
                break
    return constraints


def _last_word_tuple(seq: list[int], word_starts: list[bool]) -> tuple[int, ...]:
    """Return the BPE token tuple of the trailing word in ``seq`` (after BOS)."""
    n = len(seq)
    if n <= 1:
        return ()
    idx = n - 1
    while idx > 1 and not word_starts[seq[idx]]:
        idx -= 1
    return tuple(seq[idx:])


def _count_trie_words(
    seq: list[int],
    token_tuples: frozenset[tuple[int, ...]],
    word_starts: list[bool],
) -> int:
    """Count completed words in ``seq`` (excluding BOS) matching ``token_tuples``."""
    n = len(seq)
    if n <= 1:
        return 0
    cnt = 0
    w_start = 1
    for i in range(2, n):
        if word_starts[seq[i]]:
            if tuple(seq[w_start:i]) in token_tuples:
                cnt += 1
            w_start = i
    if tuple(seq[w_start:n]) in token_tuples:
        cnt += 1
    return cnt


def _verified_bank(
    seq: list[int],
    mask: int,
    constraints: Sequence[ConstraintTrie],
    word_starts: list[bool],
) -> int:
    """Count satisfied constraints whose multiplicity meets ``min_count``."""
    ok = 0
    for c_idx, ct in enumerate(constraints):
        if not (mask & (1 << c_idx)):
            continue
        if ct.min_count <= 1 or (
            _count_trie_words(seq, ct.token_tuples, word_starts) >= ct.min_count
        ):
            ok += 1
    return ok


def _apply_passive_match(
    seq: list[int],
    mask: int,
    constraints: Sequence[ConstraintTrie],
    word_starts: list[bool],
    *,
    est_tgt_len: int = 16,
) -> int:
    """If the trailing completed word matches an unsatisfied constraint, set bit."""
    w_tup = _last_word_tuple(seq, word_starts)
    if not w_tup:
        return mask
    word_start_step = max(len(seq) - 1 - len(w_tup), 0)
    cur_frac = max(len(seq) - 1, 0) / max(est_tgt_len, 1)
    best_idx = -1
    best_dist = 1e9
    for c_idx, ct in enumerate(constraints):
        if (mask & (1 << c_idx)) or w_tup not in ct.token_tuples:
            continue
        # Positional repair constraints only passively match inside their step window
        if ct.is_repair:
            if word_start_step < ct.min_step:
                continue
            if ct.max_step is not None and word_start_step > ct.max_step:
                continue
        dist = abs(cur_frac - ct.rel_pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = c_idx
    if best_idx >= 0:
        mask |= 1 << best_idx
    return mask


@torch.no_grad()
def constrained_ensemble_beam_search(
    *,
    forward_models: Sequence[KoineSeq2SeqTransformer],
    tokenizer: BilingualTokenizer,
    src_ids: list[int],
    constraints: Sequence[ConstraintTrie],
    beam_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    len_alpha: float,
    device: torch.device,
    max_total_beams: int = 24,
    est_tgt_len: int | None = None,
) -> list[tuple[list[int], float]]:
    """Deterministic Banked Lexically-Constrained Ensemble Beam Search.

    Guarantees that hypotheses satisfying all ``constraints`` (e.g., mandatory
    Greek loanwords) are explored alongside unconstrained hypotheses, choosing
    the highest-probability inflected surface form in context.
    """
    n_c = len(constraints)
    full_mask = (1 << n_c) - 1
    word_starts = tokenizer.word_start_flags()

    src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
    encoded = [m.encode(src_t) for m in forward_models]
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    n_models = len(forward_models)

    # Beam item: (seq, raw_score, satisfied_mask, active_c, active_node)
    # active_c == -1 -> free decoding; active_c >= 0 -> mid-trie
    active: list[tuple[list[int], float, int, int, int]] = [([bos_id], 0.0, 0, -1, 0)]
    finished_by_bank: dict[int, list[tuple[list[int], float]]] = defaultdict(list)

    def _norm(raw_sc: float, length: int) -> float:
        return raw_sc / (max(length, 1) ** len_alpha)

    tgt_len_est = (
        est_tgt_len if est_tgt_len is not None else max(min_new_tokens * 2, 10)
    )

    def _in_force_window(c_idx: int, cur_step: int) -> bool:
        """Allow forced injection near constraint source pos (passive always on)."""
        ct = constraints[c_idx]
        lo = ct.min_step
        if lo == 0 and ct.rel_pos >= 0.08:
            lo = max(2, int((ct.rel_pos - 0.16) * tgt_len_est) - 1)
        if cur_step < lo:
            return False
        if ct.max_step is not None and cur_step > ct.max_step:
            return False
        return True

    # Precompute banned Demotic/control-tag IDs so output stays strictly Greek
    banned_set: set[int] = {pad_id, bos_id}
    if tokenizer.unk_id >= 0:
        banned_set.add(tokenizer.unk_id)
    for tid in range(tokenizer.vocab_size):
        if tid in {pad_id, bos_id, eos_id, tokenizer.unk_id}:
            continue
        p_str = tokenizer.id_to_piece(tid)
        if (p_str.startswith("<") and not p_str.startswith("<0x")) or any(
            ch in _DEMOTIC_CHARS for ch in p_str
        ):
            banned_set.add(tid)
    banned_t = torch.tensor(sorted(banned_set), dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        if not active:
            break
        cur_bsz = len(active)
        seq_len = len(active[0][0])
        tgt_t = torch.tensor(
            [item[0] for item in active], dtype=torch.long, device=device
        )

        lp_list: list[Tensor] = []
        for m, (mem, spm_mask) in zip(forward_models, encoded, strict=True):
            logits, _ = m.decode(
                tgt_t,
                mem.expand(cur_bsz, -1, -1),
                spm_mask.expand(cur_bsz, -1),
                return_cross_attn=False,
            )
            lp_list.append(F.log_softmax(logits[:, -1, :], dim=-1))

        if n_models == 1:
            step_lp = lp_list[0]
        else:
            step_lp = torch.logsumexp(torch.stack(lp_list, dim=0), dim=0) - math.log(
                n_models
            )
        step_lp[:, banned_t] = -1e9

        top_k = min(beam_size * 2, step_lp.size(-1))
        top_vals, top_idx = torch.topk(step_lp, k=top_k, dim=-1)
        top_vals_cpu = top_vals.cpu().tolist()
        top_idx_cpu = top_idx.cpu().tolist()
        step_lp_cpu = step_lp.cpu()

        # Bucket candidate continuations by bank = popcount(mask) + (0.5 if active_c>=0)
        next_free: dict[int, list[tuple[list[int], float, int, int, int]]] = (
            defaultdict(list)
        )
        next_inprog: dict[int, list[tuple[list[int], float, int, int, int]]] = (
            defaultdict(list)
        )

        rem_steps = max_new_tokens - step

        for b_i, (seq, score, mask, act_c, act_node) in enumerate(active):
            row_lp = step_lp_cpu[b_i]

            if act_c >= 0:
                ct = constraints[act_c]
                # 1. If at terminal node, complete word on word_start or EOS
                if act_node in ct.terminals:
                    done_mask = mask | (1 << act_c)
                    done_bank = done_mask.bit_count()
                    for val, tid in zip(
                        top_vals_cpu[b_i], top_idx_cpu[b_i], strict=True
                    ):
                        itid = int(tid)
                        if itid in banned_set:
                            continue
                        if itid == eos_id:
                            if seq_len >= min_new_tokens:
                                fin_bk = _verified_bank(
                                    seq, done_mask, constraints, word_starts
                                )
                                finished_by_bank[fin_bk].append(
                                    ([*seq, eos_id], _norm(score + float(val), seq_len))
                                )
                            continue
                        if not word_starts[itid]:
                            continue
                        if _has_trigram_repeat(seq, itid):
                            continue
                        next_free[done_bank].append(
                            ([*seq, itid], score + float(val), done_mask, -1, 0)
                        )
                    # Also allow immediately starting another unsatisfied constraint
                    if done_mask != full_mask and rem_steps >= 2:
                        for c_j in range(n_c):
                            if (done_mask & (1 << c_j)) or not _in_force_window(
                                c_j, step
                            ):
                                continue
                            roots = constraints[c_j].root_children
                            scored_roots = sorted(
                                (
                                    (float(row_lp[rt]), rt, nxt)
                                    for rt, nxt in roots.items()
                                ),
                                key=lambda x: -x[0],
                            )[:3]
                            for r_lp, rt, nxt_n in scored_roots:
                                if _has_trigram_repeat(seq, rt):
                                    continue
                                next_inprog[done_bank].append(
                                    ([*seq, rt], score + r_lp, done_mask, c_j, nxt_n)
                                )

                # 2. Continue deeper inside the current constraint trie
                child_map = ct.edges.get(act_node, {})
                if child_map:
                    bank_k = mask.bit_count()
                    scored_ch = sorted(
                        ((float(row_lp[t]), t, nxt) for t, nxt in child_map.items()),
                        key=lambda x: -x[0],
                    )[:4]
                    for c_lp, tid, nxt_n in scored_ch:
                        if _has_trigram_repeat(seq, tid):
                            continue
                        next_inprog[bank_k].append(
                            ([*seq, tid], score + c_lp, mask, act_c, nxt_n)
                        )
                continue

            # Free decoding state (act_c == -1)
            bank_k = mask.bit_count()
            # 1. Standard top-K unconstrained expansion
            for val, tid in zip(top_vals_cpu[b_i], top_idx_cpu[b_i], strict=True):
                itid = int(tid)
                if itid in banned_set:
                    continue
                if itid == eos_id:
                    if seq_len < min_new_tokens:
                        continue
                    end_mask = _apply_passive_match(
                        seq, mask, constraints, word_starts, est_tgt_len=tgt_len_est
                    )
                    end_bank = _verified_bank(seq, end_mask, constraints, word_starts)
                    finished_by_bank[end_bank].append(
                        ([*seq, eos_id], _norm(score + float(val), seq_len))
                    )
                    continue
                if _has_trigram_repeat(seq, itid):
                    continue
                new_mask = (
                    _apply_passive_match(
                        seq, mask, constraints, word_starts, est_tgt_len=tgt_len_est
                    )
                    if word_starts[itid]
                    else mask
                )
                next_free[new_mask.bit_count()].append(
                    ([*seq, itid], score + float(val), new_mask, -1, 0)
                )

            # 2. Forced constraint injection (entering unsatisfied constraint trie
            # via word-initial ▁ token cleanly closes the previous word!)
            if mask != full_mask and rem_steps >= 2:
                closed_mask = _apply_passive_match(
                    seq, mask, constraints, word_starts, est_tgt_len=tgt_len_est
                )
                closed_bank = closed_mask.bit_count()
                for c_j in range(n_c):
                    if (closed_mask & (1 << c_j)) or not _in_force_window(c_j, step):
                        continue
                    roots = constraints[c_j].root_children
                    scored_roots = sorted(
                        ((float(row_lp[rt]), rt, nxt) for rt, nxt in roots.items()),
                        key=lambda x: -x[0],
                    )[:4]
                    for r_lp, rt, nxt_n in scored_roots:
                        if _has_trigram_repeat(seq, rt):
                            continue
                        next_inprog[closed_bank].append(
                            ([*seq, rt], score + r_lp, closed_mask, c_j, nxt_n)
                        )

        # Balanced per-bank pruning (Post & Vilar DBA): EVERY populated bank 0..n_c
        # keeps guaranteed beam slots so early free decoding in bank 0 is NEVER
        # evicted by premature constraint injection!
        new_active: list[tuple[list[int], float, int, int, int]] = []
        seen_states: set[tuple[tuple[int, ...], int, int, int]] = set()

        def _push(
            cands: list[tuple[list[int], float, int, int, int]],
            limit: int,
            dest: list[tuple[list[int], float, int, int, int]],
            seen: set[tuple[tuple[int, ...], int, int, int]],
            *,
            max_per_parent: int = 2,
        ) -> None:
            cands.sort(key=lambda item: _norm(item[1], len(item[0]) - 1), reverse=True)
            parent_counts: dict[tuple[int, ...], int] = defaultdict(int)
            added = 0
            overflow: list[tuple[list[int], float, int, int, int]] = []
            for item in cands:
                st_key = (tuple(item[0]), item[2], item[3], item[4])
                if st_key in seen:
                    continue
                p_key = tuple(item[0][:-1])
                if parent_counts[p_key] >= max_per_parent:
                    overflow.append(item)
                    continue
                parent_counts[p_key] += 1
                seen.add(st_key)
                dest.append(item)
                added += 1
                if added >= limit:
                    return
            for item in overflow:
                if added >= limit:
                    return
                st_key = (tuple(item[0]), item[2], item[3], item[4])
                if st_key in seen:
                    continue
                seen.add(st_key)
                dest.append(item)
                added += 1

        free_quota = max(3, max_total_beams // max(n_c + 2, 2))
        inprog_quota = max(2, free_quota // 2)
        for bk in range(n_c, -1, -1):
            lim = max(free_quota, beam_size // 2) if bk in {0, n_c} else free_quota
            if next_free.get(bk):
                _push(next_free[bk], lim, new_active, seen_states, max_per_parent=2)
            if bk < n_c and next_inprog.get(bk):
                _push(
                    next_inprog[bk],
                    inprog_quota,
                    new_active,
                    seen_states,
                    max_per_parent=2,
                )

        active = new_active
        # Early exit once full-constraint bank has ample high-scoring finished hyps
        full_fin = finished_by_bank.get(n_c, [])
        if len(full_fin) >= beam_size * 2:
            best_fin = max(sc for _, sc in full_fin)
            best_act = (
                max(_norm(sc, len(s) - 1) for s, sc, _, _, _ in active)
                if active
                else -1e9
            )
            if best_act < best_fin - 0.5:
                break

    # Close any remaining active hypotheses
    for seq, sc, mask, act_c, act_node in active:
        if act_c >= 0 and act_node in constraints[act_c].terminals:
            end_mask = mask | (1 << act_c)
        elif act_c == -1:
            end_mask = _apply_passive_match(
                seq, mask, constraints, word_starts, est_tgt_len=tgt_len_est
            )
        else:
            continue
        fin_bk = _verified_bank(seq, end_mask, constraints, word_starts)
        finished_by_bank[fin_bk].append(([*seq, eos_id], _norm(sc, len(seq))))

    # Collect candidates from the full-constraint bank (n_c) when populated;
    # only fall back to lower banks if no full-constraint hypothesis finished.
    dedup: list[tuple[list[int], float]] = []
    seen_seqs: set[tuple[int, ...]] = set()
    has_full = bool(finished_by_bank.get(n_c))
    for bk in range(n_c, -1, -1):
        if has_full and bk < n_c:
            break
        hyps = finished_by_bank.get(bk, [])
        hyps.sort(key=lambda item: item[1], reverse=True)
        bk_added = 0
        bk_cap = beam_size if bk == n_c else max(2, beam_size // 2)
        for seq, sc in hyps:
            key = tuple(seq)
            if key in seen_seqs:
                continue
            seen_seqs.add(key)
            dedup.append((seq, sc))
            bk_added += 1
            if bk_added >= bk_cap or len(dedup) >= beam_size + 4:
                break
        if len(dedup) >= beam_size + 4:
            break
    return dedup


def build_coptic_content_candidate_sets(
    morphemes: Sequence[CopticMorpheme],
    *,
    ccl_lexicon: CclLexicon | None = None,
    lexical_model: LexicalTranslationModel | None = None,
    native_gloss_hints: Mapping[str, str] | None = None,
    top_k: int = 12,
) -> list[tuple[int, set[str]]]:
    """Build per-content-morpheme sets of valid bare Greek translation lemmas."""
    out: list[tuple[int, set[str]]] = []
    for m_idx, m in enumerate(morphemes):
        if m.pos not in CONTENT_COPTIC_POS:
            continue
        if is_grammaticalized_prep_noun(morphemes, m_idx):
            continue
        norm_s = normalize_coptic(m.norm).replace("\u2cff", "")
        lem_s = normalize_coptic(m.lemma).replace("\u2cff", "")
        key_lem = lem_s or norm_s
        if not key_lem:
            continue
        cands: set[str] = set()
        if ccl_lexicon is not None:
            gh = resolve_morpheme_loan_hint(m, ccl_lexicon)
            if gh:
                cands.add(_strip_bare(gh))
        if native_gloss_hints:
            for k_try in (f"{key_lem}|{m.pos}", key_lem, norm_s):
                gl = native_gloss_hints.get(k_try)
                if gl:
                    cands.add(_strip_bare(gl))
        if lexical_model is not None:
            for lc in lexical_model.lookup(
                coptic_lemma=key_lem,
                coptic_pos=m.pos,
                coptic_norm=norm_s,
                lang=m.lang,
                ccl_lexicon=ccl_lexicon,
            )[:top_k]:
                cands.add(_strip_bare(lc.greek_lemma))
            if norm_s and norm_s != key_lem:
                for lc in lexical_model.lookup(
                    coptic_lemma=norm_s,
                    coptic_pos=m.pos,
                    coptic_norm=norm_s,
                    lang=m.lang,
                )[:4]:
                    cands.add(_strip_bare(lc.greek_lemma))
        if cands:
            out.append((m.seq, cands))
    return out


def bipartite_lexicon_coverage(
    coptic_cand_sets: Sequence[tuple[int, set[str]]],
    greek_lemmas: Sequence[str],
) -> float:
    """Exact maximum bipartite matching fraction of covered Coptic content lemmas.

    Multiset- and synonym-aware: if ``ⲣⲱⲙⲉ`` occurs 4 times, all 4 occurrences
    require 4 distinct matching Greek content tokens (e.g. ``ἄνθρωπος``/``ἀνήρ``).
    """
    n_left = len(coptic_cand_sets)
    if n_left == 0:
        return 1.0
    if not greek_lemmas:
        return 0.0

    n_right = len(greek_lemmas)
    bare_g = [_strip_bare(g) for g in greek_lemmas]
    # For longer chunks (>=8 content morphs), restrict edges to a generous
    # +/-0.42 relative position band so duplicating a word in one clause
    # (e.g. ἄνθρωπος ἄνθρωπος) cannot steal coverage credit for a dropped
    # occurrence in a distant clause.
    use_band = n_left >= 8
    adj: list[list[int]] = []
    for u, (_seq, cand_set) in enumerate(coptic_cand_sets):
        u_f = (u + 0.5) / n_left
        edges = [
            j
            for j, bg in enumerate(bare_g)
            if bg in cand_set
            and (not use_band or abs(u_f - (j + 0.5) / max(n_right, 1)) <= 0.42)
        ]
        adj.append(edges)

    match_right: list[int] = [-1] * len(bare_g)

    def _dfs(u: int, visited: list[bool]) -> bool:
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_right[v] == -1 or _dfs(match_right[v], visited):
                match_right[v] = u
                return True
        return False

    matched = 0
    for u in range(n_left):
        if adj[u] and _dfs(u, [False] * len(bare_g)):
            matched += 1
    return matched / n_left


def find_unmatched_content_indices(
    coptic_cand_sets: Sequence[tuple[int, set[str]]],
    greek_lemmas: Sequence[str],
) -> list[int]:
    """Return left indices in ``coptic_cand_sets`` unmatched by position-aware match.

    Uses closest-position greedy matching first, then augments via Kuhn DFS so
    when a repeated Coptic lemma (e.g. 4x ``ⲙⲟⲩⲓ``) is dropped at one position
    (e.g. clause 1), the exact dropped occurrence is pinpointed cleanly.
    """
    n_left = len(coptic_cand_sets)
    if n_left == 0:
        return []
    if not greek_lemmas:
        return list(range(n_left))

    n_right = len(greek_lemmas)
    bare_g = [_strip_bare(g) for g in greek_lemmas]
    adj: list[list[int]] = [
        sorted(
            (j for j, bg in enumerate(bare_g) if bg in cand_set),
            key=lambda j, u_i=u: abs(
                (u_i + 0.5) / n_left - (j + 0.5) / max(n_right, 1)
            ),
        )
        for u, (_seq, cand_set) in enumerate(coptic_cand_sets)
    ]
    match_right: list[int] = [-1] * n_right
    match_left: list[int] = [-1] * n_left

    # Pass 1: assign closest-position pairs first so repeated lemmas align monotonically
    cand_edges: list[tuple[float, int, int]] = []
    for u, nbrs in enumerate(adj):
        u_f = (u + 0.5) / n_left
        for v in nbrs:
            cand_edges.append((abs(u_f - (v + 0.5) / n_right), u, v))
    cand_edges.sort(key=lambda item: item[0])
    for _dist, u, v in cand_edges:
        if match_left[u] == -1 and match_right[v] == -1:
            match_left[u] = v
            match_right[v] = u

    # Pass 2: maximize cardinality via augmenting paths without disturbing close matches
    def _dfs(u: int, visited: list[bool]) -> bool:
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_right[v] == -1 or _dfs(match_right[v], visited):
                match_right[v] = u
                match_left[u] = v
                return True
        return False

    for u in range(n_left):
        if match_left[u] == -1 and adj[u]:
            _dfs(u, [False] * n_right)
    return [u for u in range(n_left) if match_left[u] == -1]


def build_missing_content_constraints(
    morphemes: Sequence[CopticMorpheme],
    *,
    existing_constraints: Sequence[ConstraintTrie],
    predicted_greek_lemmas: Sequence[str],
    ccl_lexicon: CclLexicon,
    lexical_model: LexicalTranslationModel | None,
    inflector: GreekInflector,
    tokenizer: BilingualTokenizer,
    native_gloss_hints: Mapping[str, str] | None = None,
    unaccented: bool = False,
    min_prob: float = 0.50,
    max_added: int = 3,
) -> list[ConstraintTrie]:
    """Build positional synonym tries for dropped high-confidence content morphs."""
    if lexical_model is None:
        return []
    cand_sets = build_coptic_content_candidate_sets(
        morphemes,
        ccl_lexicon=ccl_lexicon,
        lexical_model=lexical_model,
        native_gloss_hints=native_gloss_hints,
        top_k=12,
    )
    unmatched_idxs = find_unmatched_content_indices(cand_sets, predicted_greek_lemmas)
    if not unmatched_idxs:
        return []

    unmatched_seqs = {cand_sets[i][0] for i in unmatched_idxs}
    existing_seqs = {ct.source_seq for ct in existing_constraints}
    n_morphs = max(len(morphemes), 1)
    n_np = max(sum(1 for x in morphemes if x.pos != "PUNCT"), 1)
    est_tgt = max(int(n_np * 0.85), 8)
    added: list[ConstraintTrie] = []

    prec_content = 0
    for m_idx, m in enumerate(morphemes):
        cur_prec_content = prec_content
        if m.pos in CONTENT_COPTIC_POS and not is_grammaticalized_prep_noun(
            morphemes, m_idx
        ):
            prec_content += 1
        if m.seq not in unmatched_seqs or m.seq in existing_seqs:
            continue
        if m.pos not in {"N", "NPROP", "V", "VSTAT", "VIMP", "NUM"}:
            continue
        if is_grammaticalized_prep_noun(morphemes, m_idx):
            continue
        norm_s = normalize_coptic(m.norm).replace("\u2cff", "")
        lem_s = normalize_coptic(m.lemma).replace("\u2cff", "")
        key_lem = lem_s or norm_s
        if not key_lem or len(key_lem) < 2:
            continue
        # Skip ultra-generic light verbs/speech introducers
        if key_lem in {"ϣⲱⲡⲉ", "ⲉⲓⲣⲉ", "ϫⲱ", "ⲡⲉϫⲉ", "ⲉⲥⲏⲧ"} and m.pos != "VSTAT":
            continue
        hits = lexical_model.lookup(
            coptic_lemma=key_lem,
            coptic_pos=m.pos,
            coptic_norm=norm_s,
            lang=m.lang,
            ccl_lexicon=ccl_lexicon,
        )
        if not hits:
            continue
        top = hits[0]
        tot_cnt = top.count_nt + top.count_ot
        if top.provenance != "L" and (top.prob_g_given_c < min_prob or tot_cnt < 3.0):
            continue
        syn_surfaces: list[str] = []
        for lc in hits[:3]:
            if lc.prob_g_given_c < 0.08 and lc is not top:
                continue
            g_pos = (
                lc.greek_pos if lc.greek_pos in {"N-", "A-", "V-", "NPROP"} else None
            )
            syn_surfaces.extend(
                inflector.all_inflected_forms(lc.greek_lemma, pos=g_pos, max_forms=32)
            )
        if not syn_surfaces:
            continue
        rel_p = (m_idx + 0.5) / n_morphs
        min_st = (
            0
            if cur_prec_content == 0
            else max(
                cur_prec_content + (1 if cur_prec_content >= 2 else 0),
                int((rel_p - 0.14) * est_tgt),
            )
        )
        max_st = max(min_st + 6, int((rel_p + 0.16) * est_tgt) + 3)
        syn_bare = {_strip_bare(lc.greek_lemma) for lc in hits[:3]}
        p1_hits = sum(1 for gl in predicted_greek_lemmas if gl in syn_bare)
        trie = ConstraintTrie.from_surfaces(
            label=top.greek_lemma,
            lemma=top.greek_lemma,
            surfaces=syn_surfaces,
            tokenizer=tokenizer,
            source_seq=m.seq,
            rel_pos=rel_p,
            unaccented=unaccented,
            is_repair=True,
            min_step=min_st,
            max_step=max_st,
            min_count=p1_hits + 1,
        )
        if trie is not None:
            added.append(trie)
            if len(added) >= max_added:
                break
    return added


def wu_attention_coverage_penalty(
    attn: Sequence[Sequence[float]],
    tok_to_seq: Sequence[int | None],
    content_seqs: Sequence[int],
    *,
    tau: float = 0.5,
) -> float:
    """Google NMT (Wu et al. 2016) coverage penalty over content Coptic morphemes.

    Returns a non-positive value (``0.0`` when every content Coptic morpheme
    receives >= ``tau`` cumulative cross-attention mass, strongly negative when
    any content morpheme is ignored/dropped).
    """
    if not content_seqs or not attn or not tok_to_seq:
        return 0.0
    seq_mass: dict[int, float] = {s: 0.0 for s in content_seqs}
    content_set = set(content_seqs)
    n_cols = len(tok_to_seq)
    for row in attn:
        limit = min(len(row), n_cols)
        for s_pos in range(limit):
            sq = tok_to_seq[s_pos]
            if sq is not None and sq in content_set:
                seq_mass[sq] += float(row[s_pos])
    inv_tau = 1.0 / max(tau, 1e-6)
    total = 0.0
    for sq in content_seqs:
        ratio = min(max(seq_mass[sq] * inv_tau, 1e-3), 1.0)
        total += math.log(ratio)
    return total / len(content_seqs)


def assign_bound_group_coptic_seqs(
    primary_seqs: Sequence[int],
    morphemes: Sequence[CopticMorpheme],
) -> list[list[int]]:
    """Expand single-argmax ``coptic_seqs`` to include bound-group function morphs.

    Each Greek token keeps its primary attended Coptic morpheme plus any
    unclaimed non-content function morphemes (articles, TAM auxiliaries,
    pronouns, prepositions) belonging to the same Coptic bound group, while
    leaving genuinely dropped content morphemes unattached for audit visibility.
    """
    if not primary_seqs:
        return []
    m_by_seq = {m.seq: m for m in morphemes}
    group_to_func: dict[int, list[int]] = defaultdict(list)
    for m in morphemes:
        if m.pos not in CONTENT_COPTIC_POS and m.pos != "PUNCT":
            group_to_func[m.group_seq].append(m.seq)

    claimed: set[int] = set(primary_seqs)
    spans: list[list[int]] = []
    for p_seq in primary_seqs:
        bag = [p_seq]
        m = m_by_seq.get(p_seq)
        if m is not None:
            for f_seq in group_to_func.get(m.group_seq, ()):
                if f_seq not in claimed:
                    claimed.add(f_seq)
                    bag.append(f_seq)
        spans.append(sorted(set(bag)))
    return spans


def normalize_hint_for_lookup(text: str, *, unaccented: bool = False) -> str:
    """Normalize a Greek hint for BPE/lexicon lookup."""
    norm = normalize_greek_for_model(text, unaccented=unaccented)
    return strip_greek_to_bare(norm) if unaccented else norm
