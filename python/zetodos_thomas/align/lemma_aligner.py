# SPDX-License-Identifier: Apache-2.0
"""Seeded eflomal Coptic-Greek lemma+POS aligner with NT domain upweighting."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.ingest.holdouts import (
    assert_clean_training_unit,
    clean_nt_train_ids,
    load_nt_split,
)
from zetodos_thomas.ingest.morphgnt import pair_horner_morphgnt
from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.ingest.ot_swete import load_ot_swete_pairs
from zetodos_thomas.integrations.aligners.eflomal_adapter import align_eflomal
from zetodos_thomas.integrations.aligners.types import AlignerConfig, BitextSentence
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.morphology.coptic_analyzer import CopticAnalyzer
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit, GreekUnit

# Purely syntactic/analytic Sahidic clitics filtered before eflomal so
# grammatical markers like APST ⲁ or PTC ⲛϭⲓ never pollute lexical alignments.
_FILTERED_COPTIC_POS = frozenset(
    {
        "APST",
        "FUT",
        "ACAUS",
        "ACOND",
        "ACONJ",
        "CPRET",
        "CFOC",
        "CCIRC",
        "CREL",
        "APREC",
        "ALIM",
        "AJUS",
        "AAOR",
        "ANEGPST",
        "ANEGAOR",
        "ANEGJUS",
        "ANEGOPT",
        "ANY",
        "ACOND_PPERS",
        "AOPT_PPERS",
        "ACONJ_PPERS",
        "AFUTCONJ",
        "PUNCT",
    }
)
_FILTERED_COPTIC_LEMMAS = frozenset({"ⲛϭⲓ"})

_COARSE_COPTIC_POS: dict[str, str] = {
    "N": "N",
    "NPROP": "N",
    "V": "V",
    "VSTAT": "VS",
    "VBD": "V",
    "VIMP": "V",
    "PREP": "P",
    "ART": "D",
    "PPOS": "D",
    "PDEM": "D",
    "PPERS": "R",
    "PPERO": "R",
    "PPERI": "R",
    "PINT": "R",
    "ADV": "A",
    "CONJ": "C",
    "PTC": "C",
    "NEG": "X",
    "EXIST": "E",
    "COP": "E",
    "NUM": "M",
    "IMOD": "A",
}

_COARSE_GREEK_POS: dict[str, str] = {
    "N-": "N",
    "NPROP": "N",
    "V-": "V",
    "A-": "A",
    "D-": "A",
    "P-": "P",
    "RA": "D",
    "RD": "D",
    "RP": "R",
    "RI": "R",
    "RR": "R",
    "C-": "C",
    "X-": "X",
    "I-": "C",
    "UNK": "N",
}


_COMPATIBLE_GREEK_POS: dict[str, set[str]] = {
    "N": {"N-", "A-", "NPROP"},
    "V": {"V-", "A-"},
    "VS": {"V-", "A-", "N-"},
    "A": {"D-", "A-", "C-"},
    "P": {"P-", "C-", "D-"},
    "D": {"RA", "RD", "A-", "RI"},
    "R": {"RP", "RD", "RI", "RR", "A-"},
    "C": {"C-", "D-", "P-", "I-"},
    "X": {"D-", "C-", "X-"},
    "E": {"V-", "C-"},
    "M": {"A-", "N-"},
}


def coarse_coptic_pos(pos: str) -> str:
    return _COARSE_COPTIC_POS.get(pos, "N")


def coarse_greek_pos(pos: str) -> str:
    return _COARSE_GREEK_POS.get(pos, "N")


def filter_coptic_for_alignment(
    morphemes: Iterable[CopticMorpheme],
) -> list[tuple[int, str, str]]:
    """Keep content & lexical-function morphemes as ``(orig_idx, tok_key, lemma)``."""
    out: list[tuple[int, str, str]] = []
    for idx, m in enumerate(morphemes):
        lem = normalize_coptic(m.lemma) or m.lemma
        if m.pos in _FILTERED_COPTIC_POS or lem in _FILTERED_COPTIC_LEMMAS:
            continue
        if not lem:
            continue
        cpos = coarse_coptic_pos(m.pos)
        out.append((idx, f"{lem}|{cpos}", lem))
    return out


def prepare_greek_alignment_tokens(
    unit: GreekUnit,
    inflector: GreekInflector | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``(tok_key, lemma, pos)`` for each Greek token (lemmatizing Swete LXX)."""
    out: list[tuple[str, str, str]] = []
    for tok in unit.tokens:
        lemma = tok.lemma
        pos = tok.morph.pos
        if (
            pos == "UNK" or not lemma or lemma == tok.normalized
        ) and inflector is not None:
            parses = inflector.analyze_surface(tok.surface)
            if parses:
                best_lem, best_morph, _lp = parses[0]
                if best_lem and best_morph.pos != "UNK":
                    lemma = best_lem
                    pos = best_morph.pos
        if not lemma:
            continue
        gpos = coarse_greek_pos(pos)
        out.append((f"{lemma}|{gpos}", lemma, pos))
    return out


@dataclass(frozen=True)
class LexicalCandidate:
    """One ranked Coptic -> Greek lexical translation candidate with provenance."""

    greek_lemma: str
    greek_pos: str
    score: float
    prob_g_given_c: float
    prob_c_given_g: float
    provenance: str  # "L" loanword, "N" NT, "P" OT/LXX
    count_nt: float
    count_ot: float


def _coptic_orth_key(lemma: str) -> str:
    """Normalize Sahidic dialectal/NHC II doubled vowels & ⲁⲉⲓ<->ⲟⲓ diphthong."""
    collapsed: list[str] = []
    for ch in lemma:
        if ch in {"ⲁ", "ⲉ", "ⲏ", "ⲟ", "ⲱ"} and collapsed and collapsed[-1] == ch:
            continue
        collapsed.append(ch)
    s = "".join(collapsed).replace("ⲁⲉⲓ", "ⲟⲓ")
    if len(s) >= 6 and s.startswith("ϩⲓ"):
        s = s[2:]
    return s


def _coptic_ao_key(lemma: str) -> str:
    """Conservative alpha-to-o interchange fallback after vowel collapse."""
    return _coptic_orth_key(lemma).replace("ⲁ", "ⲟ")


def _coptic_ae_key(lemma: str) -> str:
    """Conservative Sahidic/NHC II vowel & nasal fallback (ⲁ<->ⲉ, -ⲉ, ⲙⲛⲧ<->ⲙⲧ)."""
    s = _coptic_orth_key(lemma)
    if len(s) >= 4 and s.endswith("ⲉ"):
        s = s[:-1]
    return s.replace("ⲉ", "ⲁ").replace("ⲙⲛⲧ", "ⲙⲧ")


def extract_coptic_right_context(morphemes: Sequence[CopticMorpheme], idx: int) -> str:
    """Extract compact syntactic right-context tag at ``morphemes[idx]``."""
    if idx + 1 >= len(morphemes) or morphemes[idx + 1].pos == "PUNCT":
        return "END"
    m1 = morphemes[idx + 1]
    m1_lem = normalize_coptic(m1.lemma) or m1.lemma
    if m1.pos in {"ADV", "IMOD"}:
        return f"adv:{m1_lem}"
    if m1.pos == "PREP":
        return f"prep:{m1_lem}"
    if m1.pos in {"CONJ", "PTC", "CREL", "CCIRC"}:
        return f"syn:{m1_lem}"
    if m1.pos in {"V", "N", "NPROP", "VSTAT"}:
        return f"lex:{m1_lem}"
    if m1.pos in {"PPERO", "PPERI"}:
        return "pro"
    if m1.pos in {"ART", "PPOS", "PDEM", "NUM"}:
        return "det"
    return ""


class LexicalTranslationModel:
    """Bidirectional alignment + CCL loanword dictionary for Clause Transfer."""

    def __init__(
        self,
        *,
        by_lemma_pos: dict[tuple[str, str], list[LexicalCandidate]],
        by_lemma: dict[str, list[LexicalCandidate]],
        by_lemma_ctx: dict[tuple[str, str, str], list[LexicalCandidate]] | None = None,
        nt_weight: float = 3.0,
    ) -> None:
        self._by_lemma_pos = by_lemma_pos
        self._by_lemma = by_lemma
        self._by_lemma_ctx = by_lemma_ctx or {}
        self.nt_weight = nt_weight

        orth_idx: dict[str, list[LexicalCandidate]] = {}
        ao_idx: dict[str, list[LexicalCandidate]] = {}
        ae_idx: dict[str, list[LexicalCandidate]] = {}
        for c_lem, cands in sorted(by_lemma.items()):
            if len(c_lem) < 3 or not cands:
                continue
            ok = _coptic_orth_key(c_lem)
            if ok not in orth_idx or cands[0].score > orth_idx[ok][0].score:
                orth_idx[ok] = cands
            ak = _coptic_ao_key(c_lem)
            if ak not in ao_idx or cands[0].score > ao_idx[ak][0].score:
                ao_idx[ak] = cands
            if len(c_lem) >= 4:
                aek = _coptic_ae_key(c_lem)
                if len(aek) >= 3 and (
                    aek not in ae_idx or cands[0].score > ae_idx[aek][0].score
                ):
                    ae_idx[aek] = cands
        self._by_orth_key = orth_idx
        self._by_ao_key = ao_idx
        self._by_ae_key = ae_idx

    @classmethod
    def build_from_bitext(
        cls,
        *,
        nt_pairs: list[tuple[CopticUnit, GreekUnit]],
        ot_pairs: list[tuple[CopticUnit, GreekUnit]],
        inflector: GreekInflector,
        ccl_lexicon: CclLexicon | None = None,
        nt_weight: float = 3.0,
        reverse_alpha: float = 0.45,
        repo_root: Path | None = None,
        seed: int = 42,
    ) -> LexicalTranslationModel:
        """Align NT (upweighted 3x) + OT bitext seeded with DDGLC loanwords."""
        sentences: list[BitextSentence] = []
        sent_meta: list[
            tuple[str, float, list[tuple[str, str]], list[str], list[tuple[str, str]]]
        ] = []

        def _append_pair(
            c_unit: CopticUnit,
            g_unit: GreekUnit,
            *,
            tag: str,
            weight: float,
            repeats: int,
        ) -> None:
            c_filtered = filter_coptic_for_alignment(c_unit.morphemes)
            g_tokens = prepare_greek_alignment_tokens(g_unit, inflector=inflector)
            if not c_filtered or not g_tokens:
                return
            src_toks = [t[1] for t in c_filtered]
            trg_toks = [t[0] for t in g_tokens]
            c_pairs = [(m_tok.split("|")[0], m_tok.split("|")[1]) for m_tok in src_toks]
            c_ctxs = [
                extract_coptic_right_context(c_unit.morphemes, orig_idx)
                for orig_idx, _, _ in c_filtered
            ]
            g_pairs = [(tok[1], tok[2]) for tok in g_tokens]
            for r in range(repeats):
                uid = f"{c_unit.unit_id}#{tag}{r}"
                sentences.append(
                    BitextSentence(
                        unit_id=uid,
                        source_tokens=tuple(src_toks),
                        target_tokens=tuple(trg_toks),
                    )
                )
                sent_meta.append((tag, weight / repeats, c_pairs, c_ctxs, g_pairs))

        nt_repeats = max(1, round(nt_weight))
        for c_u, g_u in nt_pairs:
            assert_clean_training_unit(c_u.unit_id, source_id=c_u.source_id)
            assert_clean_training_unit(g_u.unit_id, source_id=g_u.source_id)
            _append_pair(c_u, g_u, tag="N", weight=nt_weight, repeats=nt_repeats)
        for c_u, g_u in ot_pairs:
            assert_clean_training_unit(c_u.unit_id, source_id=c_u.source_id)
            assert_clean_training_unit(g_u.unit_id, source_id=g_u.source_id)
            _append_pair(c_u, g_u, tag="P", weight=1.0, repeats=1)

        # High-precision DDGLC loanword seed pairs (keep Coptic->Greek anchored)
        if ccl_lexicon is not None:
            seed_idx = 0
            for _surf, entries in sorted(ccl_lexicon._loanwords.items()):
                for e in entries[:1]:
                    c_cpos = (
                        "N"
                        if e.greek_pos in {"N-", "A-"}
                        else "V"
                        if e.greek_pos == "V-"
                        else "C"
                    )
                    g_cpos = coarse_greek_pos(e.greek_pos)
                    for r in range(2):
                        sentences.append(
                            BitextSentence(
                                unit_id=f"seed.{seed_idx}.{r}",
                                source_tokens=(f"{e.coptic_lemma}|{c_cpos}",),
                                target_tokens=(f"{e.greek_lemma}|{g_cpos}",),
                            )
                        )
                        sent_meta.append(
                            (
                                "L",
                                1.5,
                                [(e.coptic_lemma, c_cpos)],
                                [""],
                                [(e.greek_lemma, e.greek_pos)],
                            )
                        )
                    seed_idx += 1

        cfg = AlignerConfig(
            seed=seed,
            eflomal_model=3,
            eflomal_n_samplers=3,
        )
        links = align_eflomal(sentences, config=cfg, repo_root=repo_root)

        # Accumulate weighted link counts per (c_lemma, c_cpos) <-> (g_lemma, g_pos)
        link_nt: dict[tuple[str, str, str, str], float] = defaultdict(float)
        link_ot: dict[tuple[str, str, str, str], float] = defaultdict(float)
        c_marg: dict[tuple[str, str], float] = defaultdict(float)
        g_marg: dict[tuple[str, str], float] = defaultdict(float)
        ctx_nt: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
        ctx_ot: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
        ctx_marg: dict[tuple[str, str, str], float] = defaultdict(float)

        for idx, sym in enumerate(links.symmetrized):
            tag, w, c_pairs, c_ctxs, g_pairs = sent_meta[idx]
            if tag == "L":
                continue
            for si, ti in sym.pairs:
                if si < len(c_pairs) and ti < len(g_pairs):
                    c_lem, c_cpos = c_pairs[si]
                    r_ctx = c_ctxs[si] if si < len(c_ctxs) else ""
                    g_lem, g_pos = g_pairs[ti]
                    key = (c_lem, c_cpos, g_lem, g_pos)
                    if tag == "N":
                        link_nt[key] += w
                    else:
                        link_ot[key] += w
                    c_marg[(c_lem, c_cpos)] += w
                    g_marg[(g_lem, coarse_greek_pos(g_pos))] += w
                    if r_ctx:
                        ckey = (c_lem, c_cpos, r_ctx, g_lem, g_pos)
                        if tag == "N":
                            ctx_nt[ckey] += w
                        else:
                            ctx_ot[ckey] += w
                        ctx_marg[(c_lem, c_cpos, r_ctx)] += w

        all_keys = set(link_nt) | set(link_ot)
        grouped: dict[tuple[str, str], list[LexicalCandidate]] = defaultdict(list)
        by_lemma_acc: dict[str, dict[tuple[str, str], LexicalCandidate]] = defaultdict(
            dict
        )

        for c_lem, c_cpos, g_lem, g_pos in all_keys:
            nt_w = link_nt.get((c_lem, c_cpos, g_lem, g_pos), 0.0)
            ot_w = link_ot.get((c_lem, c_cpos, g_lem, g_pos), 0.0)
            tot_w = nt_w + ot_w
            if tot_w < 1.2:
                continue
            p_gc = tot_w / (c_marg[(c_lem, c_cpos)] + 2.5)
            p_cg = tot_w / (g_marg[(g_lem, coarse_greek_pos(g_pos))] + 8.0)
            if p_gc < 0.015:
                continue
            prov = "N" if nt_w >= 0.5 else "P"
            assoc = p_gc * (p_cg**reverse_alpha)
            compat_adj = (
                0.7 if g_pos in _COMPATIBLE_GREEK_POS.get(c_cpos, set()) else -1.8
            )
            score = (
                math.log(max(assoc, 1e-9))
                + (0.35 if prov == "N" else 0.0)
                + 0.12 * math.log1p(tot_w)
                + compat_adj
            )
            cand = LexicalCandidate(
                greek_lemma=g_lem,
                greek_pos=g_pos,
                score=round(score, 4),
                prob_g_given_c=round(p_gc, 4),
                prob_c_given_g=round(p_cg, 4),
                provenance=prov,
                count_nt=round(nt_w / nt_repeats, 2),
                count_ot=round(ot_w, 2),
            )
            grouped[(c_lem, c_cpos)].append(cand)
            prev = by_lemma_acc[c_lem].get((g_lem, g_pos))
            if prev is None or cand.score > prev.score:
                by_lemma_acc[c_lem][(g_lem, g_pos)] = cand

        by_lemma_pos: dict[tuple[str, str], list[LexicalCandidate]] = {
            k: sorted(v, key=lambda item: -item.score)[:8] for k, v in grouped.items()
        }
        by_lemma: dict[str, list[LexicalCandidate]] = {
            k: sorted(v.values(), key=lambda item: -item.score)[:8]
            for k, v in by_lemma_acc.items()
        }

        # Context-conditioned tables with Empirical-Bayes Dirichlet shrinkage
        base_pgc_map: dict[tuple[str, str, str, str], float] = {}
        for c_lem, c_cpos, g_lem, g_pos in all_keys:
            base_w = link_nt.get((c_lem, c_cpos, g_lem, g_pos), 0.0) + link_ot.get(
                (c_lem, c_cpos, g_lem, g_pos), 0.0
            )
            base_pgc_map[(c_lem, c_cpos, g_lem, g_pos)] = base_w / (
                c_marg[(c_lem, c_cpos)] + 2.5
            )

        ctx_grouped: dict[
            tuple[str, str, str], list[tuple[LexicalCandidate, float]]
        ] = defaultdict(list)
        all_ctx_keys = set(ctx_nt) | set(ctx_ot)
        prior_strength = 4.0
        for c_lem, c_cpos, r_ctx, g_lem, g_pos in all_ctx_keys:
            if ctx_marg[(c_lem, c_cpos, r_ctx)] < 3.0:
                continue
            nt_cw = ctx_nt.get((c_lem, c_cpos, r_ctx, g_lem, g_pos), 0.0)
            ot_cw = ctx_ot.get((c_lem, c_cpos, r_ctx, g_lem, g_pos), 0.0)
            eff_support = (nt_cw / nt_repeats) + ot_cw
            if eff_support < 2.0:
                continue
            tot_cw = nt_cw + ot_cw
            base_w = link_nt.get((c_lem, c_cpos, g_lem, g_pos), 0.0) + link_ot.get(
                (c_lem, c_cpos, g_lem, g_pos), 0.0
            )
            p_gc_base = base_pgc_map.get((c_lem, c_cpos, g_lem, g_pos), 0.01)
            p_gc_ctx = (tot_cw + prior_strength * p_gc_base) / (
                ctx_marg[(c_lem, c_cpos, r_ctx)] + prior_strength
            )
            p_cg = base_w / (g_marg[(g_lem, coarse_greek_pos(g_pos))] + 8.0)
            if p_gc_ctx < 0.02 or p_cg <= 0.0:
                continue
            prov = "N" if nt_cw >= 0.5 else "P"
            assoc = p_gc_ctx * (p_cg**reverse_alpha)
            compat_adj = (
                0.7 if g_pos in _COMPATIBLE_GREEK_POS.get(c_cpos, set()) else -1.8
            )
            score = (
                math.log(max(assoc, 1e-9))
                + (0.35 if prov == "N" else 0.0)
                + 0.12 * math.log1p(base_w)
                + compat_adj
            )
            cand = LexicalCandidate(
                greek_lemma=g_lem,
                greek_pos=g_pos,
                score=round(score, 4),
                prob_g_given_c=round(p_gc_ctx, 4),
                prob_c_given_g=round(p_cg, 4),
                provenance=prov,
                count_nt=round(nt_cw / nt_repeats, 2),
                count_ot=round(ot_cw, 2),
            )
            ctx_grouped[(c_lem, c_cpos, r_ctx)].append(
                (cand, abs(p_gc_ctx - p_gc_base))
            )

        by_lemma_ctx: dict[tuple[str, str, str], list[LexicalCandidate]] = {}
        for ctx_key, pairs in ctx_grouped.items():
            pairs.sort(key=lambda item: -item[0].score)
            kept = pairs[:5]
            c_lem, c_cpos, _ = ctx_key
            base_list = by_lemma_pos.get((c_lem, c_cpos), [])
            top_changed = bool(
                not base_list
                or (
                    kept[0][0].greek_lemma,
                    kept[0][0].greek_pos,
                )
                != (base_list[0].greek_lemma, base_list[0].greek_pos)
            )
            max_shift = max(diff for _, diff in kept)
            if top_changed or max_shift >= 0.08:
                by_lemma_ctx[ctx_key] = [cand for cand, _ in kept]

        return cls(
            by_lemma_pos=by_lemma_pos,
            by_lemma=by_lemma,
            by_lemma_ctx=by_lemma_ctx,
            nt_weight=nt_weight,
        )

    @classmethod
    def build_full(
        cls,
        *,
        downloads_dir: Path,
        split_path: Path,
        analyzer: CopticAnalyzer,
        inflector: GreekInflector,
        ccl_lexicon: CclLexicon,
        nt_weight: float = 3.0,
        repo_root: Path | None = None,
        exclude_synoptic_thomas: bool = False,
    ) -> LexicalTranslationModel:
        """Run full leak-free alignment on clean NT TRAIN split + Sahidic OT x Swete."""
        split = load_nt_split(split_path)
        train_ids = clean_nt_train_ids(
            split, exclude_synoptic_thomas=exclude_synoptic_thomas
        )
        raw_nt = pair_horner_morphgnt(
            sword_root=downloads_dir / "sword",
            morphgnt_dir=downloads_dir / "morphgnt-sblgnt",
        )
        train_raw_c = [c for c, g in raw_nt if g.unit_id in train_ids]
        greek_by_id = {g.unit_id: g for _c, g in raw_nt if g.unit_id in train_ids}
        analyzed_nt_c = analyzer.analyze_units(train_raw_c)
        nt_pairs = [
            (c_u, greek_by_id[c_u.unit_id])
            for c_u in analyzed_nt_c
            if c_u.unit_id in greek_by_id
        ]
        ot_pairs = load_ot_swete_pairs(
            tt_zip=downloads_dir / "sahidic-ot" / "sahidic.ot_TT.zip",
            swete_dir=downloads_dir / "lxx-swete",
        )
        return cls.build_from_bitext(
            nt_pairs=nt_pairs,
            ot_pairs=ot_pairs,
            inflector=inflector,
            ccl_lexicon=ccl_lexicon,
            nt_weight=nt_weight,
            repo_root=repo_root,
        )

    def lookup(
        self,
        *,
        coptic_lemma: str,
        coptic_pos: str,
        coptic_norm: str = "",
        lang: str | None = None,
        ccl_lexicon: CclLexicon | None = None,
        right_ctx: str = "",
    ) -> list[LexicalCandidate]:
        """Return ranked Greek candidates combining DDGLC loanwords and alignment."""
        coptic_lemma = normalize_coptic(coptic_lemma) or coptic_lemma
        merged: dict[tuple[str, str], LexicalCandidate] = {}

        allow_ccl = (
            lang == "Greek"
            or len(coptic_lemma) >= 4
            or (len(coptic_lemma) >= 3 and coptic_lemma not in self._by_lemma)
        )
        if ccl_lexicon is not None and allow_ccl:
            loans = ccl_lexicon.resolve(
                norm=coptic_norm or coptic_lemma,
                lemma=coptic_lemma,
                pos=coptic_pos,
                lang=lang,
            )
            for idx, entry in enumerate(loans):
                boost = 1.6 if lang == "Greek" else 0.7
                sc = boost - 0.35 * idx
                merged[(entry.greek_lemma, entry.greek_pos)] = LexicalCandidate(
                    greek_lemma=entry.greek_lemma,
                    greek_pos=entry.greek_pos,
                    score=round(sc, 4),
                    prob_g_given_c=0.95 if lang == "Greek" else 0.7,
                    prob_c_given_g=0.95,
                    provenance="L",
                    count_nt=0.0,
                    count_ot=0.0,
                )

        cpos = coarse_coptic_pos(coptic_pos)
        align_hits = self._by_lemma_pos.get((coptic_lemma, cpos))
        if not align_hits and cpos == "VS":
            align_hits = self._by_lemma_pos.get((coptic_lemma, "V"))
        if not align_hits:
            align_hits = self._by_lemma.get(coptic_lemma, [])
        orth_discount = 0.0
        if not align_hits and len(coptic_lemma) >= 3:
            orth_hits = self._by_orth_key.get(_coptic_orth_key(coptic_lemma), [])
            if orth_hits:
                align_hits = orth_hits
                orth_discount = 0.5
            elif not merged:
                ao_hits = self._by_ao_key.get(_coptic_ao_key(coptic_lemma), [])
                if ao_hits:
                    align_hits = ao_hits
                    orth_discount = 0.7
                elif len(coptic_lemma) >= 4:
                    ae_hits = self._by_ae_key.get(_coptic_ae_key(coptic_lemma), [])
                    if ae_hits:
                        align_hits = ae_hits
                        orth_discount = 0.85
        for cand in align_hits:
            adj_cand = (
                LexicalCandidate(
                    greek_lemma=cand.greek_lemma,
                    greek_pos=cand.greek_pos,
                    score=round(cand.score - orth_discount, 4),
                    prob_g_given_c=cand.prob_g_given_c,
                    prob_c_given_g=cand.prob_c_given_g,
                    provenance=cand.provenance,
                    count_nt=cand.count_nt,
                    count_ot=cand.count_ot,
                )
                if orth_discount > 0.0
                else cand
            )
            key = (adj_cand.greek_lemma, adj_cand.greek_pos)
            if key in merged:
                existing = merged[key]
                merged[key] = LexicalCandidate(
                    greek_lemma=adj_cand.greek_lemma,
                    greek_pos=adj_cand.greek_pos,
                    score=round(max(existing.score + 0.5, adj_cand.score + 0.8), 4),
                    prob_g_given_c=max(
                        existing.prob_g_given_c, adj_cand.prob_g_given_c
                    ),
                    prob_c_given_g=max(
                        existing.prob_c_given_g, adj_cand.prob_c_given_g
                    ),
                    provenance="L",
                    count_nt=adj_cand.count_nt,
                    count_ot=adj_cand.count_ot,
                )
            else:
                merged[key] = adj_cand

        if right_ctx:
            ctx_hits = self._by_lemma_ctx.get((coptic_lemma, cpos, right_ctx), [])
            for cc in ctx_hits:
                key = (cc.greek_lemma, cc.greek_pos)
                if key in merged:
                    existing = merged[key]
                    if existing.provenance != "L":
                        merged[key] = LexicalCandidate(
                            greek_lemma=existing.greek_lemma,
                            greek_pos=existing.greek_pos,
                            score=round(
                                0.65 * cc.score + 0.35 * existing.score + 0.25, 4
                            ),
                            prob_g_given_c=cc.prob_g_given_c,
                            prob_c_given_g=cc.prob_c_given_g,
                            provenance=cc.provenance,
                            count_nt=cc.count_nt,
                            count_ot=cc.count_ot,
                        )
                else:
                    merged[key] = cc

        return sorted(merged.values(), key=lambda c: (-c.score, c.greek_lemma))[:8]

    def save_shards(self, out_dir: Path, *, max_bytes: int = 680_000) -> list[Path]:
        """Save aligned lexicon compactly (< 1024 KiB per file)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("lemma-align-*.json"):
            old.unlink()
        for old in out_dir.glob("lemma-ctx-*.json"):
            old.unlink()
        files: list[Path] = []
        shard: dict[str, list[list[str | float]]] = {}
        shard_idx = 0
        est = 2
        for (c_lem, c_cpos), cands in sorted(self._by_lemma_pos.items()):
            key = f"{c_lem}|{c_cpos}"
            rows: list[list[str | float]] = [
                [
                    c.greek_lemma,
                    c.greek_pos,
                    c.score,
                    c.prob_g_given_c,
                    c.prob_c_given_g,
                    c.provenance,
                    c.count_nt,
                    c.count_ot,
                ]
                for c in cands
            ]
            srows = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
            row_sz = len(key.encode("utf-8")) + len(srows.encode("utf-8")) + 6
            if est + row_sz > max_bytes and shard:
                p = out_dir / f"lemma-align-{shard_idx}.json"
                p.write_text(
                    json.dumps(shard, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                files.append(p)
                shard = {}
                shard_idx += 1
                est = 2
            shard[key] = rows
            est += row_sz
        if shard or not files:
            p = out_dir / f"lemma-align-{shard_idx}.json"
            p.write_text(
                json.dumps(shard, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            files.append(p)

        if self._by_lemma_ctx:
            cshard: dict[str, list[list[str | float]]] = {}
            cidx = 0
            cest = 2
            for (c_lem, c_cpos, r_ctx), cands in sorted(self._by_lemma_ctx.items()):
                ckey = f"{c_lem}|{c_cpos}|{r_ctx}"
                crows: list[list[str | float]] = [
                    [
                        c.greek_lemma,
                        c.greek_pos,
                        c.score,
                        c.prob_g_given_c,
                        c.prob_c_given_g,
                        c.provenance,
                        c.count_nt,
                        c.count_ot,
                    ]
                    for c in cands
                ]
                scrows = json.dumps(crows, ensure_ascii=False, separators=(",", ":"))
                cr_sz = len(ckey.encode("utf-8")) + len(scrows.encode("utf-8")) + 6
                if cest + cr_sz > max_bytes and cshard:
                    cp = out_dir / f"lemma-ctx-{cidx}.json"
                    cp.write_text(
                        json.dumps(cshard, ensure_ascii=False, separators=(",", ":"))
                        + "\n",
                        encoding="utf-8",
                    )
                    files.append(cp)
                    cshard = {}
                    cidx += 1
                    cest = 2
                cshard[ckey] = crows
                cest += cr_sz
            if cshard:
                cp = out_dir / f"lemma-ctx-{cidx}.json"
                cp.write_text(
                    json.dumps(cshard, ensure_ascii=False, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                files.append(cp)
        return files

    @classmethod
    def load_shards(cls, out_dir: Path) -> LexicalTranslationModel:
        """Load aligned lexical translation tables offline."""
        by_lemma_pos: dict[tuple[str, str], list[LexicalCandidate]] = {}
        by_lemma_acc: dict[str, dict[tuple[str, str], LexicalCandidate]] = defaultdict(
            dict
        )
        for p in sorted(out_dir.glob("lemma-align-*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            for key, rows in data.items():
                c_lem, _, c_cpos = key.partition("|")
                cands = [
                    LexicalCandidate(
                        greek_lemma=str(r[0]),
                        greek_pos=str(r[1]),
                        score=float(r[2]),
                        prob_g_given_c=float(r[3]),
                        prob_c_given_g=float(r[4]),
                        provenance=str(r[5]),
                        count_nt=float(r[6]),
                        count_ot=float(r[7]),
                    )
                    for r in rows
                ]
                by_lemma_pos[(c_lem, c_cpos)] = cands
                for c in cands:
                    prev = by_lemma_acc[c_lem].get((c.greek_lemma, c.greek_pos))
                    if prev is None or c.score > prev.score:
                        by_lemma_acc[c_lem][(c.greek_lemma, c.greek_pos)] = c
        by_lemma_ctx: dict[tuple[str, str, str], list[LexicalCandidate]] = {}
        for p in sorted(out_dir.glob("lemma-ctx-*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            for key, rows in data.items():
                parts = key.split("|", 2)
                if len(parts) != 3:
                    continue
                c_lem, c_cpos, r_ctx = parts
                by_lemma_ctx[(c_lem, c_cpos, r_ctx)] = [
                    LexicalCandidate(
                        greek_lemma=str(r[0]),
                        greek_pos=str(r[1]),
                        score=float(r[2]),
                        prob_g_given_c=float(r[3]),
                        prob_c_given_g=float(r[4]),
                        provenance=str(r[5]),
                        count_nt=float(r[6]),
                        count_ot=float(r[7]),
                    )
                    for r in rows
                ]
        by_lemma = {
            k: sorted(v.values(), key=lambda x: -x.score)[:8]
            for k, v in by_lemma_acc.items()
        }
        return cls(
            by_lemma_pos=by_lemma_pos,
            by_lemma=by_lemma,
            by_lemma_ctx=by_lemma_ctx,
        )
