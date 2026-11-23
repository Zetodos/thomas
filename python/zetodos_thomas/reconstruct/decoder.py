# SPDX-License-Identifier: Apache-2.0
"""Deterministic log-linear beam decoder for Koine Greek reconstruction."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from zetodos_thomas.align.lemma_aligner import LexicalTranslationModel
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.lexicon.idioms import IdiomAndValencyTable
from zetodos_thomas.morphology.greek_inflector import (
    GreekInflector,
    realize_greek_tokens,
)
from zetodos_thomas.reconstruct.koine_lm import KoineNgramLM, morph_lm_key
from zetodos_thomas.reconstruct.transfer import (
    SlotCandidate,
    TransferSlot,
    build_unit_transfer_clauses,
)
from zetodos_thomas.schemas.generated import (
    GeneratedCorpus,
    GeneratedToken,
    GeneratedUnit,
    TokenAlternative,
)
from zetodos_thomas.schemas.linguistic import CopticUnit, GreekMorphFeatures


@dataclass(frozen=True)
class DecoderWeights:
    """Log-linear feature weights tuned on NT dev split (never holdout/P.Oxy.)."""

    w_lex: float = 1.0
    w_lm: float = 0.38
    w_agree: float = 0.85
    w_valency: float = 0.70
    w_omit_pen: float = 0.95
    beam_size: int = 8

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> DecoderWeights:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


@dataclass(frozen=True)
class _BeamItem:
    surface: str
    lemma: str
    morph: GreekMorphFeatures
    coptic_seqs: tuple[int, ...]
    provenance: str
    slot_idx: int
    cand_score: float
    alternatives: tuple[TokenAlternative, ...]


@dataclass(frozen=True)
class _BeamState:
    score: float
    k0: str
    k1: str
    s1: str
    expected_gender: str | None
    verb_obj_case: str
    active_verb_lemma: str
    items: tuple[_BeamItem, ...]


def _resolve_gender(
    lemma: str,
    *,
    inflector: GreekInflector,
    ccl_lexicon: CclLexicon,
    fallback: str | None = None,
) -> str:
    return (
        inflector.lemma_gender(lemma)
        or ccl_lexicon.greek_gender(lemma)
        or fallback
        or "M"
    )


def _inflect_slot(
    cand: SlotCandidate,
    morph: GreekMorphFeatures,
    *,
    inflector: GreekInflector,
) -> str:
    if cand.fixed_surface:
        return cand.fixed_surface
    return inflector.inflect(cand.lemma, morph)


def _confidence_from_cands(cands: list[SlotCandidate]) -> float:
    if not cands:
        return 0.0
    if len(cands) == 1:
        return round(1.0 / (1.0 + math.exp(-max(min(cands[0].score, 6.0), -6.0))), 4)
    top = cands[0].score
    exps = [math.exp(max(min(c.score - top, 0.0), -12.0)) for c in cands[:4]]
    return round(exps[0] / max(sum(exps), 1e-9), 4)


class KoineBeamDecoder:
    """Deterministic log-linear beam decoder for Coptic -> Koine Greek."""

    def __init__(
        self,
        *,
        lexicon: LexicalTranslationModel,
        ccl_lexicon: CclLexicon,
        idioms: IdiomAndValencyTable,
        inflector: GreekInflector,
        lm: KoineNgramLM,
        weights: DecoderWeights | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.ccl_lexicon = ccl_lexicon
        self.idioms = idioms
        self.inflector = inflector
        self.lm = lm
        self.weights = weights or DecoderWeights()

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> KoineBeamDecoder:
        """Load complete frozen reconstruction model offline from ``model_dir``."""
        inflector = GreekInflector.load_shards(model_dir)
        lexicon = LexicalTranslationModel.load_shards(model_dir)
        ccl_lexicon = CclLexicon.load_json(model_dir / "ccl-loanwords.json")
        idioms = IdiomAndValencyTable.load_json(model_dir / "idiom-valency.json")
        lm = KoineNgramLM.load_shards(model_dir)
        weights_p = model_dir / "decoder-weights.json"
        weights = (
            DecoderWeights.load_json(weights_p)
            if weights_p.is_file()
            else DecoderWeights()
        )
        return cls(
            lexicon=lexicon,
            ccl_lexicon=ccl_lexicon,
            idioms=idioms,
            inflector=inflector,
            lm=lm,
            weights=weights,
        )

    def _decode_clause(
        self, clause_slots: list[TransferSlot]
    ) -> tuple[list[_BeamItem], list[str]]:
        """Run deterministic beam search over one clause's ``TransferSlot``s."""
        w = self.weights
        beams: list[_BeamState] = [
            _BeamState(
                score=0.0,
                k0="<s>",
                k1="<s>",
                s1="<s>",
                expected_gender=None,
                verb_obj_case="A",
                active_verb_lemma="",
                items=(),
            )
        ]

        n_slots = len(clause_slots)
        for s_idx, slot in enumerate(clause_slots):
            next_beams: list[_BeamState] = []
            nxt_slot = clause_slots[s_idx + 1] if s_idx + 1 < n_slots else None

            # Determine which target genders a pre-nominal determiner should explore
            if (
                slot.agree_head_offset == 1
                and nxt_slot is not None
                and nxt_slot.candidates
            ):
                head_genders: list[str] = []
                for hc in nxt_slot.candidates[:4]:
                    g = _resolve_gender(
                        hc.lemma,
                        inflector=self.inflector,
                        ccl_lexicon=self.ccl_lexicon,
                        fallback=slot.target_morph.gender,
                    )
                    if g not in head_genders:
                        head_genders.append(g)
                gender_choices: Sequence[str | None] = head_genders[:3]
            else:
                gender_choices = [None]

            cands = slot.candidates[:5]
            # Build top-3 TokenAlternatives template for metadata
            top_alts = tuple(
                TokenAlternative(
                    surface=_inflect_slot(
                        c,
                        c.morph_override or slot.target_morph,
                        inflector=self.inflector,
                    ),
                    lemma=c.lemma,
                    morph_sig=(c.morph_override or slot.target_morph).signature(),
                    score=round(c.score, 4),
                    provenance=c.provenance,
                )
                for c in cands[:3]
            )
            slot_conf = _confidence_from_cands(cands)

            for st in beams:
                # Option 1: Omit optional function word (quotative or copula)
                if slot.optional_omit:
                    next_beams.append(
                        _BeamState(
                            score=st.score - w.w_omit_pen,
                            k0=st.k0,
                            k1=st.k1,
                            s1=st.s1,
                            expected_gender=st.expected_gender,
                            verb_obj_case=st.verb_obj_case,
                            active_verb_lemma=st.active_verb_lemma,
                            items=st.items,
                        )
                    )

                for cand in cands:
                    base_morph = cand.morph_override or slot.target_morph
                    # Adjust POS from candidate when noun/adj differs
                    if base_morph.pos != cand.pos and slot.role in {
                        "NOUN",
                        "ADV",
                        "ADJ",
                    }:
                        base_morph = base_morph.model_copy(update={"pos": cand.pos})

                    for forced_gender in gender_choices:
                        cur_morph = base_morph
                        agree_bonus = 0.0
                        valency_bonus = 0.0

                        # Apply verb-governed case (e.g. γεύομαι -> Genitive)
                        if slot.case_governor == "verb_obj" and cur_morph.case:
                            cur_morph = cur_morph.model_copy(
                                update={"case": st.verb_obj_case}
                            )

                        # Enforce & score head-noun gender agreement
                        if forced_gender is not None:
                            cur_morph = cur_morph.model_copy(
                                update={"gender": forced_gender}
                            )
                            next_exp_g = forced_gender
                        else:
                            next_exp_g = None
                            if slot.role == "NOUN":
                                head_g = _resolve_gender(
                                    cand.lemma,
                                    inflector=self.inflector,
                                    ccl_lexicon=self.ccl_lexicon,
                                    fallback=cur_morph.gender,
                                )
                                cur_morph = cur_morph.model_copy(
                                    update={"gender": head_g}
                                )
                                if st.expected_gender is not None:
                                    agree_bonus = (
                                        w.w_agree
                                        if head_g == st.expected_gender
                                        else -1.5 * w.w_agree
                                    )

                        # Track active verb's governed object case & lemma for PMI
                        next_vcase = st.verb_obj_case
                        next_vlem = (
                            cand.lemma
                            if slot.role in {"VERB", "PART"}
                            else st.active_verb_lemma
                        )
                        if slot.role in {"NOUN", "ADJ"} and st.active_verb_lemma:
                            valency_bonus += 0.35 * self.lm.score_collocation(
                                st.active_verb_lemma, cand.lemma
                            )
                        if slot.role == "VERB":
                            next_vcase = (
                                cand.required_case
                                or self.idioms.verb_object_case(cand.lemma)
                            )
                        elif slot.role == "PREP":
                            next_vcase = cand.required_case or self.idioms.prep_case(
                                cand.lemma
                            )

                        surface = _inflect_slot(
                            cand, cur_morph, inflector=self.inflector
                        )
                        if not surface:
                            continue

                        if slot.is_punct:
                            lm_sc = 0.0
                            nk0, nk1, ns1 = st.k0, st.k1, st.s1
                        else:
                            k2 = morph_lm_key(cand.lemma, cur_morph)
                            lm_sc = w.w_lm * self.lm.score_step(
                                k0=st.k0,
                                k1=st.k1,
                                k2=k2,
                                s1=st.s1,
                                s2=surface.lower(),
                            )
                            nk0, nk1, ns1 = st.k1, k2, surface.lower()

                        step_score = (
                            w.w_lex * cand.score
                            + lm_sc
                            + agree_bonus
                            + valency_bonus * w.w_valency
                        )
                        item = _BeamItem(
                            surface=surface,
                            lemma=cand.lemma,
                            morph=cur_morph,
                            coptic_seqs=tuple(slot.coptic_seqs),
                            provenance=cand.provenance,
                            slot_idx=s_idx,
                            cand_score=round(slot_conf, 4),
                            alternatives=top_alts,
                        )
                        next_beams.append(
                            _BeamState(
                                score=st.score + step_score,
                                k0=nk0,
                                k1=nk1,
                                s1=ns1,
                                expected_gender=next_exp_g,
                                verb_obj_case=next_vcase,
                                active_verb_lemma=next_vlem,
                                items=(*st.items, item),
                            )
                        )

            # Prune deterministically (score desc, then canonical Greek Unicode tuple)
            if next_beams:
                next_beams.sort(
                    key=lambda b: (-b.score, tuple(it.surface for it in b.items))
                )
                beams = next_beams[: w.beam_size]

        # Score sentence end </s> via LM
        final_ranked = sorted(
            beams,
            key=lambda b: (
                -(
                    b.score
                    + 0.25
                    * w.w_lm
                    * self.lm.score_step(
                        k0=b.k0, k1=b.k1, k2="</s>", s1=b.s1, s2="</s>"
                    )
                ),
                tuple(it.surface for it in b.items),
            ),
        )
        best = final_ranked[0]
        alt_strings: list[str] = []
        seen_strs: set[str] = set()
        for b in final_ranked[:5]:
            s = realize_greek_tokens([it.surface for it in b.items])
            if s and s not in seen_strs:
                seen_strs.add(s)
                alt_strings.append(s)
            if len(alt_strings) >= 3:
                break
        return list(best.items), alt_strings

    def decode_unit(self, unit: CopticUnit) -> GeneratedUnit:
        """Deterministically reconstruct Koine Greek for one ``CopticUnit``."""
        clauses = build_unit_transfer_clauses(
            unit,
            lexicon=self.lexicon,
            ccl_lexicon=self.ccl_lexicon,
            idioms=self.idioms,
            inflector=self.inflector,
        )

        tokens: list[GeneratedToken] = []
        all_surfaces: list[str] = []
        clause_alts: list[str] = []
        seq = 0
        for clause_slots in clauses:
            items, alts = self._decode_clause(clause_slots)
            clause_alts.extend(alts[:1])
            for it in items:
                tokens.append(
                    GeneratedToken(
                        seq=seq,
                        surface=it.surface,
                        lemma=it.lemma,
                        morph=it.morph,
                        coptic_seqs=list(it.coptic_seqs),
                        provenance=it.provenance,
                        confidence=it.cand_score,
                        alternatives=list(it.alternatives),
                    )
                )
                all_surfaces.append(it.surface)
                seq += 1

        # Guarantee zero empty outputs
        if not tokens:
            surf = "καί"
            tokens.append(
                GeneratedToken(
                    seq=0,
                    surface=surf,
                    lemma="καί",
                    morph=GreekMorphFeatures(pos="C-"),
                    coptic_seqs=[0] if unit.morphemes else [],
                    provenance="C",
                    confidence=0.1,
                )
            )
            all_surfaces.append(surf)

        greek_text = realize_greek_tokens(all_surfaces)
        # Capitalize clause/saying start if lowercase Greek
        if greek_text and greek_text[0].isalpha() and greek_text[0].islower():
            greek_text = greek_text[0].upper() + greek_text[1:]

        return GeneratedUnit(
            unit_id=unit.unit_id,
            coptic_source_id=unit.source_id,
            coptic_text=unit.normalized_text,
            greek_text=greek_text,
            tokens=tokens,
            clause_alternatives=clause_alts[:3],
        )

    def decode_corpus(
        self,
        units: list[CopticUnit],
        *,
        corpus_id: str = "thomas-generated-v1",
        title: str = "Category C Koine Greek Reconstructions (Gospel of Thomas)",
        model_id: str = "model-v1",
    ) -> GeneratedCorpus:
        """Decode a full list of ``CopticUnit``s cleanly."""
        return GeneratedCorpus(
            id=corpus_id,
            title=title,
            model_id=model_id,
            units=[self.decode_unit(u) for u in units],
        )


def tune_weights_on_dev(
    *,
    decoder: KoineBeamDecoder,
    dev_pairs: list[tuple[CopticUnit, list[str]]],
) -> DecoderWeights:
    """Coordinate search on 51-verse NT DEV split ONLY (optimizing Lemma F1)."""
    if not dev_pairs:
        return decoder.weights

    def _eval_dev(w: DecoderWeights) -> float:
        decoder.weights = w
        tp = 0
        pred_n = 0
        gold_n = 0
        for c_u, gold_lemmas in dev_pairs:
            gen = decoder.decode_unit(c_u)
            pred_lemmas = [
                t.lemma
                for t in gen.tokens
                if t.morph.pos not in {"X-", "RA"}
                and t.lemma not in {".", ",", ";", "·"}
            ]
            g_filt = [g for g in gold_lemmas if g not in {".", ",", ";", "·"}]
            p_set: dict[str, int] = {}
            for x in pred_lemmas:
                p_set[x] = p_set.get(x, 0) + 1
            for g in g_filt:
                if p_set.get(g, 0) > 0:
                    tp += 1
                    p_set[g] -= 1
            pred_n += len(pred_lemmas)
            gold_n += len(g_filt)
        p = tp / max(pred_n, 1)
        r = tp / max(gold_n, 1)
        return 2 * p * r / max(p + r, 1e-9)

    best_w = decoder.weights
    best_sc = _eval_dev(best_w)
    grid_lm = (0.25, 0.38, 0.55)
    grid_omit = (0.45, 0.85, 1.25)
    grid_agree = (0.5, 0.85, 1.2)
    for w_lm in grid_lm:
        for w_omit in grid_omit:
            for w_agree in grid_agree:
                cand_w = DecoderWeights(
                    w_lex=best_w.w_lex,
                    w_lm=w_lm,
                    w_agree=w_agree,
                    w_valency=best_w.w_valency,
                    w_omit_pen=w_omit,
                    beam_size=8,
                )
                sc = _eval_dev(cand_w)
                if sc > best_sc:
                    best_sc = sc
                    best_w = cand_w
    decoder.weights = best_w
    return best_w
