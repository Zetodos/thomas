# SPDX-License-Identifier: Apache-2.0
"""Minimal deterministic decoder (JAK-39).

Consumes JAK-37 lexical candidates and JAK-38 statistical constructions /
verified loanwords / ablation arms without collapsing evidence classes.
Prefers attested Greek; abstains when evidence is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from zetodos_thomas.reconstruct.model import DecodeModel, normalized_loanword_greek
from zetodos_thomas.reconstruct.ranking import (
    RANKING_RULES_ID,
    RankingFeatures,
    RankingKey,
    ranking_key,
    score_from_key,
    uncertainty_for,
)
from zetodos_thomas.schemas.decode import (
    AbstentionReason,
    CandidateEvidenceRef,
    DecodeRequest,
    DecodeResult,
    DecoderEvidenceKind,
    DecodeSpanInput,
    GreekCandidate,
    SpanDecodeResult,
    SpanDecodeStatus,
)
from zetodos_thomas.schemas.lexicon import ObservedWordformCandidate
from zetodos_thomas.schemas.mappings import (
    AblationCondition,
    EvidenceClass,
    SpanRef,
    StatisticalConstruction,
    VerifiedLoanword,
)

# Local article set for light grammatical notes (mirrors JAK-38 phrase helpers).
_GREEK_ART = frozenset(
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
_GREEK_NEG = frozenset({"ου", "ουκ", "ουχ", "μη", "μηδε", "ουδε", "ουτε", "μητε"})


def _dual_aligner(aligners: list[str]) -> bool:
    return set(aligners) >= {"eflomal", "fast_align"}


@dataclass(frozen=True, slots=True)
class _RawCandidate:
    greek_surfaces: tuple[str, ...]
    kind: DecoderEvidenceKind
    evidence_class: EvidenceClass | None
    ref_id: str
    layer: str | None
    dual_aligner: bool
    observation_count: int
    heuristic_ranking_score: float
    grammatical_notes: tuple[str, ...]
    restrictions: str
    notes: str = ""


def _grammatical_notes_for_construction(
    row: StatisticalConstruction,
) -> tuple[str, ...]:
    notes: list[str] = [f"family={row.family.value}"]
    if row.greek_pattern and row.greek_pattern[0] in _GREEK_ART:
        notes.append("greek_pattern_starts_with_article")
    if any(t in _GREEK_NEG for t in row.greek_pattern):
        notes.append("greek_pattern_contains_negation")
    if row.family.value == "article" and len(row.coptic_pattern) >= 2:
        notes.append("det_noun_construction")
    return tuple(notes)


def _from_construction(row: StatisticalConstruction) -> _RawCandidate:
    return _RawCandidate(
        greek_surfaces=tuple(row.greek_pattern),
        kind=DecoderEvidenceKind.STATISTICAL_CONSTRUCTION,
        evidence_class=EvidenceClass.STATISTICAL,
        ref_id=row.construction_id,
        layer=None,
        dual_aligner=_dual_aligner(row.contributing_aligners),
        observation_count=row.observation_count,
        heuristic_ranking_score=0.0,
        grammatical_notes=_grammatical_notes_for_construction(row),
        restrictions=(
            "Statistical train-aligner construction — not independently verified."
        ),
    )


def _from_observed(
    row: ObservedWordformCandidate,
    *,
    heuristic: float,
) -> _RawCandidate:
    return _RawCandidate(
        greek_surfaces=(row.greek_wordform,),
        kind=DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM,
        evidence_class=None,
        ref_id=f"observed:{row.coptic_wordform}->{row.greek_wordform}",
        layer="observed_wordform",
        dual_aligner=_dual_aligner(row.contributing_aligners),
        observation_count=row.observation_count,
        heuristic_ranking_score=heuristic,
        grammatical_notes=("attested_train_wordform",),
        restrictions="Attested train surface only — morphology not generated.",
    )


def _from_loanword(entry: VerifiedLoanword, *, matched_form: str) -> _RawCandidate:
    greek = normalized_loanword_greek(entry)
    return _RawCandidate(
        greek_surfaces=(greek,),
        kind=DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD,
        evidence_class=EvidenceClass.INDEPENDENTLY_VERIFIED,
        ref_id=entry.loanword_id,
        layer=None,
        dual_aligner=True,  # not aligner-derived; treat as independent support
        observation_count=1,
        heuristic_ranking_score=0.0,
        grammatical_notes=(
            "verified_loanword_lemma",
            f"matched_coptic_form={matched_form}",
        ),
        restrictions=entry.restrictions,
        notes="Illustrative NT spans are non-training and are not consumed.",
    )


def _materialize(
    raws: list[_RawCandidate],
    *,
    max_candidates: int,
) -> list[GreekCandidate]:
    # Dedup by (kind, greek_surfaces, ref_id) keeping best ranking key.
    best: dict[tuple[str, tuple[str, ...], str], tuple[RankingKey, _RawCandidate]] = {}
    for raw in raws:
        features = RankingFeatures(
            kind=raw.kind,
            dual_aligner=raw.dual_aligner,
            observation_count=raw.observation_count,
            heuristic_ranking_score=raw.heuristic_ranking_score,
            greek_surfaces=raw.greek_surfaces,
            ref_id=raw.ref_id,
        )
        key = ranking_key(features)
        dedup = (raw.kind.value, raw.greek_surfaces, raw.ref_id)
        prev = best.get(dedup)
        if prev is None or key < prev[0]:
            best[dedup] = (key, raw)

    ordered = sorted(best.values(), key=lambda item: item[0])
    out: list[GreekCandidate] = []
    for rank, (key, raw) in enumerate(ordered[:max_candidates], start=1):
        out.append(
            GreekCandidate(
                rank=rank,
                greek_surfaces=list(raw.greek_surfaces),
                score=score_from_key(key),
                uncertainty=uncertainty_for(
                    kind=raw.kind,
                    dual_aligner=raw.dual_aligner,
                    observation_count=raw.observation_count,
                ),
                evidence=[
                    CandidateEvidenceRef(
                        kind=raw.kind,
                        evidence_class=raw.evidence_class,
                        ref_id=raw.ref_id,
                        layer=raw.layer,
                        notes=raw.notes,
                    )
                ],
                ranking_key=list(key),
                grammatical_notes=list(raw.grammatical_notes),
                restrictions=raw.restrictions,
            )
        )
    return out


def _collect_for_surfaces(
    model: DecodeModel,
    surfaces: list[str],
    *,
    construction_id: str | None,
) -> tuple[list[_RawCandidate], AbstentionReason | None, str]:
    """Gather raw candidates or an abstention reason for one surface sequence."""
    if not surfaces:
        return [], AbstentionReason.EMPTY_SPAN, "span has no Coptic surfaces"

    pattern = tuple(surfaces)
    raws: list[_RawCandidate] = []

    if construction_id is not None:
        row = model.constructions_by_id.get(construction_id)
        if row is None:
            return (
                [],
                AbstentionReason.CONSTRUCTION_ID_UNKNOWN,
                f"unknown construction_id={construction_id!r}",
            )
        if tuple(row.coptic_pattern) != pattern:
            return (
                [],
                AbstentionReason.CONSTRUCTION_PATTERN_MISMATCH,
                (
                    f"construction {construction_id!r} pattern "
                    f"{row.coptic_pattern!r} != surfaces {list(surfaces)!r}"
                ),
            )
        raws.append(_from_construction(row))
        return raws, None, ""

    for row in model.constructions_by_pattern.get(pattern, []):
        raws.append(_from_construction(row))

    if len(surfaces) == 1:
        surface = surfaces[0]
        for row in model.observed_by_coptic.get(surface, []):
            heur = model.heuristic_by_pair.get(
                (row.coptic_wordform, row.greek_wordform), 0.0
            )
            raws.append(_from_observed(row, heuristic=heur))

        loan_hits = model.loanwords_by_form.get(surface, [])
        if loan_hits:
            if model.includes_verified_loanwords:
                for entry in loan_hits:
                    raws.append(_from_loanword(entry, matched_form=surface))
            elif not raws:
                # Only loanword evidence would apply, but arm A excludes it.
                return (
                    [],
                    AbstentionReason.ABLATION_EXCLUDES_LOANWORDS,
                    (
                        f"form {surface!r} only matches verified loanwords; "
                        "ablation arm A omits independently_verified constraints"
                    ),
                )

    if not raws:
        return (
            [],
            AbstentionReason.NO_ATTESTED_EVIDENCE,
            "no statistical construction, observed wordform, or allowed loanword",
        )
    return raws, None, ""


def decode_span(
    model: DecodeModel,
    span: DecodeSpanInput,
    *,
    max_candidates: int,
    request_unit_id: str,
) -> SpanDecodeResult:
    """Decode one span; abstain rather than invent Greek."""
    unit_id = span.unit_id or request_unit_id
    # Query unit may be Thomas (blinded reconstruction target). Forbidden markers
    # block *evidence* leakage elsewhere; querying thomas.* is allowed. We still
    # refuse if the caller somehow embeds poxy Greek unit ids as the decode unit
    # of a span that claims to be an evidence source — decode inputs are Coptic.
    if unit_id and "poxy" in unit_id.lower().replace(" ", ""):
        return SpanDecodeResult(
            span_id=span.span_id,
            unit_id=unit_id,
            surfaces=list(span.surfaces),
            construction_id=span.construction_id,
            status=SpanDecodeStatus.ABSTAIN,
            abstention_reason=AbstentionReason.FORBIDDEN_QUERY_UNIT,
            abstention_detail=(
                "refusing decode unit ids that embed P.Oxy. Greek fragment handles"
            ),
            input_span=_as_span_ref(span, unit_id),
        )

    raws, reason, detail = _collect_for_surfaces(
        model,
        list(span.surfaces),
        construction_id=span.construction_id,
    )
    if reason is not None:
        return SpanDecodeResult(
            span_id=span.span_id,
            unit_id=unit_id,
            surfaces=list(span.surfaces),
            construction_id=span.construction_id,
            status=SpanDecodeStatus.ABSTAIN,
            abstention_reason=reason,
            abstention_detail=detail,
            input_span=_as_span_ref(span, unit_id),
        )

    candidates = _materialize(raws, max_candidates=max_candidates)
    if not candidates:
        return SpanDecodeResult(
            span_id=span.span_id,
            unit_id=unit_id,
            surfaces=list(span.surfaces),
            construction_id=span.construction_id,
            status=SpanDecodeStatus.ABSTAIN,
            abstention_reason=AbstentionReason.NO_ATTESTED_EVIDENCE,
            abstention_detail="ranking produced no candidates",
            input_span=_as_span_ref(span, unit_id),
        )
    return SpanDecodeResult(
        span_id=span.span_id,
        unit_id=unit_id,
        surfaces=list(span.surfaces),
        construction_id=span.construction_id,
        status=SpanDecodeStatus.CANDIDATES,
        candidates=candidates,
        input_span=_as_span_ref(span, unit_id),
    )


def _as_span_ref(span: DecodeSpanInput, unit_id: str) -> SpanRef | None:
    if not span.token_ids:
        return None
    return SpanRef(
        span_id=span.span_id,
        source_id=span.source_id or "decode-input",
        unit_id=unit_id,
        token_ids=list(span.token_ids),
        surfaces=list(span.surfaces),
    )


def decode_request(model: DecodeModel, request: DecodeRequest) -> DecodeResult:
    """Run the deterministic decoder over a request document."""
    if model.condition is not request.ablation_condition:
        msg = (
            f"model condition {model.condition.value!r} != "
            f"request {request.ablation_condition.value!r}"
        )
        raise ValueError(msg)

    span_results = [
        decode_span(
            model,
            span,
            max_candidates=request.max_candidates_per_span,
            request_unit_id=request.unit_id,
        )
        for span in request.spans
    ]
    n_cand = sum(1 for s in span_results if s.status is SpanDecodeStatus.CANDIDATES)
    n_abs = sum(1 for s in span_results if s.status is SpanDecodeStatus.ABSTAIN)
    return DecodeResult(
        id=f"decode-result:{request.id}",
        title=request.title or f"Decode result for {request.id}",
        request_id=request.id,
        ablation_condition=request.ablation_condition,
        unit_id=request.unit_id,
        spans=span_results,
        ranking_rules_id=RANKING_RULES_ID,
        phase2_active_aligners=["eflomal", "fast_align"],
        adopted_aligner=None,
        summary={
            "span_count": len(span_results),
            "spans_with_candidates": n_cand,
            "spans_abstained": n_abs,
            "ablation_condition": request.ablation_condition.value,
            "includes_verified_loanwords": model.includes_verified_loanwords,
            "max_candidates_per_span": request.max_candidates_per_span,
        },
        provenance={
            **model.provenance,
            "request_id": request.id,
            "ranking_rules_id": RANKING_RULES_ID,
            "llm": False,
            "network": False,
        },
        notes=[
            "Minimal Phase 2 feasibility decoder (JAK-39).",
            "Does not invent unsupported Greek; abstention is explicit.",
            "Statistical constructions and verified loanwords stay separate.",
            "Keep-both aligners: eflomal + fast_align (adopted_aligner=null).",
            *request.notes,
        ],
    )


def request_from_surfaces(
    *,
    request_id: str,
    unit_id: str,
    surfaces: list[str],
    ablation_condition: AblationCondition | str,
    construction_id: str | None = None,
    span_id: str | None = None,
    max_candidates_per_span: int = 5,
) -> DecodeRequest:
    """Helper: wrap a single surface sequence as a DecodeRequest."""
    cond = (
        ablation_condition
        if isinstance(ablation_condition, AblationCondition)
        else AblationCondition(ablation_condition)
    )
    sid = span_id or f"v1:decode-input/{unit_id}/s/0"
    return DecodeRequest(
        id=request_id,
        ablation_condition=cond,
        unit_id=unit_id,
        spans=[
            DecodeSpanInput(
                span_id=sid,
                source_id="decode-input",
                unit_id=unit_id,
                surfaces=surfaces,
                construction_id=construction_id,
            )
        ],
        max_candidates_per_span=max_candidates_per_span,
    )
