# SPDX-License-Identifier: Apache-2.0
"""Minimal deterministic decoder (JAK-39).

Consumes JAK-37 lexical candidates and JAK-38 statistical constructions /
verified loanwords / ablation arms without collapsing evidence classes.
Prefers attested Greek; abstains when evidence is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zetodos_thomas.lexicon.candidates import FORBIDDEN_UNIT_MARKERS
from zetodos_thomas.reconstruct.model import DecodeModel, normalized_loanword_greek
from zetodos_thomas.reconstruct.ranking import (
    RANKING_RULES_ID,
    RankingFeatures,
    RankingKey,
    observed_row_sufficient,
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
    GreekFormKind,
    SpanDecodeResult,
    SpanDecodeStatus,
)
from zetodos_thomas.schemas.ids import parse_span_id, parse_token_id
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


def _unit_forbidden(unit_id: str) -> bool:
    lowered = unit_id.lower().replace(" ", "")
    return any(marker in lowered for marker in FORBIDDEN_UNIT_MARKERS) or (
        "p.oxy" in lowered
    )


def _fragment_forbidden(unit_id: str) -> bool:
    """P.Oxy. Greek fragment handles must not be decode query units."""
    lowered = unit_id.lower().replace(" ", "")
    return "poxy" in lowered or "p.oxy" in lowered


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
    form_kind: GreekFormKind
    notes: str = ""
    hint_matched: bool = False
    source_unit_ids: tuple[str, ...] = ()


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


def _from_construction(
    row: StatisticalConstruction,
    *,
    hint_matched: bool = False,
) -> _RawCandidate:
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
        form_kind=GreekFormKind.ATTESTED_SURFACE,
        hint_matched=hint_matched,
        source_unit_ids=tuple(row.source_unit_ids),
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
        form_kind=GreekFormKind.ATTESTED_SURFACE,
        source_unit_ids=tuple(row.source_unit_ids),
    )


def _from_loanword(entry: VerifiedLoanword, *, matched_form: str) -> _RawCandidate:
    greek = normalized_loanword_greek(entry)
    return _RawCandidate(
        greek_surfaces=(greek,),
        kind=DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD,
        evidence_class=EvidenceClass.INDEPENDENTLY_VERIFIED,
        ref_id=entry.loanword_id,
        layer=None,
        dual_aligner=True,  # not aligner-derived; independent L-support
        observation_count=1,
        heuristic_ranking_score=0.0,
        grammatical_notes=(
            "verified_loanword_lemma",
            f"matched_coptic_form={matched_form}",
        ),
        restrictions=entry.restrictions,
        form_kind=GreekFormKind.LEMMA_CLUE,
        notes=(
            "Lemma-level L-clue only — not an attested surface inflection; "
            "illustrative NT spans are non-training and are not consumed."
        ),
    )


@dataclass
class _HypothesisAccum:
    """Aggregate independent evidence for one Greek output sequence."""

    greek_surfaces: tuple[str, ...]
    raws: list[_RawCandidate] = field(default_factory=list)
    best_key: RankingKey | None = None
    best_raw: _RawCandidate | None = None


def _materialize(
    raws: list[_RawCandidate],
    *,
    max_candidates: int,
) -> list[GreekCandidate]:
    """Deduplicate by normalized Greek hypothesis; aggregate evidence refs."""
    by_greek: dict[tuple[str, ...], _HypothesisAccum] = {}
    for raw in raws:
        features = RankingFeatures(
            kind=raw.kind,
            dual_aligner=raw.dual_aligner,
            observation_count=raw.observation_count,
            heuristic_ranking_score=raw.heuristic_ranking_score,
            greek_surfaces=raw.greek_surfaces,
            ref_id=raw.ref_id,
            hint_matched=raw.hint_matched,
        )
        key = ranking_key(features)
        accum = by_greek.get(raw.greek_surfaces)
        if accum is None:
            accum = _HypothesisAccum(greek_surfaces=raw.greek_surfaces)
            by_greek[raw.greek_surfaces] = accum
        accum.raws.append(raw)
        if accum.best_key is None or key < accum.best_key:
            accum.best_key = key
            accum.best_raw = raw

    ordered = sorted(
        by_greek.values(),
        key=lambda a: (
            a.best_key
            if a.best_key is not None
            else ranking_key(
                RankingFeatures(
                    kind=DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM,
                    dual_aligner=False,
                    observation_count=0,
                    heuristic_ranking_score=0.0,
                    greek_surfaces=a.greek_surfaces,
                    ref_id="",
                )
            )
        ),
    )
    out: list[GreekCandidate] = []
    for rank, accum in enumerate(ordered[:max_candidates], start=1):
        assert accum.best_key is not None and accum.best_raw is not None
        # Stable evidence order: by ranking key of each raw, then ref_id.
        evidence_raws = sorted(
            accum.raws,
            key=lambda r: ranking_key(
                RankingFeatures(
                    kind=r.kind,
                    dual_aligner=r.dual_aligner,
                    observation_count=r.observation_count,
                    heuristic_ranking_score=r.heuristic_ranking_score,
                    greek_surfaces=r.greek_surfaces,
                    ref_id=r.ref_id,
                    hint_matched=r.hint_matched,
                )
            ),
        )
        # Prefer attested_surface form_kind if any contributing evidence is attested.
        form_kind = GreekFormKind.LEMMA_CLUE
        for r in evidence_raws:
            if r.form_kind is GreekFormKind.ATTESTED_SURFACE:
                form_kind = GreekFormKind.ATTESTED_SURFACE
                break
        notes: list[str] = []
        for r in evidence_raws:
            for n in r.grammatical_notes:
                if n not in notes:
                    notes.append(n)
        restrictions = accum.best_raw.restrictions
        if (
            any(
                r.kind is DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD
                for r in evidence_raws
            )
            and form_kind is GreekFormKind.ATTESTED_SURFACE
        ):
            restrictions = (
                "Attested surface preferred; verified loanword L-clue also "
                "supports this hypothesis but does not invent inflection."
            )
        out.append(
            GreekCandidate(
                rank=rank,
                greek_surfaces=list(accum.greek_surfaces),
                form_kind=form_kind,
                score=score_from_key(accum.best_key),
                uncertainty=uncertainty_for(
                    kind=accum.best_raw.kind,
                    dual_aligner=accum.best_raw.dual_aligner,
                    observation_count=accum.best_raw.observation_count,
                ),
                evidence=[
                    CandidateEvidenceRef(
                        kind=r.kind,
                        evidence_class=r.evidence_class,
                        ref_id=r.ref_id,
                        layer=r.layer,
                        notes=r.notes,
                    )
                    for r in evidence_raws
                ],
                ranking_key=list(accum.best_key),
                grammatical_notes=notes,
                restrictions=restrictions,
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
    hinted_id: str | None = None

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
        # Valid hint: boost that construction; still gather all alternatives.
        hinted_id = construction_id

    raws: list[_RawCandidate] = []
    weak_statistical_hits = 0

    for row in model.constructions_by_pattern.get(pattern, []):
        cand = _from_construction(
            row,
            hint_matched=(hinted_id is not None and row.construction_id == hinted_id),
        )
        if observed_row_sufficient(
            dual_aligner=cand.dual_aligner,
            observation_count=cand.observation_count,
        ):
            raws.append(cand)
        else:
            weak_statistical_hits += 1

    if len(surfaces) == 1:
        surface = surfaces[0]
        for row in model.observed_by_coptic.get(surface, []):
            heur = model.heuristic_by_pair.get(
                (row.coptic_wordform, row.greek_wordform), 0.0
            )
            cand = _from_observed(row, heuristic=heur)
            if observed_row_sufficient(
                dual_aligner=cand.dual_aligner,
                observation_count=cand.observation_count,
            ):
                raws.append(cand)
            else:
                weak_statistical_hits += 1

        loan_hits = model.loanwords_by_form.get(surface, [])
        if loan_hits:
            if model.includes_verified_loanwords:
                for entry in loan_hits:
                    raws.append(_from_loanword(entry, matched_form=surface))
            elif not raws:
                return (
                    [],
                    AbstentionReason.ABLATION_EXCLUDES_LOANWORDS,
                    (
                        f"form {surface!r} only matches verified loanwords; "
                        "ablation arm A omits independently_verified constraints"
                    ),
                )

    if not raws:
        if weak_statistical_hits:
            return (
                [],
                AbstentionReason.INSUFFICIENT_EVIDENCE,
                (
                    "only uncorroborated one-aligner / weak statistical links; "
                    "refusing to assert unsupported Greek"
                ),
            )
        return (
            [],
            AbstentionReason.NO_ATTESTED_EVIDENCE,
            "no statistical construction, observed wordform, or allowed loanword",
        )
    return raws, None, ""


def _check_input_ids(
    span: DecodeSpanInput,
    *,
    request_unit_id: str,
) -> tuple[AbstentionReason | None, str]:
    """Enforce consistent request/span/source/token IDs; reject fragment bypass."""
    if _fragment_forbidden(request_unit_id):
        return (
            AbstentionReason.FORBIDDEN_QUERY_UNIT,
            "request.unit_id embeds a P.Oxy. Greek fragment handle",
        )
    if span.unit_id and span.unit_id != request_unit_id:
        if _fragment_forbidden(span.unit_id) or _fragment_forbidden(request_unit_id):
            return (
                AbstentionReason.FORBIDDEN_QUERY_UNIT,
                (
                    "span.unit_id / request.unit_id fragment bypass rejected: "
                    f"span={span.unit_id!r} request={request_unit_id!r}"
                ),
            )
        return (
            AbstentionReason.INPUT_ID_INCONSISTENT,
            (f"span.unit_id {span.unit_id!r} != request.unit_id {request_unit_id!r}"),
        )
    unit_id = span.unit_id or request_unit_id
    if _fragment_forbidden(unit_id):
        return (
            AbstentionReason.FORBIDDEN_QUERY_UNIT,
            "decode unit embeds a P.Oxy. Greek fragment handle",
        )

    try:
        parsed_span = parse_span_id(span.span_id)
    except ValueError as exc:
        return (
            AbstentionReason.INPUT_ID_INCONSISTENT,
            f"invalid span_id {span.span_id!r}: {exc}",
        )
    if parsed_span.unit_id != unit_id:
        return (
            AbstentionReason.INPUT_ID_INCONSISTENT,
            (f"span_id unit {parsed_span.unit_id!r} != effective unit {unit_id!r}"),
        )
    if span.source_id and parsed_span.source_id != span.source_id:
        return (
            AbstentionReason.INPUT_ID_INCONSISTENT,
            (
                f"span_id source {parsed_span.source_id!r} != span.source_id "
                f"{span.source_id!r}"
            ),
        )
    if _fragment_forbidden(parsed_span.unit_id) or _unit_forbidden(
        parsed_span.source_id
    ):
        # source_id "poxy" check via unit; also reject poxy in source
        pass
    if "poxy" in parsed_span.source_id.lower() or "poxy" in span.source_id.lower():
        return (
            AbstentionReason.FORBIDDEN_QUERY_UNIT,
            "span source_id embeds a P.Oxy. Greek fragment handle",
        )

    for tid in span.token_ids:
        try:
            parsed_tok = parse_token_id(tid)
        except ValueError as exc:
            return (
                AbstentionReason.INPUT_ID_INCONSISTENT,
                f"invalid token_id {tid!r}: {exc}",
            )
        if parsed_tok.unit_id != unit_id:
            return (
                AbstentionReason.INPUT_ID_INCONSISTENT,
                (f"token_id unit {parsed_tok.unit_id!r} != effective unit {unit_id!r}"),
            )
        if span.source_id and parsed_tok.source_id != span.source_id:
            return (
                AbstentionReason.INPUT_ID_INCONSISTENT,
                (
                    f"token_id source {parsed_tok.source_id!r} != span.source_id "
                    f"{span.source_id!r}"
                ),
            )
        if parsed_tok.source_id != parsed_span.source_id:
            return (
                AbstentionReason.INPUT_ID_INCONSISTENT,
                (
                    f"token_id source {parsed_tok.source_id!r} != span_id source "
                    f"{parsed_span.source_id!r}"
                ),
            )
        if _fragment_forbidden(parsed_tok.unit_id) or "poxy" in (
            parsed_tok.source_id.lower()
        ):
            return (
                AbstentionReason.FORBIDDEN_QUERY_UNIT,
                "token_id embeds a P.Oxy. Greek fragment handle",
            )
    return None, ""


def decode_span(
    model: DecodeModel,
    span: DecodeSpanInput,
    *,
    max_candidates: int,
    request_unit_id: str,
) -> SpanDecodeResult:
    """Decode one span; abstain rather than invent Greek."""
    id_reason, id_detail = _check_input_ids(span, request_unit_id=request_unit_id)
    unit_id = request_unit_id if not span.unit_id else span.unit_id
    if id_reason is not None:
        return SpanDecodeResult(
            span_id=span.span_id,
            unit_id=unit_id,
            surfaces=list(span.surfaces),
            construction_id=span.construction_id,
            status=SpanDecodeStatus.ABSTAIN,
            abstention_reason=id_reason,
            abstention_detail=id_detail,
            input_span=None,
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
    source_id = span.source_id
    if not source_id:
        try:
            source_id = parse_span_id(span.span_id).source_id
        except ValueError:
            source_id = "decode-input"
    return SpanRef(
        span_id=span.span_id,
        source_id=source_id,
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
            "Verified loanwords are lemma-level L-clues below attested surfaces.",
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
