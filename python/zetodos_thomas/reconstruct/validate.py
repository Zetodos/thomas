# SPDX-License-Identifier: Apache-2.0
"""Validate JAK-39 decode results (schema + leakage / layering guards)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from zetodos_thomas.lexicon.candidates import FORBIDDEN_UNIT_MARKERS
from zetodos_thomas.reconstruct.ranking import RANKING_RULES_ID
from zetodos_thomas.schemas.decode import (
    DecodeResult,
    DecoderEvidenceKind,
    SpanDecodeStatus,
)
from zetodos_thomas.schemas.ids import parse_span_id
from zetodos_thomas.schemas.mappings import EvidenceClass


@dataclass
class DecodeValidationIssue:
    """One structural / leakage / ranking problem."""

    severity: str
    code: str
    message: str


@dataclass
class DecodeValidationReport:
    """Offline validation result for a decode document."""

    ok: bool
    artifact_id: str
    issues: list[DecodeValidationIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "artifact_id": self.artifact_id,
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message}
                for i in self.issues
            ],
        }


def _evidence_forbidden_ref(ref_id: str) -> bool:
    lowered = ref_id.lower().replace(" ", "")
    return any(marker in lowered for marker in FORBIDDEN_UNIT_MARKERS) or (
        "p.oxy" in lowered
    )


def validate_decode_result(result: DecodeResult) -> DecodeValidationReport:
    """Check ranking stability metadata, abstention rules, and leakage."""
    issues: list[DecodeValidationIssue] = []
    if result.owned_by != "JAK-39":
        issues.append(
            DecodeValidationIssue(
                "error",
                "owned_by",
                f"owned_by must be JAK-39; got {result.owned_by!r}",
            )
        )
    if result.ranking_rules_id != RANKING_RULES_ID:
        issues.append(
            DecodeValidationIssue(
                "error",
                "ranking_rules_id",
                f"expected {RANKING_RULES_ID!r}; got {result.ranking_rules_id!r}",
            )
        )
    if list(result.phase2_active_aligners) != ["eflomal", "fast_align"]:
        issues.append(
            DecodeValidationIssue(
                "error",
                "aligners",
                "phase2_active_aligners must be [eflomal, fast_align]",
            )
        )
    if result.adopted_aligner is not None:
        issues.append(
            DecodeValidationIssue(
                "error",
                "adopted_aligner",
                "adopted_aligner must be null (keep-both)",
            )
        )
    if result.provenance.get("llm") is True:
        issues.append(
            DecodeValidationIssue(
                "error",
                "llm_forbidden",
                "decode provenance must not claim llm=true",
            )
        )
    # Digests are mandatory for fail-closed loads from committed JAK-37/38
    # artifacts; fixture models used in unit tests may omit them.
    if result.provenance.get("validation") == "jak37_jak38_fail_closed":
        digests = result.provenance.get("artifact_digests")
        if not isinstance(digests, dict) or not digests.get("lexical_sha256"):
            issues.append(
                DecodeValidationIssue(
                    "error",
                    "missing_artifact_digests",
                    "decode provenance must record immutable artifact digests",
                )
            )
        if not result.provenance.get("evidence_provenance"):
            issues.append(
                DecodeValidationIssue(
                    "error",
                    "missing_evidence_provenance",
                    "decode provenance must record shared evidence provenance",
                )
            )

    for span in result.spans:
        try:
            parse_span_id(span.span_id)
        except ValueError as exc:
            issues.append(
                DecodeValidationIssue(
                    "error",
                    "span_id",
                    f"{span.span_id!r}: {exc}",
                )
            )
        if span.status is SpanDecodeStatus.ABSTAIN:
            if span.candidates:
                issues.append(
                    DecodeValidationIssue(
                        "error",
                        "abstain_with_candidates",
                        f"{span.span_id}: abstain must not list greek candidates",
                    )
                )
            if span.abstention_reason is None:
                issues.append(
                    DecodeValidationIssue(
                        "error",
                        "missing_abstention_reason",
                        f"{span.span_id}: abstain requires abstention_reason",
                    )
                )
            continue

        if not span.candidates:
            issues.append(
                DecodeValidationIssue(
                    "error",
                    "empty_candidates",
                    f"{span.span_id}: candidates status with empty list",
                )
            )
        ranks = [c.rank for c in span.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            issues.append(
                DecodeValidationIssue(
                    "error",
                    "rank_sequence",
                    f"{span.span_id}: ranks must be 1..n contiguous; got {ranks}",
                )
            )
        # Ranking keys must be non-decreasing in sort order (already ranked).
        keys = [tuple(c.ranking_key) for c in span.candidates]
        if keys != sorted(keys):
            issues.append(
                DecodeValidationIssue(
                    "error",
                    "ranking_order",
                    f"{span.span_id}: candidates not sorted by ranking_key",
                )
            )
        greek_seqs = [tuple(c.greek_surfaces) for c in span.candidates]
        if len(greek_seqs) != len(set(greek_seqs)):
            issues.append(
                DecodeValidationIssue(
                    "error",
                    "duplicate_greek_hypothesis",
                    f"{span.span_id}: duplicate greek_surfaces among candidates",
                )
            )
        for cand in span.candidates:
            if not cand.greek_surfaces or any(not s for s in cand.greek_surfaces):
                issues.append(
                    DecodeValidationIssue(
                        "error",
                        "empty_greek",
                        f"{span.span_id} rank {cand.rank}: empty greek surface",
                    )
                )
            for ev in cand.evidence:
                if _evidence_forbidden_ref(ev.ref_id):
                    issues.append(
                        DecodeValidationIssue(
                            "error",
                            "evidence_leakage",
                            (
                                f"{span.span_id}: evidence ref_id embeds forbidden "
                                f"Thomas/P.Oxy. marker: {ev.ref_id!r}"
                            ),
                        )
                    )
                if (
                    ev.kind is DecoderEvidenceKind.STATISTICAL_CONSTRUCTION
                    and ev.evidence_class is not EvidenceClass.STATISTICAL
                ):
                    issues.append(
                        DecodeValidationIssue(
                            "error",
                            "evidence_class_collapse",
                            (
                                f"{span.span_id}: statistical construction must keep "
                                "evidence_class=statistical"
                            ),
                        )
                    )
                if (
                    ev.kind is DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD
                    and ev.evidence_class is not EvidenceClass.INDEPENDENTLY_VERIFIED
                ):
                    issues.append(
                        DecodeValidationIssue(
                            "error",
                            "evidence_class_collapse",
                            (
                                f"{span.span_id}: loanword must keep "
                                "evidence_class=independently_verified"
                            ),
                        )
                    )
                if (
                    ev.kind is DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM
                    and ev.layer != "observed_wordform"
                ):
                    issues.append(
                        DecodeValidationIssue(
                            "error",
                            "lexical_layer",
                            (
                                f"{span.span_id}: lexical candidate must declare "
                                "layer=observed_wordform"
                            ),
                        )
                    )

    ok = not any(i.severity == "error" for i in issues)
    return DecodeValidationReport(ok=ok, artifact_id=result.id, issues=issues)


def validate_decode_result_path(path: Path) -> DecodeValidationReport:
    """Load JSON and validate."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    result = DecodeResult.model_validate(payload)
    return validate_decode_result(result)
