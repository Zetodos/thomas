# SPDX-License-Identifier: Apache-2.0
"""Structural / textual-fidelity validation for the Horner x Vaticanus pilot."""

from __future__ import annotations

from dataclasses import dataclass, field

from zetodos_thomas.align.paratext import has_editorial_marks, unit_has_mark_heading
from zetodos_thomas.align.splits import PilotSplits, SplitRole, partition_corpus
from zetodos_thomas.ingest.normalize import recompute_normalized
from zetodos_thomas.ingest.ntvmr_vaticanus import SOURCE_ID as VATICANUS_SOURCE_ID
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus


@dataclass
class ValidationIssue:
    """One validation finding."""

    code: str
    message: str
    unit_id: str | None = None
    severity: str = "error"  # error | warning


@dataclass
class PilotValidationReport:
    """Aggregated validation result."""

    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    unit_count: int = 0
    nonempty_positive: int = 0
    omission_units: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]


def _check_unit_structure(unit: ParallelUnit, issues: list[ValidationIssue]) -> None:
    if unit.unit_id != unit.greek.unit_id or unit.unit_id != unit.sahidic.unit_id:
        issues.append(
            ValidationIssue(
                code="unit_id_mismatch",
                message=(
                    f"parallel unit_id {unit.unit_id!r} disagrees with side ids "
                    f"greek={unit.greek.unit_id!r} sahidic={unit.sahidic.unit_id!r}"
                ),
                unit_id=unit.unit_id,
            )
        )
    expected = f"{unit.greek.book}.{unit.greek.chapter}.{unit.greek.verse}"
    if unit.greek.unit_id != expected:
        issues.append(
            ValidationIssue(
                code="greek_ref_mismatch",
                message=f"greek unit_id {unit.greek.unit_id!r} != {expected!r}",
                unit_id=unit.unit_id,
            )
        )
    sah_expected = f"{unit.sahidic.book}.{unit.sahidic.chapter}.{unit.sahidic.verse}"
    if unit.sahidic.unit_id != sah_expected:
        issues.append(
            ValidationIssue(
                code="sahidic_ref_mismatch",
                message=(
                    f"sahidic unit_id {unit.sahidic.unit_id!r} != {sah_expected!r}"
                ),
                unit_id=unit.unit_id,
            )
        )
    if (
        unit.greek.book != unit.sahidic.book
        or unit.greek.chapter != unit.sahidic.chapter
        or unit.greek.verse != unit.sahidic.verse
    ):
        issues.append(
            ValidationIssue(
                code="cross_language_ref_mismatch",
                message="greek and sahidic book/chapter/verse disagree",
                unit_id=unit.unit_id,
            )
        )


def _check_normalization(unit: ParallelUnit, issues: list[ValidationIssue]) -> None:
    for side, language, text_unit in (
        ("greek", "greek", unit.greek),
        ("sahidic", "coptic", unit.sahidic),
    ):
        expected = recompute_normalized(
            language=language,
            original_text=text_unit.original_text,
        )
        if text_unit.normalized_text != expected:
            issues.append(
                ValidationIssue(
                    code="normalization_mismatch",
                    message=(
                        f"{side} normalized_text does not recompute from "
                        f"original_text for {text_unit.unit_id}"
                    ),
                    unit_id=unit.unit_id,
                )
            )


def _check_source_ids(corpus: PilotCorpus, issues: list[ValidationIssue]) -> None:
    if corpus.greek_source_id != VATICANUS_SOURCE_ID:
        issues.append(
            ValidationIssue(
                code="greek_source",
                message=(
                    f"greek_source_id must be {VATICANUS_SOURCE_ID} (JAK-25); "
                    f"got {corpus.greek_source_id!r}"
                ),
            )
        )
    if corpus.sahidic_source_id != "copsahhorner":
        issues.append(
            ValidationIssue(
                code="sahidic_source",
                message=(
                    "sahidic_source_id must be copsahhorner (JAK-14); "
                    f"got {corpus.sahidic_source_id!r}"
                ),
            )
        )
    for unit in corpus.units:
        if unit.greek.source_id != corpus.greek_source_id:
            issues.append(
                ValidationIssue(
                    code="greek_unit_source",
                    message=(
                        f"greek.source_id {unit.greek.source_id!r} != corpus "
                        f"{corpus.greek_source_id!r}"
                    ),
                    unit_id=unit.unit_id,
                )
            )
        if unit.sahidic.source_id != corpus.sahidic_source_id:
            issues.append(
                ValidationIssue(
                    code="sahidic_unit_source",
                    message=(
                        f"sahidic.source_id {unit.sahidic.source_id!r} != corpus "
                        f"{corpus.sahidic_source_id!r}"
                    ),
                    unit_id=unit.unit_id,
                )
            )


def _check_ordering_uniqueness(
    corpus: PilotCorpus,
    issues: list[ValidationIssue],
) -> None:
    seen: set[str] = set()
    prev_key: tuple[str, int, int] | None = None
    for unit in corpus.units:
        if unit.unit_id in seen:
            issues.append(
                ValidationIssue(
                    code="duplicate_unit_id",
                    message=f"duplicate unit_id {unit.unit_id}",
                    unit_id=unit.unit_id,
                )
            )
        seen.add(unit.unit_id)
        key = (unit.greek.book, unit.greek.chapter, unit.greek.verse)
        if prev_key is not None and key < prev_key:
            # Soft warning: ranges may be intentionally non-canonical order
            # (holdout after train); only flag strict decrease within same book.
            if key[0] == prev_key[0] and key < prev_key:
                issues.append(
                    ValidationIssue(
                        code="ordering",
                        message=f"unit order decreases within book at {unit.unit_id}",
                        unit_id=unit.unit_id,
                        severity="warning",
                    )
                )
        prev_key = key


def _check_omission_and_paratext(
    corpus: PilotCorpus,
    splits: PilotSplits | None,
    issues: list[ValidationIssue],
) -> tuple[int, int]:
    nonempty = 0
    omission = 0
    for unit in corpus.units:
        g_empty = not unit.greek.original_text.strip()
        s_empty = not unit.sahidic.original_text.strip()
        role = splits.role_for(unit.unit_id) if splits is not None else None
        if role is SplitRole.OMISSION:
            omission += 1
            if not (g_empty and s_empty):
                issues.append(
                    ValidationIssue(
                        code="omission_not_empty",
                        message="omission-split unit must be empty on both sides",
                        unit_id=unit.unit_id,
                    )
                )
            if not unit.notes.strip():
                issues.append(
                    ValidationIssue(
                        code="omission_missing_notes",
                        message="omission fixture requires explanatory notes",
                        unit_id=unit.unit_id,
                    )
                )
        elif not g_empty and not s_empty:
            nonempty += 1
        elif g_empty or s_empty:
            issues.append(
                ValidationIssue(
                    code="unexpected_empty",
                    message=(
                        "non-omission unit has empty side(s); classify as "
                        "omission fixture or fix extraction"
                    ),
                    unit_id=unit.unit_id,
                )
            )

        if unit_has_mark_heading(unit.sahidic) and unit.unit_id == "mark.1.1":
            # Informational: heading must be treated as paratext, not Greek word.
            notes_l = unit.notes.lower()
            if "heading" not in notes_l and "paratext" not in notes_l:
                issues.append(
                    ValidationIssue(
                        code="mark_heading_paratext",
                        message=(
                            "Mark 1:1 Sahidic begins with book heading "
                            "ⲙⲁⲣⲕⲟⲥ; do not align as a Greek verse token "
                            "(paratext)"
                        ),
                        unit_id=unit.unit_id,
                        severity="warning",
                    )
                )

        if has_editorial_marks(unit.greek.original_text):
            # Editorial marks must remain recoverable on original; normalized clean.
            if has_editorial_marks(unit.greek.normalized_text):
                issues.append(
                    ValidationIssue(
                        code="editorial_in_normalized_greek",
                        message="Greek normalized_text still contains editorial marks",
                        unit_id=unit.unit_id,
                    )
                )
        if has_editorial_marks(unit.sahidic.original_text):
            if has_editorial_marks(unit.sahidic.normalized_text):
                issues.append(
                    ValidationIssue(
                        code="editorial_in_normalized_sahidic",
                        message=(
                            "Sahidic normalized_text still contains editorial marks"
                        ),
                        unit_id=unit.unit_id,
                    )
                )
    return nonempty, omission


def validate_pilot_corpus(
    corpus: PilotCorpus,
    *,
    splits: PilotSplits | None = None,
) -> PilotValidationReport:
    """Run structural + normalization + omission/paratext checks offline."""
    issues: list[ValidationIssue] = []
    _check_source_ids(corpus, issues)
    _check_ordering_uniqueness(corpus, issues)
    for unit in corpus.units:
        _check_unit_structure(unit, issues)
        _check_normalization(unit, issues)
    nonempty, omission = _check_omission_and_paratext(corpus, splits, issues)

    if splits is not None:
        try:
            partition_corpus(corpus, splits)
        except ValueError as exc:
            issues.append(ValidationIssue(code="split_coverage", message=str(exc)))
        if splits.greek_source_id != corpus.greek_source_id:
            issues.append(
                ValidationIssue(
                    code="splits_greek_source",
                    message="splits.greek_source_id disagrees with pilot corpus",
                )
            )
        if splits.sahidic_source_id != corpus.sahidic_source_id:
            issues.append(
                ValidationIssue(
                    code="splits_sahidic_source",
                    message="splits.sahidic_source_id disagrees with pilot corpus",
                )
            )

    errors = [i for i in issues if i.severity == "error"]
    return PilotValidationReport(
        ok=not errors,
        issues=issues,
        unit_count=len(corpus.units),
        nonempty_positive=nonempty,
        omission_units=omission,
    )
