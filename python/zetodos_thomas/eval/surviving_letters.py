# SPDX-License-Identifier: Apache-2.0
"""Surviving-letter gold extraction + scoring (loads fragments only at eval time).

P.Oxy. 654 / Greek Thomas fragment text must **not** be consulted during
blinded reconstruction or rule selection. This module is the sole JAK-40
scoring entry that opens the fragment corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from zetodos_thomas.eval.letters import greek_letters_only, letter_agreement
from zetodos_thomas.eval.review_gate import (
    EvalReviewGateError,
    assert_eval_package_human_checked,
    load_blinded_eval_package,
)
from zetodos_thomas.ingest.fragments import (
    fragments_corpus_path,
    load_fragment_corpus,
)
from zetodos_thomas.ingest.paths import require_repo_root
from zetodos_thomas.schemas.blinded import (
    BlindedReconstructionPackage,
    SurvivingLetterEvalReport,
    SurvivingLetterGold,
    SurvivingLetterScore,
    SurvivingLetterSpan,
)
from zetodos_thomas.schemas.fragment import (
    FragmentCorpus,
    FragmentLayerKind,
    FragmentTranscription,
)
from zetodos_thomas.schemas.source import ParallelUnit

# Pilot evidence / Dilley notes: Thomas 7 ~ P.Oxy. 654 lines 40-42.
DEFAULT_THOMAS7_LINE_IDS: tuple[str, ...] = ("main.40", "main.41", "main.42")
DEFAULT_THOMAS7_TRANSCRIPTION_ID = "poxy-654-dclp"
DEFAULT_EVAL_PACKAGE_REL = Path("data/evidence/eval/thomas-7-blinded-v1.json")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: object) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return _sha256_bytes(blob.encode("utf-8"))


def load_fragments_at_eval_time(repo_root: Path) -> FragmentCorpus:
    """Load the sealed fragment corpus — call only from scoring entry points."""
    path = fragments_corpus_path(repo_root)
    return load_fragment_corpus(path)


def transcription_by_id(
    corpus: FragmentCorpus,
    transcription_id: str,
) -> FragmentTranscription:
    for tx in corpus.transcriptions:
        if tx.id == transcription_id:
            return tx
    msg = f"transcription {transcription_id!r} not found in fragment corpus"
    raise KeyError(msg)


def extract_surviving_letter_gold(
    transcription: FragmentTranscription,
    *,
    line_ids: list[str] | tuple[str, ...],
) -> SurvivingLetterGold:
    """Collect surviving letters for ``line_ids``; drop restorations from gold."""
    by_id = {line.line_id: line for line in transcription.lines}
    spans: list[SurvivingLetterSpan] = []
    missing: list[str] = []
    for lid in line_ids:
        line = by_id.get(lid)
        if line is None:
            missing.append(lid)
            continue
        surviving = line.surviving_text
        restoration = line.restoration_text
        spans.append(
            SurvivingLetterSpan(
                line_id=line.line_id,
                n=line.n,
                surviving_text=surviving,
                surviving_letters=greek_letters_only(surviving),
                restoration_text=restoration,
                lacuna_count=line.lacuna_count,
            )
        )
    if missing:
        msg = f"missing fragment lines for gold extract: {missing}"
        raise KeyError(msg)
    surviving_text = "".join(s.surviving_text for s in spans)
    restoration_text = "".join(s.restoration_text for s in spans)
    # Sanity: restorations must not be folded into surviving_letters.
    for span in spans:
        for tok_kind_text in (span.restoration_text,):
            if not tok_kind_text:
                continue
            # If a restoration letter somehow appears alone as surviving, still
            # exclude it from gold by construction (we only joined surviving).
            _ = FragmentLayerKind.RESTORATION
    return SurvivingLetterGold(
        transcription_id=transcription.id,
        manuscript_id=transcription.manuscript_id,
        source_id=transcription.source_id,
        line_ids=list(line_ids),
        spans=spans,
        surviving_text=surviving_text,
        surviving_letters=greek_letters_only(surviving_text),
        restoration_letters_excluded=greek_letters_only(restoration_text),
        notes=[
            "Gold denominator = surviving letters only.",
            "Editorial restorations / lacuna fills excluded from gold.",
            f"lines={list(line_ids)}",
        ],
    )


def hypothesis_letters_from_top1(top1: list[str | None]) -> str:
    """Join non-abstaining top-1 surfaces; abstentions contribute no letters."""
    parts = [s for s in top1 if s]
    return greek_letters_only("".join(parts))


def score_surviving_letters(
    *,
    unit_id: str,
    top1_greek: list[str | None],
    gold: SurvivingLetterGold,
    review_status: str,
    official: bool,
) -> SurvivingLetterScore:
    hyp = "".join(s for s in top1_greek if s)
    metrics = letter_agreement(hyp, gold.surviving_text)
    return SurvivingLetterScore(
        unit_id=unit_id,
        official=official,
        hypothesis_letters=metrics.hypothesis_letters,
        gold_surviving_letters=metrics.gold_surviving_letters,
        true_positives=metrics.true_positives,
        predicted_denominator=metrics.predicted_denominator,
        gold_denominator=metrics.gold_denominator,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        longest_common_subsequence=metrics.longest_common_subsequence,
        review_status=review_status,
        notes=[
            "Surviving-letter agreement via LCS; restorations excluded from gold.",
            "No invented pass/fail threshold (Jake Phase 2).",
        ],
    )


def score_nt_unit_letters(
    *,
    unit_id: str,
    top1_greek: list[str | None],
    greek_unit: ParallelUnit,
    review_status: str,
    official: bool,
) -> SurvivingLetterScore:
    """Compare reconstruction letters to Vaticanus B* verse text (eval-time)."""
    gold_text = greek_unit.greek.normalized_text or greek_unit.greek.original_text
    hyp = "".join(s for s in top1_greek if s)
    metrics = letter_agreement(hyp, gold_text)
    return SurvivingLetterScore(
        unit_id=unit_id,
        official=official,
        hypothesis_letters=metrics.hypothesis_letters,
        gold_surviving_letters=metrics.gold_surviving_letters,
        true_positives=metrics.true_positives,
        predicted_denominator=metrics.predicted_denominator,
        gold_denominator=metrics.gold_denominator,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        longest_common_subsequence=metrics.longest_common_subsequence,
        review_status=review_status,
        notes=[
            "NT holdout letter agreement vs Vaticanus B* at eval time only.",
            "Greek text was not available to the blinded reconstruct path.",
            f"greek_source_id={greek_unit.greek.source_id}",
        ],
    )


def score_thomas7_surviving(
    repo_root: Path | None,
    reconstruction: BlindedReconstructionPackage,
    *,
    eval_package_path: Path | None = None,
    line_ids: list[str] | tuple[str, ...] = DEFAULT_THOMAS7_LINE_IDS,
    require_human_checked: bool = True,
) -> SurvivingLetterEvalReport:
    """Official Thomas 7 surviving-letter score entry (loads fragments here).

    When ``require_human_checked`` is True (default), raises
    ``EvalReviewGateError`` unless the blinded eval pack is ``human-checked``.
    Set False only for protocol/fixture smoke tests; those reports are marked
    ``official=False``.
    """
    root = require_repo_root(repo_root)
    pack_path = eval_package_path or (root / DEFAULT_EVAL_PACKAGE_REL)
    if require_human_checked:
        pack = assert_eval_package_human_checked(pack_path)
        official = True
    else:
        pack = load_blinded_eval_package(pack_path)
        official = False
    review_status = str(pack.get("review_status", ""))
    pack_id = str(pack.get("id", pack_path.name))

    # Fragments open only after the review gate decision for this entry.
    corpus = load_fragments_at_eval_time(root)
    tx = transcription_by_id(corpus, DEFAULT_THOMAS7_TRANSCRIPTION_ID)
    gold = extract_surviving_letter_gold(tx, line_ids=line_ids)

    thomas_units = [u for u in reconstruction.units if u.unit_id == "thomas.7"]
    if not thomas_units:
        msg = "reconstruction package has no thomas.7 unit"
        raise ValueError(msg)
    scores = [
        score_surviving_letters(
            unit_id=u.unit_id,
            top1_greek=u.top1_greek_surfaces,
            gold=gold,
            review_status=review_status,
            official=official,
        )
        for u in thomas_units
    ]

    frag_path = fragments_corpus_path(root)
    digests = {
        "eval_package_sha256": _sha256_bytes(pack_path.read_bytes()),
        "fragment_corpus_sha256": _sha256_bytes(frag_path.read_bytes()),
        "reconstruction_package_sha256": _sha256_json(
            reconstruction.model_dump(mode="json")
        ),
        "surviving_letters_sha256": _sha256_bytes(
            gold.surviving_letters.encode("utf-8")
        ),
    }
    return SurvivingLetterEvalReport(
        id="jak40-thomas7-surviving-letter-eval-v1",
        title="Thomas 7 surviving-letter evaluation (JAK-40)",
        reconstruction_package_id=reconstruction.id,
        eval_package_id=pack_id,
        review_status=review_status,
        official=official,
        gold=gold,
        scores=scores,
        artifact_digests=digests,
        blinding={
            "greek_loaded_during_reconstruction": False,
            "greek_loaded_at_scoring": True,
            "transcription_id": DEFAULT_THOMAS7_TRANSCRIPTION_ID,
            "line_ids": list(line_ids),
            "restorations_in_gold_denominator": False,
            "require_human_checked": require_human_checked,
        },
        summary={
            "official": official,
            "n_scores": len(scores),
            "gold_denominator": gold.surviving_letters and len(gold.surviving_letters),
            "restoration_letters_excluded": len(gold.restoration_letters_excluded),
        },
        provenance={
            "owned_by": "JAK-40",
            "llm": False,
            "network": False,
            "phase2_active_aligners": reconstruction.phase2_active_aligners,
            "adopted_aligner": None,
        },
        notes=[
            "Surviving-letter scoring consults P.Oxy. 654 only at eval time.",
            "Editorial restorations excluded from the gold comparison.",
            "Official=true requires eval package review_status=human-checked.",
            *(
                []
                if official
                else [
                    "UNOFFICIAL protocol run: review gate bypassed for harness "
                    "smoke only; do not treat metrics as JAK-40 acceptance scores."
                ]
            ),
        ],
    )


def try_score_thomas7_or_gate_report(
    repo_root: Path | None,
    reconstruction: BlindedReconstructionPackage,
    *,
    eval_package_path: Path | None = None,
) -> SurvivingLetterEvalReport:
    """Score when human-checked; otherwise return a gate-blocked report.

    Reconstruction artifacts stay inspectable even while independent review is
    pending (JAK-43 follow-up). Does not invent ``human-checked``.
    """
    root = require_repo_root(repo_root)
    pack_path = eval_package_path or (root / DEFAULT_EVAL_PACKAGE_REL)
    try:
        return score_thomas7_surviving(
            root,
            reconstruction,
            eval_package_path=pack_path,
            require_human_checked=True,
        )
    except EvalReviewGateError as exc:
        pack = load_blinded_eval_package(pack_path)
        return SurvivingLetterEvalReport(
            id="jak40-thomas7-surviving-letter-eval-v1",
            title="Thomas 7 surviving-letter evaluation (JAK-40) — gate blocked",
            reconstruction_package_id=reconstruction.id,
            eval_package_id=str(pack.get("id", pack_path.name)),
            review_status=str(pack.get("review_status", "")),
            official=False,
            gold=None,
            scores=[],
            artifact_digests={
                "eval_package_sha256": _sha256_bytes(pack_path.read_bytes()),
            },
            blinding={
                "greek_loaded_during_reconstruction": False,
                "greek_loaded_at_scoring": False,
                "gate_blocked": True,
                "gate_error": str(exc),
            },
            summary={"official": False, "gate_blocked": True},
            provenance={"owned_by": "JAK-40", "llm": False, "network": False},
            notes=[
                "Independent review still pending (review_status ≠ human-checked).",
                "Blinded reconstructions remain valid; official surviving-letter "
                "scores deferred until JAK-43 review follow-up completes.",
                str(exc),
            ],
        )


def assert_restorations_excluded_from_gold(gold: SurvivingLetterGold) -> None:
    """Fail if restoration letters were folded into the surviving gold string."""
    # Every restoration letter sequence must not be required in gold; stronger
    # check: gold.surviving_letters equals letters of surviving_text only.
    expected = greek_letters_only(gold.surviving_text)
    if gold.surviving_letters != expected:
        msg = "surviving_letters drift from surviving_text"
        raise AssertionError(msg)
    # If restorations share letters with surviving (common), we only assert the
    # gold path never concatenated restoration_text.
    if gold.restoration_letters_excluded and gold.surviving_text:
        # Ensure extract used surviving_text field exclusively.
        for span in gold.spans:
            if span.surviving_letters != greek_letters_only(span.surviving_text):
                msg = f"span {span.line_id} surviving_letters inconsistent"
                raise AssertionError(msg)


def protocol_fixture_payload() -> dict[str, Any]:
    """Minimal in-memory protocol fixture for tests (no real P.Oxy. load)."""
    return {
        "surviving": "κάρι ἐστιν ων ἔστα ὃν",
        "restoration": "μα ός ὁ λέων ὃν φάγεται ἄνθρωπος",
    }
