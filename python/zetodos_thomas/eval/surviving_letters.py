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

from zetodos_thomas.eval.letters import (
    gap_aware_letter_score,
    greek_letters_only,
    token_aware_letter_score,
)
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
    BlindedFreezeRecord,
    BlindedReconstructionPackage,
    SurvivingLetterEvalReport,
    SurvivingLetterGold,
    SurvivingLetterScore,
    SurvivingLetterSpan,
    SurvivingLetterToken,
    UnclearLetterPolicy,
)
from zetodos_thomas.schemas.fragment import (
    FragmentCorpus,
    FragmentLayerKind,
    FragmentTranscription,
)
from zetodos_thomas.schemas.source import ParallelUnit

DEFAULT_THOMAS7_LINE_IDS: tuple[str, ...] = ("main.40", "main.41", "main.42")
DEFAULT_THOMAS7_TRANSCRIPTION_ID = "poxy-654-dclp"
DEFAULT_EVAL_PACKAGE_REL = Path("data/evidence/eval/thomas-7-blinded-v1.json")
DEFAULT_FREEZE_REL = Path("data/reconstructions/jak40/thomas-7-blinded-freeze-v1.json")
DEFAULT_UNCLEAR_POLICY = UnclearLetterPolicy.EXCLUDE


class BlindedScoreIntegrityError(ValueError):
    """Official scoring refused due to freeze / identity mismatch."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: object) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return _sha256_bytes(blob.encode("utf-8"))


def package_content_sha256(package: BlindedReconstructionPackage) -> str:
    """Canonical digest of a reconstruction package (pre-unseal pin)."""
    return _sha256_json(package.model_dump(mode="json"))


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
    unclear_policy: UnclearLetterPolicy = DEFAULT_UNCLEAR_POLICY,
) -> SurvivingLetterGold:
    """Collect surviving letters; preserve certainty; drop restorations."""
    by_id = {line.line_id: line for line in transcription.lines}
    spans: list[SurvivingLetterSpan] = []
    missing: list[str] = []
    for lid in line_ids:
        line = by_id.get(lid)
        if line is None:
            missing.append(lid)
            continue
        surviving_tokens: list[SurvivingLetterToken] = []
        clear_parts: list[str] = []
        unclear_parts: list[str] = []
        for tok in line.tokens:
            if tok.kind is not FragmentLayerKind.SURVIVING:
                continue
            letters = greek_letters_only(tok.text)
            surviving_tokens.append(
                SurvivingLetterToken(
                    text=tok.text,
                    certainty=tok.certainty,
                    letters=letters,
                )
            )
            if tok.certainty == "unclear":
                unclear_parts.append(tok.text)
            else:
                clear_parts.append(tok.text)
        surviving = "".join(t.text for t in surviving_tokens)
        clear_text = "".join(clear_parts)
        unclear_text = "".join(unclear_parts)
        restoration = line.restoration_text
        spans.append(
            SurvivingLetterSpan(
                line_id=line.line_id,
                n=line.n,
                surviving_text=surviving,
                surviving_letters=greek_letters_only(surviving),
                clear_letters=greek_letters_only(clear_text),
                unclear_letters=greek_letters_only(unclear_text),
                restoration_text=restoration,
                lacuna_count=line.lacuna_count,
                surviving_tokens=surviving_tokens,
            )
        )
    if missing:
        msg = f"missing fragment lines for gold extract: {missing}"
        raise KeyError(msg)

    surviving_text = "".join(s.surviving_text for s in spans)
    clear_letters = "".join(s.clear_letters for s in spans)
    unclear_letters = "".join(s.unclear_letters for s in spans)
    restoration_text = "".join(s.restoration_text for s in spans)
    if unclear_policy is UnclearLetterPolicy.INCLUDE:
        primary = clear_letters + unclear_letters
    else:
        # exclude + sensitivity_only: primary gold is clear letters only
        primary = clear_letters
    clear_runs = [s.clear_letters for s in spans]

    return SurvivingLetterGold(
        transcription_id=transcription.id,
        manuscript_id=transcription.manuscript_id,
        source_id=transcription.source_id,
        line_ids=list(line_ids),
        spans=spans,
        surviving_text=surviving_text,
        surviving_letters=greek_letters_only(surviving_text),
        clear_letters=clear_letters,
        unclear_letters=unclear_letters,
        restoration_letters_excluded=greek_letters_only(restoration_text),
        unclear_policy=unclear_policy,
        primary_gold_letters=primary,
        clear_runs_by_line=clear_runs,
        notes=[
            "Primary gold denominator follows unclear_policy (default: exclude).",
            "Editorial restorations / lacuna fills excluded from gold.",
            "Token-level certainty preserved on surviving_tokens.",
            f"lines={list(line_ids)} unclear_policy={unclear_policy.value}",
        ],
    )


def assert_restorations_excluded_from_gold(gold: SurvivingLetterGold) -> None:
    """Fail if restoration letters were folded into surviving gold fields."""
    expected = greek_letters_only(gold.surviving_text)
    if gold.surviving_letters != expected:
        msg = "surviving_letters drift from surviving_text"
        raise AssertionError(msg)
    for span in gold.spans:
        if span.surviving_letters != greek_letters_only(span.surviving_text):
            msg = f"span {span.line_id} surviving_letters inconsistent"
            raise AssertionError(msg)
        clear = "".join(
            t.text for t in span.surviving_tokens if t.certainty != "unclear"
        )
        unclear = "".join(
            t.text for t in span.surviving_tokens if t.certainty == "unclear"
        )
        if span.clear_letters != greek_letters_only(clear):
            msg = f"span {span.line_id} clear_letters inconsistent"
            raise AssertionError(msg)
        if span.unclear_letters != greek_letters_only(unclear):
            msg = f"span {span.line_id} unclear_letters inconsistent"
            raise AssertionError(msg)


def write_freeze_record(
    package: BlindedReconstructionPackage,
    path: Path,
    *,
    unit_id: str = "thomas.7",
) -> BlindedFreezeRecord:
    """Commit a pre-unseal freeze pin for official Thomas 7 scoring."""
    thomas = [u for u in package.units if u.unit_id == unit_id]
    if not thomas:
        msg = f"freeze requires unit {unit_id!r} in reconstruction package"
        raise BlindedScoreIntegrityError(msg)
    record = BlindedFreezeRecord(
        reconstruction_package_id=package.id,
        unit_id=unit_id,
        reconstruction_sha256=package_content_sha256(package),
        ablation_condition=package.ablation_condition,
        ranking_rules_id=package.ranking_rules_id,
        evidence_provenance=dict(package.evidence_provenance),
        artifact_digests=dict(package.artifact_digests),
        sealed_before_fragment_load=True,
        notes=[
            "Frozen before any P.Oxy. / fragment corpus load.",
            "Official scoring must verify this digest and Thomas-7 identity.",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def load_freeze_record(path: Path) -> BlindedFreezeRecord:
    return BlindedFreezeRecord.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def verify_official_freeze(
    package: BlindedReconstructionPackage,
    freeze: BlindedFreezeRecord,
    *,
    unit_id: str = "thomas.7",
) -> None:
    """Require package digest + Thomas-7 identity match the pre-unseal freeze."""
    if freeze.unit_id != unit_id:
        msg = f"freeze unit_id {freeze.unit_id!r} != required {unit_id!r}"
        raise BlindedScoreIntegrityError(msg)
    if package.id != freeze.reconstruction_package_id:
        msg = (
            f"package id {package.id!r} != freeze id "
            f"{freeze.reconstruction_package_id!r}"
        )
        raise BlindedScoreIntegrityError(msg)
    actual = package_content_sha256(package)
    if actual != freeze.reconstruction_sha256:
        msg = (
            "reconstruction digest does not match pre-unseal freeze "
            f"(got {actual[:12]}… expected {freeze.reconstruction_sha256[:12]}…); "
            "post-reveal tampering cannot receive official=true"
        )
        raise BlindedScoreIntegrityError(msg)
    if not any(u.unit_id == unit_id for u in package.units):
        msg = f"reconstruction missing required unit {unit_id!r}"
        raise BlindedScoreIntegrityError(msg)
    if package.ablation_condition is not freeze.ablation_condition:
        msg = "ablation_condition drift vs freeze"
        raise BlindedScoreIntegrityError(msg)
    if not freeze.sealed_before_fragment_load:
        msg = "freeze must declare sealed_before_fragment_load=true"
        raise BlindedScoreIntegrityError(msg)


def score_surviving_letters(
    *,
    unit_id: str,
    top1_greek: list[str | None],
    gold: SurvivingLetterGold,
    review_status: str,
    official: bool,
) -> SurvivingLetterScore:
    hyp = " ".join(s for s in top1_greek if s)
    primary = gap_aware_letter_score(hyp, gold.clear_runs_by_line)
    sensitivity = gap_aware_letter_score(
        hyp,
        ["".join(s.clear_letters + s.unclear_letters for s in gold.spans)],
    )
    return SurvivingLetterScore(
        unit_id=unit_id,
        official=official,
        hypothesis_letters=primary.hypothesis_letters,
        gold_surviving_letters=primary.gold_clear_letters,
        true_positives=primary.true_positives,
        predicted_denominator=primary.predicted_denominator,
        gold_denominator=primary.gold_denominator,
        precision=primary.precision,
        recall=primary.recall,
        f1=primary.f1,
        incidental_global_lcs=primary.incidental_global_lcs,
        metric_kind=primary.metric_kind,
        review_status=review_status,
        unclear_policy=gold.unclear_policy.value,
        sensitivity_include_unclear={
            "true_positives": sensitivity.true_positives,
            "gold_denominator": sensitivity.gold_denominator,
            "precision": sensitivity.precision,
            "recall": sensitivity.recall,
            "f1": sensitivity.f1,
            "note": "Sensitivity only — unclear letters included; not primary gold.",
        },
        notes=[
            *primary.notes,
            "Restorations excluded; unclear excluded from primary gold (default).",
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
    """Token-positional letter agreement vs Vaticanus B* (eval-time)."""
    gold_text = greek_unit.greek.normalized_text or greek_unit.greek.original_text
    gold_tokens = gold_text.split()
    hyp_tokens = [s for s in top1_greek if s]
    # Flatten multi-surface span hypotheses into whitespace tokens.
    hyp_flat: list[str] = []
    for item in hyp_tokens:
        hyp_flat.extend(item.split())
    metrics = token_aware_letter_score(hyp_flat, gold_tokens)
    return SurvivingLetterScore(
        unit_id=unit_id,
        official=official,
        hypothesis_letters=metrics.hypothesis_letters,
        gold_surviving_letters=metrics.gold_letters,
        true_positives=metrics.true_positives,
        predicted_denominator=metrics.predicted_denominator,
        gold_denominator=metrics.gold_denominator,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        incidental_global_lcs=metrics.incidental_global_lcs,
        metric_kind=metrics.metric_kind,
        review_status=review_status,
        notes=[
            *metrics.notes,
            "NT holdout letter agreement vs Vaticanus B* at eval time only.",
            "Greek text was not available to the blinded reconstruct path.",
            f"greek_source_id={greek_unit.greek.source_id}",
            f"paired_tokens={metrics.paired_tokens}",
        ],
    )


def score_thomas7_surviving(
    repo_root: Path | None,
    reconstruction: BlindedReconstructionPackage,
    *,
    eval_package_path: Path | None = None,
    freeze_path: Path | None = None,
    line_ids: list[str] | tuple[str, ...] = DEFAULT_THOMAS7_LINE_IDS,
    require_human_checked: bool = True,
    unclear_policy: UnclearLetterPolicy = DEFAULT_UNCLEAR_POLICY,
) -> SurvivingLetterEvalReport:
    """Official Thomas 7 surviving-letter score entry (loads fragments here)."""
    root = require_repo_root(repo_root)
    pack_path = eval_package_path or (root / DEFAULT_EVAL_PACKAGE_REL)
    freeze_p = freeze_path or (root / DEFAULT_FREEZE_REL)
    freeze_verified = False

    if require_human_checked:
        pack = assert_eval_package_human_checked(pack_path)
        if not freeze_p.is_file():
            msg = (
                f"official scoring requires freeze record at {freeze_p}; "
                "freeze the blinded reconstruction before unsealing P.Oxy. 654"
            )
            raise BlindedScoreIntegrityError(msg)
        freeze = load_freeze_record(freeze_p)
        verify_official_freeze(reconstruction, freeze)
        freeze_verified = True
        official = True
    else:
        pack = load_blinded_eval_package(pack_path)
        official = False
        if freeze_p.is_file():
            try:
                verify_official_freeze(reconstruction, load_freeze_record(freeze_p))
                freeze_verified = True
            except BlindedScoreIntegrityError:
                freeze_verified = False

    review_status = str(pack.get("review_status", ""))
    pack_id = str(pack.get("id", pack_path.name))

    corpus = load_fragments_at_eval_time(root)
    tx = transcription_by_id(corpus, DEFAULT_THOMAS7_TRANSCRIPTION_ID)
    gold = extract_surviving_letter_gold(
        tx, line_ids=line_ids, unclear_policy=unclear_policy
    )
    assert_restorations_excluded_from_gold(gold)

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
        "reconstruction_package_sha256": package_content_sha256(reconstruction),
        "primary_gold_letters_sha256": _sha256_bytes(
            gold.primary_gold_letters.encode("utf-8")
        ),
    }
    if freeze_p.is_file():
        digests["freeze_sha256"] = _sha256_bytes(freeze_p.read_bytes())

    return SurvivingLetterEvalReport(
        id="jak40-thomas7-surviving-letter-eval-v1",
        title="Thomas 7 surviving-letter evaluation (JAK-40)",
        reconstruction_package_id=reconstruction.id,
        eval_package_id=pack_id,
        review_status=review_status,
        official=official,
        freeze_verified=freeze_verified,
        gold=gold,
        scores=scores,
        artifact_digests=digests,
        blinding={
            "greek_loaded_during_reconstruction": False,
            "greek_loaded_at_scoring": True,
            "transcription_id": DEFAULT_THOMAS7_TRANSCRIPTION_ID,
            "line_ids": list(line_ids),
            "restorations_in_gold_denominator": False,
            "unclear_in_primary_gold": unclear_policy is UnclearLetterPolicy.INCLUDE,
            "unclear_policy": unclear_policy.value,
            "require_human_checked": require_human_checked,
            "freeze_verified": freeze_verified,
        },
        summary={
            "official": official,
            "freeze_verified": freeze_verified,
            "n_scores": len(scores),
            "gold_denominator": len(gold.primary_gold_letters),
            "clear_letters": len(gold.clear_letters),
            "unclear_letters": len(gold.unclear_letters),
            "restoration_letters_excluded": len(gold.restoration_letters_excluded),
            "metric_kind": scores[0].metric_kind if scores else None,
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
            "Editorial restorations excluded; unclear handled via unclear_policy.",
            "Gap-aware metrics — incidental_global_lcs is not gate precision/F1.",
            "Official=true requires human-checked + matching pre-unseal freeze.",
            *(
                []
                if official
                else [
                    "UNOFFICIAL protocol run: do not treat metrics as JAK-40 "
                    "acceptance scores."
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
    """Score when human-checked+frozen; otherwise return a gate-blocked report."""
    root = require_repo_root(repo_root)
    pack_path = eval_package_path or (root / DEFAULT_EVAL_PACKAGE_REL)
    try:
        return score_thomas7_surviving(
            root,
            reconstruction,
            eval_package_path=pack_path,
            require_human_checked=True,
        )
    except (EvalReviewGateError, BlindedScoreIntegrityError) as exc:
        pack = load_blinded_eval_package(pack_path)
        return SurvivingLetterEvalReport(
            id="jak40-thomas7-surviving-letter-eval-v1",
            title="Thomas 7 surviving-letter evaluation (JAK-40) — gate blocked",
            reconstruction_package_id=reconstruction.id,
            eval_package_id=str(pack.get("id", pack_path.name)),
            review_status=str(pack.get("review_status", "")),
            official=False,
            freeze_verified=False,
            gold=None,
            scores=[],
            artifact_digests={
                "eval_package_sha256": _sha256_bytes(pack_path.read_bytes()),
                "reconstruction_package_sha256": package_content_sha256(reconstruction),
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
                "Official scoring blocked (review gate and/or freeze mismatch).",
                "Blinded reconstructions remain valid; official scores deferred.",
                str(exc),
            ],
        )
