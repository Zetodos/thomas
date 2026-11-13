# SPDX-License-Identifier: Apache-2.0
"""Blinded reconstruction of Thomas 7 + NT holdouts (JAK-40).

Uses the JAK-39 decoder + JAK-37/38 evidence only. Never loads P.Oxy. / fragment
Greek or holdout Greek during reconstruction / rule selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from zetodos_thomas.align.lexical_baseline import fit_lexical_baseline
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    load_canonical_splits,
    nonempty_positive_units,
    partition_corpus,
)
from zetodos_thomas.eval.spans import (
    DILLEY_SEGMENTATION,
    HORNER_SEGMENTATION,
    construction_patterns_from_model,
    dilley_normalized_tokens,
    horner_bound_group_tokens,
    resolve_covering_spans,
    windows_to_decode_spans,
)
from zetodos_thomas.eval.surviving_letters import write_freeze_record
from zetodos_thomas.eval.word_for_word import baseline_for_unit, fit_argmax_table
from zetodos_thomas.ingest.dilley_thomas import CORPUS_ID as THOMAS_CORPUS_ID
from zetodos_thomas.ingest.dilley_thomas import SOURCE_ID as THOMAS_SOURCE_ID
from zetodos_thomas.ingest.paths import (
    require_repo_root,
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
    resolve_pilot_path,
    resolve_thomas_path,
)
from zetodos_thomas.reconstruct.decoder import decode_request
from zetodos_thomas.reconstruct.model import DecodeModel, load_decode_model
from zetodos_thomas.reconstruct.ranking import RANKING_RULES_ID
from zetodos_thomas.schemas.alignment import ThomasCorpus
from zetodos_thomas.schemas.blinded import (
    BlindedReconstructionPackage,
    BlindedTargetKind,
    BlindedUnitReconstruction,
)
from zetodos_thomas.schemas.decode import (
    DecodeRequest,
    DecodeResult,
    SpanDecodeStatus,
)
from zetodos_thomas.schemas.mappings import AblationCondition
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus, TextUnit

DEFAULT_THOMAS_UNIT = "thomas.7"
FORBIDDEN_RECONSTRUCT_MARKERS = ("poxy", "p.oxy", "dclp-poxy")
COMMITTED_ARTIFACT_MAX_BYTES = 1024 * 1024
DEFAULT_THOMAS_RECON_REL = Path(
    "data/reconstructions/jak40/blinded-reconstruction-v1.json"
)
DEFAULT_NT_RECON_REL = Path(
    "data/reconstructions/jak40/nt-holdout-blinded-reconstruction-v1.json"
)
DEFAULT_FREEZE_REL = Path("data/reconstructions/jak40/thomas-7-blinded-freeze-v1.json")
COMMITTED_NT_MAX_CANDIDATES = 1


class BlindedReconstructError(ValueError):
    """Blinded reconstruction failed a leakage or input guard."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_no_fragment_greek_in_scope(names_or_paths: list[str]) -> None:
    """Refuse reconstruct if a caller tries to inject fragment paths/handles."""
    for raw in names_or_paths:
        lowered = raw.lower().replace(" ", "")
        if any(marker in lowered for marker in FORBIDDEN_RECONSTRUCT_MARKERS):
            msg = (
                "blinded reconstruct must not receive P.Oxy. / fragment Greek "
                f"handles or paths (got {raw!r})"
            )
            raise BlindedReconstructError(msg)


def load_thomas_corpus(repo_root: Path) -> ThomasCorpus:
    path = resolve_thomas_path(repo_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    return ThomasCorpus.model_validate(data)


def load_pilot_corpus(repo_root: Path) -> PilotCorpus:
    path = resolve_pilot_path(repo_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    return PilotCorpus.model_validate(data)


def thomas_unit(corpus: ThomasCorpus, unit_id: str) -> TextUnit:
    for unit in corpus.units:
        if unit.unit_id == unit_id:
            return unit
    msg = f"Thomas unit {unit_id!r} not found in {THOMAS_CORPUS_ID}"
    raise KeyError(msg)


def top1_from_result(result: DecodeResult) -> list[str | None]:
    out: list[str | None] = []
    for span in result.spans:
        if span.status is SpanDecodeStatus.CANDIDATES and span.candidates:
            out.append(" ".join(span.candidates[0].greek_surfaces))
        else:
            out.append(None)
    return out


def reconstruct_unit_from_text(
    model: DecodeModel,
    *,
    text_unit: TextUnit,
    target_kind: BlindedTargetKind,
    request_id: str,
    max_candidates_per_span: int = 5,
) -> BlindedUnitReconstruction:
    """Tokenize with source-appropriate path; cover with multi-token windows."""
    assert_no_fragment_greek_in_scope(
        [text_unit.unit_id, text_unit.source_id, request_id]
    )
    if text_unit.source_id == THOMAS_SOURCE_ID:
        stream = dilley_normalized_tokens(text_unit)
    elif text_unit.source_id == "copsahhorner":
        stream = horner_bound_group_tokens(text_unit)
    else:
        msg = f"unsupported Coptic source_id {text_unit.source_id!r}"
        raise BlindedReconstructError(msg)

    patterns = construction_patterns_from_model(model.constructions_by_pattern)
    windows = resolve_covering_spans(stream, patterns)
    spans = windows_to_decode_spans(stream, windows)
    if not spans:
        msg = f"no decode spans for {text_unit.unit_id}"
        raise BlindedReconstructError(msg)

    request = DecodeRequest(
        id=request_id,
        title=f"Blinded decode {text_unit.unit_id}",
        owned_by="JAK-40",
        ablation_condition=model.condition,
        unit_id=text_unit.unit_id,
        spans=spans,
        max_candidates_per_span=max_candidates_per_span,
        provenance={
            "blinded": True,
            "greek_consulted": False,
            "llm": False,
            "network": False,
            "segmentation": stream.segmentation,
            "id_scheme_version": stream.id_scheme_version,
        },
        notes=[
            "Blinded reconstruction input (JAK-40): Coptic only.",
            "P.Oxy. / evaluation Greek must not be consulted on this path.",
            f"segmentation={stream.segmentation}",
            "Covering spans include multi-token JAK-38 constructions when matched.",
        ],
    )
    result = decode_request(model, request)
    if result.provenance.get("llm") is not False:
        msg = "decode result must record llm=false"
        raise BlindedReconstructError(msg)
    if result.provenance.get("network") is not False:
        msg = "decode result must record network=false"
        raise BlindedReconstructError(msg)
    if result.adopted_aligner is not None:
        msg = "adopted_aligner must remain null (keep both aligners)"
        raise BlindedReconstructError(msg)
    for span in result.spans:
        if span.input_span is None:
            msg = f"span {span.span_id} missing input_span provenance"
            raise BlindedReconstructError(msg)

    surfaces = [t.normalized for t in stream.tokens]
    n_construction = sum(
        1 for w in windows if w.kind == "construction" and w.end - w.start > 1
    )
    return BlindedUnitReconstruction(
        unit_id=text_unit.unit_id,
        target_kind=target_kind,
        source_id=text_unit.source_id,
        coptic_surfaces=surfaces,
        segmentation=stream.segmentation,
        id_scheme_version=stream.id_scheme_version,
        decode_result=result,
        top1_greek_surfaces=top1_from_result(result),
        covering_span_kinds=[w.kind for w in windows],
        notes=[
            "Blinded: no evaluation Greek consulted during decode.",
            f"ablation={model.condition.value}",
            f"segmentation={stream.segmentation}",
            f"multi_token_construction_spans={n_construction}",
            f"covering_spans={len(windows)}",
        ],
    )


def reconstruct_unit(
    model: DecodeModel,
    *,
    unit_id: str,
    target_kind: BlindedTargetKind,
    source_id: str,
    coptic_surfaces: list[str],
    request_id: str,
    max_candidates_per_span: int = 5,
) -> BlindedUnitReconstruction:
    """Compatibility wrapper: mint a TextUnit and run the source-aware path."""
    text = " ".join(coptic_surfaces)
    # Minimal TextUnit for tests / ad-hoc surfaces; production uses corpus units.
    book = unit_id.split(".", 1)[0] if "." in unit_id else unit_id
    text_unit = TextUnit(
        unit_id=unit_id,
        source_id=source_id,
        book=book,
        chapter=0,
        verse=0,
        original_text=text,
        normalized_text=text,
    )
    return reconstruct_unit_from_text(
        model,
        text_unit=text_unit,
        target_kind=target_kind,
        request_id=request_id,
        max_candidates_per_span=max_candidates_per_span,
    )


def _pilot_splits(repo_root: Path) -> PilotSplits:
    return load_canonical_splits(
        resolve_nt_split_path(repo_root),
        resolve_omission_fixtures_path(repo_root),
    )


def holdout_units(repo_root: Path) -> list[ParallelUnit]:
    """Return nonempty holdout ParallelUnits (Greek present but unused here)."""
    corpus = load_pilot_corpus(repo_root)
    splits = _pilot_splits(repo_root)
    buckets = partition_corpus(corpus, splits)
    return nonempty_positive_units(buckets[SplitRole.HOLDOUT])


def train_units(repo_root: Path) -> list[ParallelUnit]:
    corpus = load_pilot_corpus(repo_root)
    splits = _pilot_splits(repo_root)
    buckets = partition_corpus(corpus, splits)
    return nonempty_positive_units(buckets[SplitRole.TRAIN])


def reconstruct_blinded_package(
    repo_root: Path | None = None,
    *,
    ablation: AblationCondition | str = (
        AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
    ),
    thomas_unit_id: str = DEFAULT_THOMAS_UNIT,
    include_nt_holdouts: bool = True,
    max_candidates_per_span: int = 5,
    include_word_for_word: bool = True,
) -> BlindedReconstructionPackage:
    """Run blinded Thomas 7 (+ optional NT holdouts) reconstruction offline."""
    root = require_repo_root(repo_root)
    if isinstance(ablation, AblationCondition):
        condition = ablation
    else:
        condition = AblationCondition(ablation)
    assert_no_fragment_greek_in_scope([thomas_unit_id, "dilley-thomas"])

    model = load_decode_model(root, condition=condition)
    thomas = load_thomas_corpus(root)
    unit = thomas_unit(thomas, thomas_unit_id)

    units: list[BlindedUnitReconstruction] = [
        reconstruct_unit_from_text(
            model,
            text_unit=unit,
            target_kind=BlindedTargetKind.THOMAS_PILOT,
            request_id=f"jak40-blinded-{unit.unit_id}",
            max_candidates_per_span=max_candidates_per_span,
        )
    ]

    if include_nt_holdouts:
        for pu in holdout_units(root):
            units.append(
                reconstruct_unit_from_text(
                    model,
                    text_unit=pu.sahidic,
                    target_kind=BlindedTargetKind.NT_HOLDOUT,
                    request_id=f"jak40-blinded-holdout-{pu.unit_id}",
                    max_candidates_per_span=max_candidates_per_span,
                )
            )

    baselines = []
    if include_word_for_word:
        train = train_units(root)
        # Train-parity segmentation for the elementary baseline.
        lex = fit_lexical_baseline(train, coptic_segmentation="bound-group-v1")
        table = fit_argmax_table(lex)
        for u in units:
            baselines.append(
                baseline_for_unit(
                    unit_id=u.unit_id,
                    target_kind=u.target_kind,
                    source_id=u.source_id,
                    coptic_surfaces=u.coptic_surfaces,
                    argmax_table=table,
                    segmentation=u.segmentation,
                )
            )

    thomas_path = resolve_thomas_path(root)
    pilot_path = resolve_pilot_path(root)
    split_path = resolve_nt_split_path(root)
    digests = {
        "dilley_thomas_sha256": _sha256_bytes(thomas_path.read_bytes()),
        "pilot_corpus_sha256": _sha256_bytes(pilot_path.read_bytes()),
        "nt_split_sha256": _sha256_bytes(split_path.read_bytes()),
    }

    n_cand = sum(u.decode_result.summary.get("spans_with_candidates", 0) for u in units)
    n_abs = sum(u.decode_result.summary.get("spans_abstained", 0) for u in units)
    n_multi = sum(
        1
        for u in units
        for kind, span in zip(
            u.covering_span_kinds, u.decode_result.spans, strict=False
        )
        if kind == "construction" and len(span.surfaces) > 1
    )
    return BlindedReconstructionPackage(
        id="jak40-blinded-reconstruction-v1",
        title="Blinded Thomas 7 + NT holdout reconstructions (JAK-40)",
        ablation_condition=condition,
        ranking_rules_id=RANKING_RULES_ID,
        phase2_active_aligners=["eflomal", "fast_align"],
        adopted_aligner=None,
        units=units,
        word_for_word_baseline=baselines,
        artifact_digests=digests,
        evidence_provenance=dict(model.provenance),
        blinding={
            "protocol": "surviving-letter-blinded-v1",
            "greek_consulted_during_reconstruction": False,
            "poxy_654_sealed_until": "scoring",
            "thomas_unit_id": thomas_unit_id,
            "nt_holdout_unit_count": len(
                [u for u in units if u.target_kind is BlindedTargetKind.NT_HOLDOUT]
            ),
            "train_dev_not_scored_as_holdout": True,
            "omission_fixtures_excluded": True,
            "fragment_corpus_opened": False,
            "dilley_segmentation": DILLEY_SEGMENTATION,
            "horner_segmentation": HORNER_SEGMENTATION,
            "multi_token_construction_spans": n_multi,
        },
        summary={
            "unit_count": len(units),
            "thomas_units": sum(
                1 for u in units if u.target_kind is BlindedTargetKind.THOMAS_PILOT
            ),
            "nt_holdout_units": len(
                [u for u in units if u.target_kind is BlindedTargetKind.NT_HOLDOUT]
            ),
            "spans_with_candidates": n_cand,
            "spans_abstained": n_abs,
            "word_for_word_units": len(baselines),
            "ablation_condition": condition.value,
            "multi_token_construction_spans": n_multi,
        },
        provenance={
            "owned_by": "JAK-40",
            "decoder": "JAK-39",
            "llm": False,
            "network": False,
            "phase2_active_aligners": ["eflomal", "fast_align"],
            "adopted_aligner": None,
            "pilot_corpus_id": model.provenance.get("pilot_corpus_id"),
            "split_id": model.provenance.get("split_id"),
            "coptic_segmentation_train": model.provenance.get("coptic_segmentation"),
        },
        notes=[
            "Blinded reconstruction: Coptic Dilley thomas.7 + NT holdout Sahidic only.",
            "P.Oxy. 654 Greek is not consulted on this path (evaluation witness).",
            "Dual aligners kept (eflomal + fast_align); adopted_aligner=null.",
            "No LLM on the reconstruction path.",
            "NT uses bound-group-v1; Dilley uses dilley-normalized-whitespace-v1.",
            "Covering spans apply multi-token JAK-38 constructions when matched.",
            "Word-for-word baseline uses train-parity bound-group-v1 cooccurrence.",
        ],
    )


def write_reconstruction_package(
    package: BlindedReconstructionPackage,
    path: Path,
    *,
    max_bytes: int | None = COMMITTED_ARTIFACT_MAX_BYTES,
    compact: bool = False,
) -> str:
    """Write package JSON; return sha256 of bytes written.

    When ``compact`` is false but the pretty payload exceeds ``max_bytes``,
    automatically rewrite with compact separators so provenance-heavy NT
    packages stay under ``check-added-large-files`` (1024 KiB).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = package.model_dump(mode="json")

    def _encode(*, pretty: bool) -> bytes:
        if pretty:
            return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            )
        return (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    data = _encode(pretty=not compact)
    used_compact = compact
    if max_bytes is not None and len(data) > max_bytes and not compact:
        data = _encode(pretty=False)
        used_compact = True
    if max_bytes is not None and len(data) > max_bytes:
        msg = (
            f"reconstruction artifact {path} is {len(data)} bytes "
            f"(limit {max_bytes}); split Thomas vs NT holdouts or lower "
            "max_candidates_per_span before committing"
        )
        raise BlindedReconstructError(msg)
    path.write_bytes(data)
    if used_compact and not compact:
        # Auditable note on disk size strategy (digest is of bytes written).
        _ = used_compact
    return _sha256_bytes(data)


def _retitle_package(
    package: BlindedReconstructionPackage,
    *,
    package_id: str,
    title: str,
    extra_notes: list[str],
) -> BlindedReconstructionPackage:
    n_cand = sum(
        u.decode_result.summary.get("spans_with_candidates", 0) for u in package.units
    )
    n_abs = sum(
        u.decode_result.summary.get("spans_abstained", 0) for u in package.units
    )
    thomas_n = sum(
        1 for u in package.units if u.target_kind is BlindedTargetKind.THOMAS_PILOT
    )
    holdout_n = sum(
        1 for u in package.units if u.target_kind is BlindedTargetKind.NT_HOLDOUT
    )
    return package.model_copy(
        update={
            "id": package_id,
            "title": title,
            "summary": {
                **package.summary,
                "unit_count": len(package.units),
                "thomas_units": thomas_n,
                "nt_holdout_units": holdout_n,
                "spans_with_candidates": n_cand,
                "spans_abstained": n_abs,
                "word_for_word_units": len(package.word_for_word_baseline),
            },
            "blinding": {
                **package.blinding,
                "nt_holdout_unit_count": holdout_n,
            },
            "notes": [*package.notes, *extra_notes],
        }
    )


def write_committed_reconstruction_artifacts(
    repo_root: Path | None = None,
    *,
    ablation: AblationCondition | str = (
        AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
    ),
    thomas_path: Path | None = None,
    nt_path: Path | None = None,
    freeze_path: Path | None = None,
) -> dict[str, str]:
    """Write split committed artifacts + Thomas 7 pre-unseal freeze pin."""
    root = require_repo_root(repo_root)
    thomas_out = thomas_path or (root / DEFAULT_THOMAS_RECON_REL)
    nt_out = nt_path or (root / DEFAULT_NT_RECON_REL)
    freeze_out = freeze_path or (root / DEFAULT_FREEZE_REL)

    thomas_pkg = reconstruct_blinded_package(
        root,
        ablation=ablation,
        include_nt_holdouts=False,
        max_candidates_per_span=5,
        include_word_for_word=True,
    )
    thomas_pkg = _retitle_package(
        thomas_pkg,
        package_id="jak40-blinded-reconstruction-v1",
        title="Blinded Thomas 7 reconstruction (JAK-40)",
        extra_notes=[
            "Committed artifact is Thomas 7 only; NT holdouts are a sibling file.",
        ],
    )

    nt_pkg = reconstruct_blinded_package(
        root,
        ablation=ablation,
        include_nt_holdouts=True,
        max_candidates_per_span=COMMITTED_NT_MAX_CANDIDATES,
        include_word_for_word=True,
    )
    nt_units = [
        u for u in nt_pkg.units if u.target_kind is BlindedTargetKind.NT_HOLDOUT
    ]
    nt_base = [
        b
        for b in nt_pkg.word_for_word_baseline
        if any(u.unit_id == b.unit_id for u in nt_units)
    ]
    nt_pkg = nt_pkg.model_copy(
        update={"units": nt_units, "word_for_word_baseline": nt_base}
    )
    nt_pkg = _retitle_package(
        nt_pkg,
        package_id="jak40-blinded-reconstruction-nt-holdout-v1",
        title="Blinded NT holdout reconstructions (JAK-40)",
        extra_notes=[
            f"Committed with max_candidates_per_span={COMMITTED_NT_MAX_CANDIDATES} "
            "to stay under check-added-large-files (1024 KiB).",
            "Regenerate offline with a higher --max-candidates if more ranks needed.",
        ],
    )

    digests = {
        "thomas": write_reconstruction_package(thomas_pkg, thomas_out),
        "nt_holdout": write_reconstruction_package(nt_pkg, nt_out),
    }
    # Freeze Thomas 7 BEFORE any scoring path can open P.Oxy. 654.
    freeze = write_freeze_record(thomas_pkg, freeze_out)
    digests["freeze"] = freeze.reconstruction_sha256
    return digests


def package_summary(package: BlindedReconstructionPackage) -> dict[str, Any]:
    return dict(package.summary)
