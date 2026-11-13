# SPDX-License-Identifier: Apache-2.0
"""Blinded reconstruction of Thomas 7 + NT holdouts (JAK-40).

Uses the JAK-39 decoder + JAK-37/38 evidence only. Never loads P.Oxy. / fragment
Greek or holdout Greek during reconstruction / rule selection.
"""

from __future__ import annotations

import hashlib
import json
import re
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
    DecodeSpanInput,
    SpanDecodeStatus,
)
from zetodos_thomas.schemas.ids import span_id
from zetodos_thomas.schemas.mappings import AblationCondition
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus, TextUnit

_TOKEN_RE = re.compile(r"\S+")
DECODE_SOURCE_ID = "decode-input"
DEFAULT_THOMAS_UNIT = "thomas.7"
FORBIDDEN_RECONSTRUCT_MARKERS = ("poxy", "p.oxy", "dclp-poxy")


class BlindedReconstructError(ValueError):
    """Blinded reconstruction failed a leakage or input guard."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: object) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return _sha256_bytes(blob.encode("utf-8"))


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


def coptic_surfaces_from_normalized(text: str) -> list[str]:
    """Whitespace-split Dilley / Horner normalized Coptic (already segmented)."""
    return _TOKEN_RE.findall(text)


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


def build_decode_request_for_surfaces(
    *,
    request_id: str,
    unit_id: str,
    surfaces: list[str],
    ablation_condition: AblationCondition,
    max_candidates_per_span: int = 5,
    title: str = "",
) -> DecodeRequest:
    """One span per Coptic surface (token-level blinded decode)."""
    assert_no_fragment_greek_in_scope([unit_id, request_id, DECODE_SOURCE_ID])
    spans = [
        DecodeSpanInput(
            span_id=span_id(DECODE_SOURCE_ID, unit_id, str(i)),
            source_id=DECODE_SOURCE_ID,
            unit_id=unit_id,
            surfaces=[surface],
        )
        for i, surface in enumerate(surfaces)
    ]
    return DecodeRequest(
        id=request_id,
        title=title or f"Blinded decode {unit_id}",
        owned_by="JAK-40",
        ablation_condition=ablation_condition,
        unit_id=unit_id,
        spans=spans,
        max_candidates_per_span=max_candidates_per_span,
        provenance={
            "blinded": True,
            "greek_consulted": False,
            "llm": False,
            "network": False,
        },
        notes=[
            "Blinded reconstruction input (JAK-40): Coptic only.",
            "P.Oxy. / evaluation Greek must not be consulted on this path.",
        ],
    )


def top1_from_result(result: DecodeResult) -> list[str | None]:
    out: list[str | None] = []
    for span in result.spans:
        if span.status is SpanDecodeStatus.CANDIDATES and span.candidates:
            # Join multi-surface Greek hypotheses with space for letter extract.
            out.append(" ".join(span.candidates[0].greek_surfaces))
        else:
            out.append(None)
    return out


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
    request = build_decode_request_for_surfaces(
        request_id=request_id,
        unit_id=unit_id,
        surfaces=coptic_surfaces,
        ablation_condition=model.condition,
        max_candidates_per_span=max_candidates_per_span,
    )
    result = decode_request(model, request)
    # Leakage guard: decode provenance must stay offline / non-LLM.
    if result.provenance.get("llm") is not False:
        msg = "decode result must record llm=false"
        raise BlindedReconstructError(msg)
    if result.provenance.get("network") is not False:
        msg = "decode result must record network=false"
        raise BlindedReconstructError(msg)
    if result.adopted_aligner is not None:
        msg = "adopted_aligner must remain null (keep both aligners)"
        raise BlindedReconstructError(msg)
    return BlindedUnitReconstruction(
        unit_id=unit_id,
        target_kind=target_kind,
        source_id=source_id,
        coptic_surfaces=list(coptic_surfaces),
        decode_result=result,
        top1_greek_surfaces=top1_from_result(result),
        notes=[
            "Blinded: no evaluation Greek consulted during decode.",
            f"ablation={model.condition.value}",
        ],
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
    # Reconstruct path must not touch fragment corpus files.
    assert_no_fragment_greek_in_scope(
        [
            thomas_unit_id,
            "dilley-thomas",
            # intentional: we check the marker list API, not open the file
            "decode-input",
        ]
    )

    model = load_decode_model(root, condition=condition)
    thomas = load_thomas_corpus(root)
    unit = thomas_unit(thomas, thomas_unit_id)
    surfaces = coptic_surfaces_from_normalized(unit.normalized_text)
    if not surfaces:
        msg = f"{thomas_unit_id} has empty normalized Coptic"
        raise BlindedReconstructError(msg)

    units: list[BlindedUnitReconstruction] = [
        reconstruct_unit(
            model,
            unit_id=unit.unit_id,
            target_kind=BlindedTargetKind.THOMAS_PILOT,
            source_id=THOMAS_SOURCE_ID,
            coptic_surfaces=surfaces,
            request_id=f"jak40-blinded-{unit.unit_id}",
            max_candidates_per_span=max_candidates_per_span,
        )
    ]

    holdout: list[ParallelUnit] = []
    if include_nt_holdouts:
        holdout = holdout_units(root)
        for pu in holdout:
            # Coptic only into decoder — do not pass greek surfaces.
            sah = pu.sahidic.normalized_text or pu.sahidic.original_text
            coptic = coptic_surfaces_from_normalized(sah)
            if not coptic:
                continue
            units.append(
                reconstruct_unit(
                    model,
                    unit_id=pu.unit_id,
                    target_kind=BlindedTargetKind.NT_HOLDOUT,
                    source_id=pu.sahidic.source_id,
                    coptic_surfaces=coptic,
                    request_id=f"jak40-blinded-holdout-{pu.unit_id}",
                    max_candidates_per_span=max_candidates_per_span,
                )
            )

    baselines = []
    if include_word_for_word:
        train = train_units(root)
        lex = fit_lexical_baseline(train, coptic_segmentation="whitespace")
        table = fit_argmax_table(lex)
        for u in units:
            baselines.append(
                baseline_for_unit(
                    unit_id=u.unit_id,
                    target_kind=u.target_kind,
                    source_id=u.source_id,
                    coptic_surfaces=u.coptic_surfaces,
                    argmax_table=table,
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
        },
        summary={
            "unit_count": len(units),
            "thomas_units": 1,
            "nt_holdout_units": len(
                [u for u in units if u.target_kind is BlindedTargetKind.NT_HOLDOUT]
            ),
            "spans_with_candidates": n_cand,
            "spans_abstained": n_abs,
            "word_for_word_units": len(baselines),
            "ablation_condition": condition.value,
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
        },
        notes=[
            "Blinded reconstruction: Coptic Dilley thomas.7 + NT holdout Sahidic only.",
            "P.Oxy. 654 Greek is not consulted on this path (evaluation witness).",
            "Dual aligners kept (eflomal + fast_align); adopted_aligner=null.",
            "No LLM on the reconstruction path.",
            "Word-for-word baseline is train-cooccurrence argmax for JAK-41 compare.",
        ],
    )


def write_reconstruction_package(
    package: BlindedReconstructionPackage,
    path: Path,
) -> str:
    """Write package JSON; return sha256 of bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(package.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    )
    data = text.encode("utf-8")
    path.write_bytes(data)
    return _sha256_bytes(data)


def package_summary(package: BlindedReconstructionPackage) -> dict[str, Any]:
    return dict(package.summary)
