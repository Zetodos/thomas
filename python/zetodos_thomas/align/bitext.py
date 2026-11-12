# SPDX-License-Identifier: Apache-2.0
"""Prepare Greek-Sahidic bitext for statistical aligners (JAK-36).

Uses ``bound-group-v1`` Coptic segmentation and whitespace Greek tokens.
Never emits omission fixtures or evaluation-witness / Thomas Greek units as
training pairs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from zetodos_thomas.align.paratext import filter_alignable_tokens
from zetodos_thomas.align.segment_coptic import (
    ALIGNMENT_READY,
    SEGMENTATION_VERSION,
    SEGMENTED_ID_SCHEME_VERSION,
    segmented_coptic_tokens,
)
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    nonempty_positive_units,
    partition_corpus,
)
from zetodos_thomas.align.tokenize import whitespace_tokens
from zetodos_thomas.integrations.aligners.types import BitextSentence
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus

COPTIC_SEGMENTATION = "bound-group-v1"
GREEK_TOKENIZATION = "whitespace-v1"
FORBIDDEN_TRAIN_MARKERS = (
    "thomas.7",
    "poxy-654",
    "poxy.654",
    "p.oxy.654",
)


@dataclass(frozen=True)
class BitextBundle:
    """Role-bucketed bitext with provenance metadata."""

    pilot_corpus_id: str
    split_id: str
    greek_source_id: str
    sahidic_source_id: str
    coptic_segmentation: str
    greek_tokenization: str
    id_scheme_segmented: str
    alignment_ready: bool
    by_role: dict[SplitRole, list[BitextSentence]]
    checksum_sha256: str
    excluded_from_training_stack: list[str]
    notes: list[str]

    def sentences_for(self, *roles: SplitRole) -> list[BitextSentence]:
        out: list[BitextSentence] = []
        for role in roles:
            out.extend(self.by_role.get(role, []))
        return out


def _unit_to_sentence(unit: ParallelUnit, role: SplitRole) -> BitextSentence | None:
    if not (unit.greek.original_text.strip() and unit.sahidic.original_text.strip()):
        return None
    left = filter_alignable_tokens(segmented_coptic_tokens(unit.sahidic))
    right = filter_alignable_tokens(whitespace_tokens(unit.greek))
    if not left or not right:
        return None
    return BitextSentence(
        unit_id=unit.unit_id,
        source_tokens=tuple(t.normalized for t in left),
        target_tokens=tuple(t.normalized for t in right),
        source_token_ids=tuple(t.id for t in left),
        target_token_ids=tuple(t.id for t in right),
        split_role=role.value,
    )


def _assert_safe_training_stack(splits: PilotSplits) -> None:
    """Reject Thomas / P.Oxy. unit ids on the train role (belt and suspenders)."""
    for unit_id in splits.unit_ids_for(SplitRole.TRAIN):
        lowered = unit_id.lower()
        if lowered.startswith("thomas.") or "poxy" in lowered:
            msg = f"forbidden training unit in split: {unit_id}"
            raise ValueError(msg)
    blob = " ".join(splits.excluded_from_training_stack).lower()
    normalized = blob.replace(".", "").replace("-", "").replace(" ", "")
    if "poxy654" not in normalized:
        msg = (
            "split exclusions must list P.Oxy. 654 evaluation-witness "
            "(Thomas 7 leakage control)"
        )
        raise ValueError(msg)


def build_bitext(
    corpus: PilotCorpus,
    splits: PilotSplits,
) -> BitextBundle:
    """Tokenize nonempty positive units by split role for aligner consumption."""
    if not ALIGNMENT_READY:
        msg = "Coptic bound-group segmentation is not ALIGNMENT_READY"
        raise ValueError(msg)
    if splits.pilot_corpus_id and splits.pilot_corpus_id != corpus.id:
        msg = (
            f"split pilot_corpus_id {splits.pilot_corpus_id!r} != "
            f"pilot id {corpus.id!r}"
        )
        raise ValueError(msg)
    _assert_safe_training_stack(splits)

    buckets = partition_corpus(corpus, splits)
    by_role: dict[SplitRole, list[BitextSentence]] = {role: [] for role in SplitRole}
    digest = hashlib.sha256()
    for role in SplitRole:
        units = buckets[role]
        if role is SplitRole.OMISSION:
            # Omission fixtures are never positive aligner pairs.
            by_role[role] = []
            continue
        for unit in nonempty_positive_units(units):
            if role is SplitRole.TRAIN:
                lowered = unit.unit_id.lower()
                if lowered.startswith("thomas.") or "poxy" in lowered:
                    msg = f"refusing to train on excluded unit {unit.unit_id}"
                    raise ValueError(msg)
            sentence = _unit_to_sentence(unit, role)
            if sentence is None:
                continue
            by_role[role].append(sentence)
            digest.update(sentence.unit_id.encode())
            digest.update(b"\0")
            digest.update(" ".join(sentence.source_tokens).encode())
            digest.update(b"|||")
            digest.update(" ".join(sentence.target_tokens).encode())
            digest.update(b"\n")

    return BitextBundle(
        pilot_corpus_id=corpus.id,
        split_id=splits.id,
        greek_source_id=corpus.greek_source_id,
        sahidic_source_id=corpus.sahidic_source_id,
        coptic_segmentation=SEGMENTATION_VERSION,
        greek_tokenization=GREEK_TOKENIZATION,
        id_scheme_segmented=SEGMENTED_ID_SCHEME_VERSION,
        alignment_ready=ALIGNMENT_READY,
        by_role=by_role,
        checksum_sha256=digest.hexdigest(),
        excluded_from_training_stack=list(splits.excluded_from_training_stack),
        notes=[
            "Coptic = bound-group-v1 (v2 token ids); Greek = whitespace v1 ids.",
            "Omission (John 8 PA) excluded from positive bitext.",
            "Holdout units are never used as the training subset.",
            "Thomas 7 / P.Oxy. 654 remain outside this NT training stack.",
        ],
    )


def subsample_train(
    train: list[BitextSentence],
    *,
    n: int,
    seed: int,
) -> list[BitextSentence]:
    """Deterministic subsample of train (seeded sha256 order, stable across runs)."""
    if n < 0:
        msg = "n must be >= 0"
        raise ValueError(msg)

    def _key(sentence: BitextSentence) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{sentence.unit_id}".encode()).hexdigest()
        return (digest, sentence.unit_id)

    decorated = sorted(train, key=_key)
    return decorated[: min(n, len(decorated))]
