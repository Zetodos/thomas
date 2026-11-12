# SPDX-License-Identifier: Apache-2.0
"""Build Coptic→Greek lexical candidate tables from train alignments (JAK-37).

Only train-split observations feed the table. Dual-aligner provenance is
preserved (per-aligner counts + contributing_aligners). Layers stay separate:
observed wordforms, inferred lemmas, confidence estimates.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zetodos_thomas.align.lexical_baseline import (
    LexicalBaselineModel,
    fit_lexical_baseline,
)
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    nonempty_positive_units,
    partition_corpus,
)
from zetodos_thomas.align.statistical import export_alignment_artifact
from zetodos_thomas.integrations.aligners.eflomal_adapter import (
    EflomalNotAvailableError,
    align_eflomal,
)
from zetodos_thomas.integrations.aligners.fast_align_adapter import (
    FastAlignNotAvailableError,
    align_fast_align,
)
from zetodos_thomas.integrations.aligners.pharaoh import parse_pharaoh_line
from zetodos_thomas.integrations.aligners.pins import pinned_versions_payload
from zetodos_thomas.integrations.aligners.types import (
    AlignerConfig,
    AlignerName,
    AlignmentLinks,
    BitextSentence,
    DirectionalAlignment,
)
from zetodos_thomas.lexicon.lemmas import (
    infer_coptic_lemma,
    infer_greek_lemma,
    inference_method_payload,
)
from zetodos_thomas.schemas.lexicon import (
    ConfidenceComponents,
    ConfidenceEstimate,
    InferredLemmaCandidate,
    LexicalCandidateTable,
    LexicalLayer,
    ObservationExample,
    ObservedWordformCandidate,
    SplitMembership,
)
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus

DEFAULT_PHASE2_ALIGNERS: tuple[AlignerName, ...] = (
    AlignerName.EFLOMAL,
    AlignerName.FAST_ALIGN,
)
MAX_EXAMPLES_PER_ENTRY = 5
FORBIDDEN_UNIT_MARKERS = ("thomas.", "poxy")
COLUMNAR_FORMAT = "columnar-v1"


@dataclass
class _PairAccum:
    """Mutable aggregator for one Coptic↔Greek wordform pair."""

    per_aligner: Counter[str] = field(default_factory=Counter)
    unit_ids: set[str] = field(default_factory=set)
    examples: list[ObservationExample] = field(default_factory=list)


def _assert_train_only(sentences: Sequence[BitextSentence]) -> None:
    for sentence in sentences:
        if sentence.split_role != SplitRole.TRAIN.value:
            msg = (
                f"lexical candidates require train-only links; "
                f"got split_role={sentence.split_role!r} for {sentence.unit_id}"
            )
            raise ValueError(msg)
        lowered = sentence.unit_id.lower()
        if lowered.startswith("thomas.") or "poxy" in lowered:
            msg = (
                f"refusing forbidden evaluation unit in train links: {sentence.unit_id}"
            )
            raise ValueError(msg)


def _assert_safe_unit_id(unit_id: str) -> None:
    lowered = unit_id.lower()
    for marker in FORBIDDEN_UNIT_MARKERS:
        if marker in lowered:
            msg = f"forbidden unit id in lexical observations: {unit_id}"
            raise ValueError(msg)


def align_train_sentences(
    train: list[BitextSentence],
    *,
    aligners: Sequence[AlignerName] = DEFAULT_PHASE2_ALIGNERS,
    config: AlignerConfig | None = None,
    repo_root: Path | None = None,
) -> dict[str, AlignmentLinks]:
    """Run each active Phase-2 aligner on train sentences only."""
    _assert_train_only(train)
    cfg = config or AlignerConfig()
    out: dict[str, AlignmentLinks] = {}
    errors: dict[str, str] = {}
    for name in aligners:
        try:
            if name is AlignerName.EFLOMAL:
                links = align_eflomal(train, config=cfg, repo_root=repo_root)
            elif name is AlignerName.FAST_ALIGN:
                links = align_fast_align(train, config=cfg, repo_root=repo_root)
            else:
                msg = f"unsupported aligner {name}"
                raise ValueError(msg)
            links.notes = [
                *links.notes,
                "JAK-37: train-only fit+align for lexical candidate observations.",
            ]
            out[name.value] = links
        except (EflomalNotAvailableError, FastAlignNotAvailableError) as exc:
            errors[name.value] = str(exc)
    if not out:
        msg = f"no aligners available for lexical candidates: {errors}"
        raise RuntimeError(msg)
    return out


def load_compact_alignment_artifact(
    path: Path,
    sentences: Sequence[BitextSentence],
) -> AlignmentLinks:
    """Rehydrate compact JAK-36/37 link JSON onto bitext sentences."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_unit = {s.unit_id: s for s in sentences}
    rows = payload.get("rows") or []
    ordered: list[BitextSentence] = []
    forward: list[DirectionalAlignment] = []
    reverse: list[DirectionalAlignment] = []
    symmetrized: list[DirectionalAlignment] = []
    for row in rows:
        unit_id = row["unit_id"]
        sentence = by_unit.get(unit_id)
        if sentence is None:
            msg = f"artifact unit {unit_id} not found in provided bitext"
            raise ValueError(msg)
        if row.get("src_len") != len(sentence.source_tokens):
            msg = (
                f"src_len mismatch for {unit_id}: "
                f"artifact={row.get('src_len')} bitext={len(sentence.source_tokens)}"
            )
            raise ValueError(msg)
        if row.get("trg_len") != len(sentence.target_tokens):
            msg = (
                f"trg_len mismatch for {unit_id}: "
                f"artifact={row.get('trg_len')} bitext={len(sentence.target_tokens)}"
            )
            raise ValueError(msg)
        ordered.append(sentence)
        forward.append(
            DirectionalAlignment(
                unit_id=unit_id, pairs=parse_pharaoh_line(row.get("fwd", ""))
            )
        )
        reverse.append(
            DirectionalAlignment(
                unit_id=unit_id, pairs=parse_pharaoh_line(row.get("rev", ""))
            )
        )
        symmetrized.append(
            DirectionalAlignment(
                unit_id=unit_id, pairs=parse_pharaoh_line(row.get("sym", ""))
            )
        )
    aligner = AlignerName(payload["aligner"])
    return AlignmentLinks(
        aligner=aligner,
        sentences=ordered,
        forward=forward,
        reverse=reverse,
        symmetrized=symmetrized,
        symmetrization=payload.get("symmetrization", "grow-diag-final-and"),
        config=payload.get("config") or {},
        notes=list(payload.get("notes") or []),
    )


def _iter_sym_observations(
    aligner: str,
    links: AlignmentLinks,
) -> Iterable[tuple[str, str, str, str, str]]:
    """Yield (unit_id, coptic_wf, greek_wf, coptic_tid, greek_tid) from symmetrized."""
    by_pairs = {a.unit_id: a.pairs for a in links.symmetrized}
    for sentence in links.sentences:
        _assert_safe_unit_id(sentence.unit_id)
        for i, j in by_pairs.get(sentence.unit_id, ()):
            if i < 0 or j < 0:
                continue
            if i >= len(sentence.source_tokens) or j >= len(sentence.target_tokens):
                continue
            if i >= len(sentence.source_token_ids) or j >= len(
                sentence.target_token_ids
            ):
                continue
            yield (
                sentence.unit_id,
                sentence.source_tokens[i],
                sentence.target_tokens[j],
                sentence.source_token_ids[i],
                sentence.target_token_ids[j],
            )


def accumulate_wordform_observations(
    aligner_links: Mapping[str, AlignmentLinks],
    *,
    max_examples: int = MAX_EXAMPLES_PER_ENTRY,
) -> dict[tuple[str, str], _PairAccum]:
    """Aggregate train symmetrized links across aligners into wordform pairs."""
    accum: dict[tuple[str, str], _PairAccum] = {}
    for aligner, links in aligner_links.items():
        _assert_train_only(links.sentences)
        for unit_id, c_wf, g_wf, c_tid, g_tid in _iter_sym_observations(aligner, links):
            key = (c_wf, g_wf)
            bucket = accum.setdefault(key, _PairAccum())
            bucket.per_aligner[aligner] += 1
            bucket.unit_ids.add(unit_id)
            if len(bucket.examples) < max_examples:
                # Prefer examples that show dual-aligner when already present.
                existing_keys = {
                    (ex.unit_id, ex.coptic_token_id, ex.greek_token_id)
                    for ex in bucket.examples
                }
                ek = (unit_id, c_tid, g_tid)
                if ek in existing_keys:
                    for ex in bucket.examples:
                        if (
                            ex.unit_id,
                            ex.coptic_token_id,
                            ex.greek_token_id,
                        ) == ek and aligner not in ex.aligners:
                            ex.aligners.append(aligner)
                    continue
                bucket.examples.append(
                    ObservationExample(
                        unit_id=unit_id,
                        split_role=SplitMembership.TRAIN,
                        coptic_token_id=c_tid,
                        greek_token_id=g_tid,
                        coptic_wordform=c_wf,
                        greek_wordform=g_wf,
                        aligners=[aligner],
                    )
                )
    return accum


def _observation_count(bucket: _PairAccum) -> int:
    """Union-style count: max per-aligner (not sum — avoids double-counting).

    Dual-aligner provenance stays in ``per_aligner_observation_counts``. The
    headline ``observation_count`` is the strongest single-aligner attestation
    count so we never invent a larger figure by summing overlapping views of
    the same verse links.
    """
    if not bucket.per_aligner:
        return 0
    return max(bucket.per_aligner.values())


def build_observed_wordforms(
    accum: Mapping[tuple[str, str], _PairAccum],
    *,
    dice_model: LexicalBaselineModel | None = None,
) -> list[ObservedWordformCandidate]:
    """Materialize sorted observed-wordform candidates."""
    rows: list[ObservedWordformCandidate] = []
    for (c_wf, g_wf), bucket in sorted(
        accum.items(), key=lambda item: (-_observation_count(item[1]), item[0])
    ):
        count = _observation_count(bucket)
        if count == 0:
            continue
        dice: float | None = None
        if dice_model is not None:
            co = dice_model.pair_counts.get((c_wf, g_wf), 0)
            dice = dice_model.dice(c_wf, g_wf) if co else None
        rows.append(
            ObservedWordformCandidate(
                layer=LexicalLayer.OBSERVED_WORDFORM,
                coptic_wordform=c_wf,
                greek_wordform=g_wf,
                observation_count=count,
                per_aligner_observation_counts=dict(sorted(bucket.per_aligner.items())),
                contributing_aligners=sorted(bucket.per_aligner),
                source_unit_ids=sorted(bucket.unit_ids),
                split_membership=SplitMembership.TRAIN,
                examples=bucket.examples,
                dice_cooccurrence=dice,
            )
        )
    return rows


def build_inferred_lemmas(
    wordforms: Sequence[ObservedWordformCandidate],
) -> list[InferredLemmaCandidate]:
    """Roll observed wordforms up to inferred lemma proxies (separate layer)."""
    groups: dict[tuple[str, str], list[ObservedWordformCandidate]] = defaultdict(list)
    for row in wordforms:
        key = (
            infer_coptic_lemma(row.coptic_wordform),
            infer_greek_lemma(row.greek_wordform),
        )
        groups[key].append(row)

    method = inference_method_payload()
    out: list[InferredLemmaCandidate] = []
    for (c_lem, g_lem), members in sorted(
        groups.items(),
        key=lambda item: (-sum(m.observation_count for m in item[1]), item[0]),
    ):
        per_aligner: Counter[str] = Counter()
        unit_ids: set[str] = set()
        pairs: list[dict[str, str | int]] = []
        for member in members:
            for aligner, count in member.per_aligner_observation_counts.items():
                per_aligner[aligner] += count
            unit_ids.update(member.source_unit_ids)
            pairs.append(
                {
                    "coptic_wordform": member.coptic_wordform,
                    "greek_wordform": member.greek_wordform,
                    "observation_count": member.observation_count,
                }
            )
        # Headline count = max over aligners of summed wordform counts for that
        # aligner (still evidence-backed; never invented).
        obs = max(per_aligner.values()) if per_aligner else 0
        out.append(
            InferredLemmaCandidate(
                layer=LexicalLayer.INFERRED_LEMMA,
                coptic_lemma=c_lem,
                greek_lemma=g_lem,
                inference_method=method,
                observation_count=obs,
                underlying_wordform_pairs=pairs,
                contributing_aligners=sorted(per_aligner),
                source_unit_ids=sorted(unit_ids),
                split_membership=SplitMembership.TRAIN,
                per_aligner_observation_counts=dict(sorted(per_aligner.items())),
            )
        )
    return out


def _observation_support(count: int) -> float | None:
    """Monotone support in (0, 1); None when count is zero (missing evidence)."""
    if count <= 0:
        return None
    # 1→~0.5, 3→~0.75, 10→~0.91 — never claims certainty from sparse data.
    return count / (count + 1.0)


def build_confidence_estimates(
    wordforms: Sequence[ObservedWordformCandidate],
) -> list[ConfidenceEstimate]:
    """Separate confidence layer over observed wordforms (not collapsed into them)."""
    by_coptic: dict[str, list[ObservedWordformCandidate]] = defaultdict(list)
    for row in wordforms:
        by_coptic[row.coptic_wordform].append(row)

    estimates: list[ConfidenceEstimate] = []
    for row in wordforms:
        siblings = by_coptic[row.coptic_wordform]
        sibling_total = sum(s.observation_count for s in siblings)
        rel_freq: float | None
        if sibling_total <= 0:
            rel_freq = None
        else:
            rel_freq = row.observation_count / sibling_total

        n_aligners = len(row.contributing_aligners)
        agreement: float | None
        if n_aligners <= 0:
            agreement = None
        else:
            # 1 aligner → 0.5 (unsupported dual view); 2 → 1.0
            agreement = 1.0 if n_aligners >= 2 else 0.5

        support = _observation_support(row.observation_count)
        dice = row.dice_cooccurrence

        components: list[float] = []
        if rel_freq is not None:
            components.append(rel_freq)
        if agreement is not None:
            components.append(agreement)
        if support is not None:
            components.append(support)
        if dice is not None and dice > 0:
            components.append(dice)

        confidence: float | None
        uncertainty: float | None
        if not components:
            confidence = None
            uncertainty = None
        else:
            confidence = sum(components) / len(components)
            uncertainty = 1.0 - confidence

        estimates.append(
            ConfidenceEstimate(
                layer=LexicalLayer.CONFIDENCE_ESTIMATE,
                applies_to_layer=LexicalLayer.OBSERVED_WORDFORM,
                key={
                    "coptic_wordform": row.coptic_wordform,
                    "greek_wordform": row.greek_wordform,
                },
                method="relative_frequency_aligner_agreement_support_v1",
                confidence=confidence,
                uncertainty=uncertainty,
                components=ConfidenceComponents(
                    relative_frequency_among_coptic_candidates=rel_freq,
                    aligner_agreement=agreement,
                    observation_support=support,
                    dice_baseline=dice,
                    notes=(
                        "Mean of available components only; null components omitted. "
                        "Never fabricates observation counts."
                    ),
                ),
                evidence={
                    "observation_count": row.observation_count,
                    "per_aligner_observation_counts": (
                        row.per_aligner_observation_counts
                    ),
                    "contributing_aligners": row.contributing_aligners,
                    "source_unit_id_count": len(row.source_unit_ids),
                },
            )
        )
    return estimates


def build_lexical_candidate_table(
    aligner_links: Mapping[str, AlignmentLinks],
    *,
    provenance: dict[str, Any] | None = None,
    train_units: Sequence[ParallelUnit] | None = None,
    max_examples: int = MAX_EXAMPLES_PER_ENTRY,
) -> LexicalCandidateTable:
    """Assemble the three-layer lexical candidate table from train links."""
    for links in aligner_links.values():
        _assert_train_only(links.sentences)

    dice_model: LexicalBaselineModel | None = None
    if train_units:
        dice_model = fit_lexical_baseline(
            list(train_units),
            coptic_segmentation="bound-group-v1",
        )

    accum = accumulate_wordform_observations(aligner_links, max_examples=max_examples)
    wordforms = build_observed_wordforms(accum, dice_model=dice_model)
    lemmas = build_inferred_lemmas(wordforms)
    confidence = build_confidence_estimates(wordforms)

    active = sorted(aligner_links)
    per_aligner_pair_types: dict[str, int] = {}
    for name in active:
        pair_types = {
            (c_wf, g_wf)
            for _unit, c_wf, g_wf, _c_tid, _g_tid in _iter_sym_observations(
                name, aligner_links[name]
            )
        }
        per_aligner_pair_types[name] = len(pair_types)

    return LexicalCandidateTable(
        id="coptic-greek-lexical-candidates-v1",
        title=(
            "Coptic→Greek lexical candidates "
            "(observed wordforms / inferred lemmas / confidence)"
        ),
        owned_by="JAK-37",
        schema_version="1",
        layers={
            "observed_wordform": {
                "description": (
                    "Attested Coptic↔Greek surface forms from train symmetrized "
                    "alignments. Counts are evidence-backed."
                ),
                "entry_count": len(wordforms),
            },
            "inferred_lemma": {
                "description": (
                    "Derived lemma proxies from observed wordforms. Method labels "
                    "are mandatory; not MorphGNT/SCRIPTORIUM lemmas."
                ),
                "entry_count": len(lemmas),
                "inference_method": inference_method_payload(),
            },
            "confidence_estimate": {
                "description": (
                    "Separate confidence/uncertainty scores keyed to observed "
                    "wordform pairs. Null when evidence is insufficient."
                ),
                "entry_count": len(confidence),
                "method": "relative_frequency_aligner_agreement_support_v1",
            },
        },
        observed_wordforms=wordforms,
        inferred_lemmas=lemmas,
        confidence_estimates=confidence,
        provenance=provenance or {},
        summary={
            "phase2_active_aligners": active,
            "adopted_aligner": None,
            "observed_wordform_types": len(wordforms),
            "inferred_lemma_types": len(lemmas),
            "confidence_estimate_rows": len(confidence),
            "per_aligner_wordform_pair_types": per_aligner_pair_types,
            "train_sentence_counts": {
                name: len(links.sentences) for name, links in aligner_links.items()
            },
        },
        notes=[
            "Layers are separate: do not collapse wordforms, lemmas, and confidence.",
            "Only train-split observations feed this table (JAK-33/43 holdout rules).",
            "No Thomas-7 / P.Oxy. 654 leakage; Vaticanus B* Greek only.",
            "observation_count is max(per_aligner counts), not a fabricated sum.",
            "Jake decision (JAK-36): phase2_active_aligners=[eflomal, fast_align]; "
            "adopted_aligner=null.",
            "Dice values are optional secondary metrics from JAK-28 fit on the same "
            "train units — not a redo of the elementary baseline as the table.",
        ],
    )


def table_to_columnar_payload(table: LexicalCandidateTable) -> dict[str, Any]:
    """Serialize layers in columnar form (keeps committed artifacts under 1 MB)."""
    aligner_names = sorted(
        {a for row in table.observed_wordforms for a in row.contributing_aligners}
        | {a for row in table.inferred_lemmas for a in row.contributing_aligners}
    )
    if not aligner_names:
        aligner_names = ["eflomal", "fast_align"]

    unit_id_table: list[str] = sorted(
        {uid for row in table.observed_wordforms for uid in row.source_unit_ids}
    )
    unit_to_idx = {uid: i for i, uid in enumerate(unit_id_table)}

    obs_fields = [
        "coptic_wordform",
        "greek_wordform",
        "observation_count",
        *[f"count_{name}" for name in aligner_names],
        "source_unit_indexes",
        "example_refs",
        "dice_cooccurrence",
    ]
    obs_rows: list[list[Any]] = []
    for row in table.observed_wordforms:
        # Unit-index examples only (token ids recoverable from train-links).
        example_refs: list[int] = []
        for ex in row.examples:
            if ex.unit_id not in unit_to_idx:
                continue
            idx = unit_to_idx[ex.unit_id]
            if idx not in example_refs:
                example_refs.append(idx)
        obs_rows.append(
            [
                row.coptic_wordform,
                row.greek_wordform,
                row.observation_count,
                *[
                    row.per_aligner_observation_counts.get(name, 0)
                    for name in aligner_names
                ],
                [unit_to_idx[uid] for uid in row.source_unit_ids if uid in unit_to_idx],
                example_refs,
                row.dice_cooccurrence,
            ]
        )

    obs_index = {
        (row.coptic_wordform, row.greek_wordform): i
        for i, row in enumerate(table.observed_wordforms)
    }
    lem_fields = [
        "coptic_lemma",
        "greek_lemma",
        "observation_count",
        *[f"count_{name}" for name in aligner_names],
        "observed_indices",
    ]
    lem_rows: list[list[Any]] = []
    for row in table.inferred_lemmas:
        indices = [
            obs_index[(str(pair["coptic_wordform"]), str(pair["greek_wordform"]))]
            for pair in row.underlying_wordform_pairs
            if (str(pair["coptic_wordform"]), str(pair["greek_wordform"])) in obs_index
        ]
        lem_rows.append(
            [
                row.coptic_lemma,
                row.greek_lemma,
                row.observation_count,
                *[
                    row.per_aligner_observation_counts.get(name, 0)
                    for name in aligner_names
                ],
                indices,
            ]
        )

    conf_fields = [
        "observed_index",
        "confidence",
        "uncertainty",
        "relative_frequency",
        "aligner_agreement",
        "observation_support",
        "dice_baseline",
    ]
    conf_rows: list[list[Any]] = []
    for row in table.confidence_estimates:
        key = (
            row.key.get("coptic_wordform", ""),
            row.key.get("greek_wordform", ""),
        )
        idx = obs_index.get(key)
        if idx is None:
            continue
        conf_rows.append(
            [
                idx,
                row.confidence,
                row.uncertainty,
                row.components.relative_frequency_among_coptic_candidates,
                row.components.aligner_agreement,
                row.components.observation_support,
                row.components.dice_baseline,
            ]
        )

    return {
        "id": table.id,
        "title": table.title,
        "owned_by": table.owned_by,
        "schema_version": table.schema_version,
        "format": COLUMNAR_FORMAT,
        "layers": table.layers,
        "aligner_names": aligner_names,
        "unit_id_table": unit_id_table,
        "observed_wordforms": {"fields": obs_fields, "rows": obs_rows},
        "inferred_lemmas": {
            "fields": lem_fields,
            "rows": lem_rows,
            "inference_method": inference_method_payload(),
        },
        "confidence_estimates": {
            "fields": conf_fields,
            "rows": conf_rows,
            "method": "relative_frequency_aligner_agreement_support_v1",
            "applies_to_layer": LexicalLayer.OBSERVED_WORDFORM.value,
            "notes": (
                "observed_index references observed_wordforms.rows; "
                "null scores mean insufficient evidence (not fabricated)."
            ),
        },
        "provenance": table.provenance,
        "summary": table.summary,
        "notes": table.notes,
    }


def columnar_payload_to_table(payload: dict[str, Any]) -> LexicalCandidateTable:
    """Expand columnar-v1 JSON back into the rich pydantic table model."""
    aligner_names: list[str] = list(payload.get("aligner_names") or [])
    unit_id_table: list[str] = list(payload.get("unit_id_table") or [])
    obs_block = payload["observed_wordforms"]
    obs_fields: list[str] = obs_block["fields"]
    observed: list[ObservedWordformCandidate] = []
    for raw in obs_block["rows"]:
        row = dict(zip(obs_fields, raw, strict=True))
        per_aligner = {
            name: int(row.get(f"count_{name}", 0) or 0)
            for name in aligner_names
            if int(row.get(f"count_{name}", 0) or 0) > 0
        }
        if "source_unit_indexes" in row:
            source_unit_ids = [
                unit_id_table[i]
                for i in (row.get("source_unit_indexes") or [])
                if isinstance(i, int) and 0 <= i < len(unit_id_table)
            ]
        else:
            source_unit_ids = list(row.get("source_unit_ids") or [])
        examples: list[ObservationExample] = []
        for ref in row.get("example_refs") or []:
            # columnar-v1: unit index; older shapes: [unit, c_tid, g_tid, ...].
            c_tid = ""
            g_tid = ""
            aligners = sorted(per_aligner)
            if isinstance(ref, int):
                unit_ref = ref
            elif isinstance(ref, list) and ref:
                unit_ref = ref[0]
                if len(ref) >= 3:
                    c_tid = str(ref[1])
                    g_tid = str(ref[2])
                if len(ref) >= 4:
                    aligners = list(ref[3])
            else:
                continue
            if isinstance(unit_ref, int):
                unit_id = (
                    unit_id_table[unit_ref]
                    if 0 <= unit_ref < len(unit_id_table)
                    else str(unit_ref)
                )
            else:
                unit_id = str(unit_ref)
            examples.append(
                ObservationExample(
                    unit_id=unit_id,
                    split_role=SplitMembership.TRAIN,
                    coptic_token_id=c_tid,
                    greek_token_id=g_tid,
                    coptic_wordform=str(row["coptic_wordform"]),
                    greek_wordform=str(row["greek_wordform"]),
                    aligners=list(aligners),
                )
            )
        observed.append(
            ObservedWordformCandidate(
                layer=LexicalLayer.OBSERVED_WORDFORM,
                coptic_wordform=str(row["coptic_wordform"]),
                greek_wordform=str(row["greek_wordform"]),
                observation_count=int(row["observation_count"]),
                per_aligner_observation_counts=per_aligner,
                contributing_aligners=sorted(per_aligner),
                source_unit_ids=source_unit_ids,
                split_membership=SplitMembership.TRAIN,
                examples=examples,
                dice_cooccurrence=row.get("dice_cooccurrence"),
            )
        )

    lem_block = payload["inferred_lemmas"]
    lem_fields = lem_block["fields"]
    lemmas: list[InferredLemmaCandidate] = []
    method = lem_block.get("inference_method") or inference_method_payload()
    for raw in lem_block["rows"]:
        row = dict(zip(lem_fields, raw, strict=True))
        per_aligner = {
            name: int(row.get(f"count_{name}", 0) or 0)
            for name in aligner_names
            if int(row.get(f"count_{name}", 0) or 0) > 0
        }
        underlying: list[dict[str, str | int]] = []
        unit_ids: set[str] = set()
        if "observed_indices" in row:
            for idx in row.get("observed_indices") or []:
                if not isinstance(idx, int) or idx < 0 or idx >= len(observed):
                    continue
                src = observed[idx]
                underlying.append(
                    {
                        "coptic_wordform": src.coptic_wordform,
                        "greek_wordform": src.greek_wordform,
                        "observation_count": src.observation_count,
                    }
                )
                unit_ids.update(src.source_unit_ids)
        else:
            underlying = list(row.get("underlying_wordform_pairs") or [])
            unit_ids = set(row.get("source_unit_ids") or [])
        lemmas.append(
            InferredLemmaCandidate(
                layer=LexicalLayer.INFERRED_LEMMA,
                coptic_lemma=str(row["coptic_lemma"]),
                greek_lemma=str(row["greek_lemma"]),
                inference_method=method,
                observation_count=int(row["observation_count"]),
                underlying_wordform_pairs=underlying,
                contributing_aligners=sorted(per_aligner),
                source_unit_ids=sorted(unit_ids),
                split_membership=SplitMembership.TRAIN,
                per_aligner_observation_counts=per_aligner,
            )
        )

    conf_block = payload["confidence_estimates"]
    conf_fields = conf_block["fields"]
    conf_method = conf_block.get(
        "method", "relative_frequency_aligner_agreement_support_v1"
    )
    confidence: list[ConfidenceEstimate] = []
    for raw in conf_block["rows"]:
        row = dict(zip(conf_fields, raw, strict=True))
        idx = int(row["observed_index"])
        if idx < 0 or idx >= len(observed):
            continue
        target = observed[idx]
        confidence.append(
            ConfidenceEstimate(
                layer=LexicalLayer.CONFIDENCE_ESTIMATE,
                applies_to_layer=LexicalLayer.OBSERVED_WORDFORM,
                key={
                    "coptic_wordform": target.coptic_wordform,
                    "greek_wordform": target.greek_wordform,
                },
                method=conf_method,
                confidence=row.get("confidence"),
                uncertainty=row.get("uncertainty"),
                components=ConfidenceComponents(
                    relative_frequency_among_coptic_candidates=row.get(
                        "relative_frequency"
                    ),
                    aligner_agreement=row.get("aligner_agreement"),
                    observation_support=row.get("observation_support"),
                    dice_baseline=row.get("dice_baseline"),
                ),
                evidence={
                    "observation_count": target.observation_count,
                    "per_aligner_observation_counts": (
                        target.per_aligner_observation_counts
                    ),
                    "contributing_aligners": target.contributing_aligners,
                    "source_unit_id_count": len(target.source_unit_ids),
                    "observed_index": idx,
                },
            )
        )

    return LexicalCandidateTable(
        id=payload["id"],
        title=payload.get("title", ""),
        owned_by=payload.get("owned_by", "JAK-37"),
        schema_version=str(payload.get("schema_version", "1")),
        layers=payload.get("layers") or {},
        observed_wordforms=observed,
        inferred_lemmas=lemmas,
        confidence_estimates=confidence,
        provenance=payload.get("provenance") or {},
        summary=payload.get("summary") or {},
        notes=list(payload.get("notes") or []),
    )


def write_lexical_candidate_table(
    table: LexicalCandidateTable,
    path: Path,
    *,
    compact: bool = True,
    columnar: bool = True,
) -> None:
    """Write inspectable JSON.

    Default is columnar-v1 + compact separators so the committed train table
    stays under the pre-commit large-file ceiling while preserving all three
    layers, unit ids, and counts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    if columnar:
        payload = table_to_columnar_payload(table)
    else:
        payload = table.model_dump(mode="json")
    if compact:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def load_lexical_candidate_table(path: Path) -> LexicalCandidateTable:
    """Load columnar or expanded lexical candidate JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") == COLUMNAR_FORMAT:
        return columnar_payload_to_table(payload)
    return LexicalCandidateTable.model_validate(payload)


def export_train_link_artifacts(
    aligner_links: Mapping[str, AlignmentLinks],
    *,
    out_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Path]:
    """Write compact train-only link JSON per aligner (inspectable inputs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, links in aligner_links.items():
        path = out_dir / f"{name}-train-links-v1.json"
        export_alignment_artifact(
            links,
            path=path,
            provenance={**provenance, "aligner": name, "owned_by": "JAK-37"},
            compact=True,
        )
        written[name] = path
    return written


def generate_lexical_candidates(
    *,
    corpus: PilotCorpus,
    splits: PilotSplits,
    bitext_sentences: list[BitextSentence],
    repo_root: Path | None = None,
    config: AlignerConfig | None = None,
    aligners: Sequence[AlignerName] = DEFAULT_PHASE2_ALIGNERS,
    link_paths: Mapping[str, Path] | None = None,
) -> tuple[LexicalCandidateTable, dict[str, AlignmentLinks]]:
    """End-to-end: align (or load) train links → three-layer candidate table."""
    train = [s for s in bitext_sentences if s.split_role == SplitRole.TRAIN.value]
    if not train:
        msg = "no train bitext sentences available for lexical candidates"
        raise ValueError(msg)
    _assert_train_only(train)

    if link_paths:
        aligner_links = {
            name: load_compact_alignment_artifact(path, train)
            for name, path in link_paths.items()
        }
    else:
        aligner_links = align_train_sentences(
            train,
            aligners=aligners,
            config=config,
            repo_root=repo_root,
        )

    # Train ParallelUnits for optional Dice annotation (JAK-28 reuse).
    buckets = partition_corpus(corpus, splits)
    train_units = nonempty_positive_units(buckets[SplitRole.TRAIN])

    provenance = {
        "owned_by": "JAK-37",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pilot_corpus_id": corpus.id,
        "split_id": splits.id,
        "greek_source_id": corpus.greek_source_id,
        "sahidic_source_id": corpus.sahidic_source_id,
        "phase2_active_aligners": sorted(aligner_links),
        "adopted_aligner": None,
        "jake_decision": {
            "status": "decided",
            "decision": "keep-both",
            "adopted_aligner": None,
            "phase2_active_aligners": ["eflomal", "fast_align"],
            "decided_by": "Jake",
            "from_issue": "JAK-36",
        },
        "aligner_pins": pinned_versions_payload(),
        "config": (config or AlignerConfig()).as_dict(),
        "coptic_segmentation": "bound-group-v1",
        "greek_tokenization": "whitespace-v1",
        "protocol": (
            "Train-only symmetrized links from phase2_active_aligners; "
            "holdout/dev/omission/Thomas-7 excluded from observations."
        ),
    }
    table = build_lexical_candidate_table(
        aligner_links,
        provenance=provenance,
        train_units=train_units,
    )
    return table, aligner_links


def default_lexicon_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "sources" / "lexicon"


def default_table_path(repo_root: Path) -> Path:
    return default_lexicon_dir(repo_root) / "coptic-greek-lexical-candidates-v1.json"
