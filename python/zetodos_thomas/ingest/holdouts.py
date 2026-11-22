# SPDX-License-Identifier: Apache-2.0
"""Load and cross-check benchmark holdout, omission, and divergence artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

from zetodos_thomas.schemas.holdout import (
    FragmentDivergenceDocument,
    NtCorpusSplit,
    OmissionFixtureSet,
    PassageRole,
)

_GREEK_TOKEN_RE = re.compile(r"\S+")
_UNIT_ID_RE = re.compile(r"^([a-z0-9]+)\.(\d+)\.(\d+)$")
GREEK_NGRAM_ORDER = 5
_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
_FORBIDDEN_TRAIN_UNIT_PREFIXES = ("thomas.", "poxy", "p.oxy", "frag")
_FORBIDDEN_TRAIN_SOURCE_MARKERS = ("thomas", "poxy", "dclp", "fragment")

# Canonical NT verses parallel to the 20 P.Oxy. 1 / 654 / 655 evaluated logia.
# Excluded from BOTH model-v1 and model-v2 training (aligner, inflector, idioms,
# n-gram LM, BPE tokenizer, reverse/forward neural seq2seq, and neural LM) so
# P.Oxy. evaluation cannot be inflated by canonical Synoptic/Johannine parallels.
SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS: frozenset[str] = frozenset(
    {
        # Logion 1 (taste death): John 8:51-52; Matt 16:28; Mark 9:1; Luke 9:27; Heb 2:9
        "john.8.51",
        "john.8.52",
        "matthew.16.28",
        "mark.9.1",
        "luke.9.27",
        "hebrews.2.9",
        # Logion 2 (seek and find): Matt 7:7-8; Luke 11:9-10
        "matthew.7.7",
        "matthew.7.8",
        "luke.11.9",
        "luke.11.10",
        # Logion 3 (kingdom within / outside): Luke 17:20-21
        "luke.17.20",
        "luke.17.21",
        # Logion 4 (first last / last first): Matt 19:30, 20:16; Mark 10:31; Luke 13:30
        "matthew.19.30",
        "matthew.20.16",
        "mark.10.31",
        "luke.13.30",
        # Logia 5-6 (nothing hidden; fast/pray/alms; Golden Rule): Mark 4:22; Luke 8:17;
        # Matt 6:2-3,5,16; Matt 7:12 (Matt 10:26, Luke 12:2 holdout; Luke 6:31 dev)
        "mark.4.22",
        "luke.8.17",
        "matthew.6.2",
        "matthew.6.3",
        "matthew.6.5",
        "matthew.6.16",
        "matthew.7.12",
        # Logion 26 (speck/beam in brother's eye): Matt 7:3-5 (Luke 6:41-42 already dev)
        "matthew.7.3",
        "matthew.7.4",
        "matthew.7.5",
        # Logion 30 (where two or three are gathered): Matt 18:20
        "matthew.18.20",
        # Logion 77 (light of the world opening): John 8:12; John 9:5
        "john.8.12",
        "john.9.5",
        # Logion 31 (prophet/physician): Matt 13:57; Mark 6:4; Luke 4:23-24; John 4:44
        "matthew.13.57",
        "mark.6.4",
        "luke.4.23",
        "luke.4.24",
        "john.4.44",
        # Logia 32-33 (city on hill; lamp): Matt 5:14-15; Mark 4:21; Luke 8:16, 11:33
        # (Matt 10:27 & Luke 12:3 are already in holdout)
        "matthew.5.14",
        "matthew.5.15",
        "mark.4.21",
        "luke.8.16",
        "luke.11.33",
        # Logion 36 (anxious / clothing / lilies): Matt 6:25-31; Luke 12:22-29
        "matthew.6.25",
        "matthew.6.26",
        "matthew.6.27",
        "matthew.6.28",
        "matthew.6.29",
        "matthew.6.30",
        "matthew.6.31",
        "luke.12.22",
        "luke.12.23",
        "luke.12.24",
        "luke.12.25",
        "luke.12.26",
        "luke.12.27",
        "luke.12.28",
        "luke.12.29",
        # Logion 38 (desired to hear; seek me):
        # Matt 13:17; Luke 10:24, 17:22; John 7:34, 7:36, 8:21
        "matthew.13.17",
        "luke.10.24",
        "luke.17.22",
        "john.7.34",
        "john.7.36",
        "john.8.21",
        # Logion 39 (keys of knowledge; serpents/doves): Matt 23:13, 10:16; Luke 11:52
        "matthew.23.13",
        "luke.11.52",
        "matthew.10.16",
    }
)

# P.Oxy. evaluated Thomas logia whose scored spans have NO close Synoptic parallel
NON_SYNOPTIC_POXY_THOMAS_UNITS: frozenset[str] = frozenset(
    {
        "thomas.0",
        "thomas.7",
        "thomas.27",
        "thomas.28",
        "thomas.37",
        "thomas.77",
    }
)


class HoldoutValidationError(ValueError):
    """Invalid holdout / split / omission / divergence artifact."""


SPLIT_REL = Path("data/training/splits/nt-split-v1.json")
OMISSION_REL = Path("data/evidence/omissions/john-8-pa-omission-fixture-v3.json")
DIVERGENCE_REL = Path(
    "data/evidence/divergences/greek-thomas-fragment-divergences-v1.json"
)


def load_nt_split(path: Path) -> NtCorpusSplit:
    """Load and schema-validate the NT train/dev/holdout split."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NtCorpusSplit.model_validate(payload)


def load_omission_fixtures(path: Path) -> OmissionFixtureSet:
    """Load and schema-validate the omission-fixture catalog."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OmissionFixtureSet.model_validate(payload)


def load_fragment_divergences(path: Path) -> FragmentDivergenceDocument:
    """Load and schema-validate Greek fragment divergence documentation."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FragmentDivergenceDocument.model_validate(payload)


def _unit_ids_by_role(split: NtCorpusSplit, role: PassageRole) -> set[str]:
    ids: set[str] = set()
    for passage in split.passages:
        if passage.role is role:
            ids.update(passage.unit_ids)
    return ids


def _assert_passage_unit_ids_within_range(split: NtCorpusSplit) -> None:
    """Verify every PassageRef unit_id matches its declared book and verse range."""
    for passage in split.passages:
        start = (passage.chapter_from, passage.verse_from)
        end = (passage.chapter_to, passage.verse_to)
        for uid in passage.unit_ids:
            match = _UNIT_ID_RE.match(uid)
            if match is None:
                msg = f"malformed unit_id {uid!r} in passage {passage.book}"
                raise HoldoutValidationError(msg)
            book, ch_s, v_s = match.groups()
            pos = (int(ch_s), int(v_s))
            if book != passage.book or not (start <= pos <= end):
                msg = (
                    f"unit_id {uid!r} outside passage role={passage.role.value} "
                    f"book={passage.book} {passage.chapter_from}:{passage.verse_from}-"
                    f"{passage.chapter_to}:{passage.verse_to}"
                )
                raise HoldoutValidationError(msg)


def greek_token_ngrams(text: str, order: int = GREEK_NGRAM_ORDER) -> set[str]:
    """Return whitespace token n-grams from normalized Greek text."""
    if order < 1:
        msg = "ngram order must be >= 1"
        raise ValueError(msg)
    tokens = _GREEK_TOKEN_RE.findall(text)
    if len(tokens) < order:
        return set()
    return {" ".join(tokens[i : i + order]) for i in range(len(tokens) - order + 1)}


def cross_split_greek_ngram_leaks(
    split: NtCorpusSplit,
    greek_texts: Mapping[str, str],
    *,
    order: int = GREEK_NGRAM_ORDER,
    left: PassageRole = PassageRole.TRAIN,
    right: PassageRole = PassageRole.HOLDOUT,
) -> dict[str, tuple[set[str], set[str]]]:
    """Map shared Greek n-grams between two split roles to their unit_ids."""
    left_ids = _unit_ids_by_role(split, left)
    right_ids = _unit_ids_by_role(split, right)
    left_ngrams: dict[str, set[str]] = defaultdict(set)
    right_ngrams: dict[str, set[str]] = defaultdict(set)
    for unit_ids, bucket in ((left_ids, left_ngrams), (right_ids, right_ngrams)):
        for uid in unit_ids:
            text = greek_texts.get(uid, "")
            if not text:
                continue
            for ng in greek_token_ngrams(text, order=order):
                bucket[ng].add(uid)
    shared = set(left_ngrams) & set(right_ngrams)
    return {ng: (left_ngrams[ng], right_ngrams[ng]) for ng in sorted(shared)}


def assert_no_train_holdout_greek_ngram_leakage(
    split: NtCorpusSplit,
    greek_texts: Mapping[str, str],
    *,
    order: int = GREEK_NGRAM_ORDER,
) -> None:
    """Reject splits where train and holdout share exact Greek token n-grams."""
    leaks = cross_split_greek_ngram_leaks(split, greek_texts, order=order)
    if not leaks:
        return
    examples: list[str] = []
    for ngram, (train_uids, hold_uids) in list(leaks.items())[:8]:
        examples.append(
            f"{ngram!r} train={sorted(train_uids)} holdout={sorted(hold_uids)}"
        )
    msg = (
        f"train/holdout share {len(leaks)} exact Greek {order}-token sequence(s); "
        "exclude overlapping Synoptic verses from train. "
        "Examples: " + "; ".join(examples)
    )
    raise HoldoutValidationError(msg)


def crosscheck_against_units(
    split: NtCorpusSplit,
    omissions: OmissionFixtureSet,
    greek_texts: Mapping[str, str] | None = None,
    sahidic_texts: Mapping[str, str] | None = None,
) -> None:
    """Verify split/omission disjointness, witness exclusions, and 0 5-gram leaks."""
    train_ids = _unit_ids_by_role(split, PassageRole.TRAIN)
    holdout_ids = _unit_ids_by_role(split, PassageRole.HOLDOUT)
    dev_ids = _unit_ids_by_role(split, PassageRole.DEV)
    omission_ids = {unit.unit_id for unit in omissions.units}

    _assert_passage_unit_ids_within_range(split)

    if train_ids & holdout_ids:
        msg = f"train/holdout overlap: {sorted(train_ids & holdout_ids)}"
        raise HoldoutValidationError(msg)
    if train_ids & dev_ids:
        msg = f"train/dev overlap: {sorted(train_ids & dev_ids)}"
        raise HoldoutValidationError(msg)
    if dev_ids & holdout_ids:
        msg = f"dev/holdout overlap: {sorted(dev_ids & holdout_ids)}"
        raise HoldoutValidationError(msg)
    if train_ids & omission_ids:
        msg = f"train overlaps omission fixtures: {sorted(train_ids & omission_ids)}"
        raise HoldoutValidationError(msg)
    if holdout_ids & omission_ids:
        msg = (
            f"holdout overlaps omission fixtures: {sorted(holdout_ids & omission_ids)}"
        )
        raise HoldoutValidationError(msg)
    if dev_ids & omission_ids:
        msg = f"dev overlaps omission fixtures: {sorted(dev_ids & omission_ids)}"
        raise HoldoutValidationError(msg)

    required_witnesses = {"poxy-1-dclp", "poxy-654-dclp", "poxy-655-dclp"}
    present = {ex.witness_id for ex in split.evaluation_witness_exclusions}
    missing_witnesses = required_witnesses - present
    if missing_witnesses:
        msg = (
            "evaluation_witness_exclusions missing Greek Thomas fragments: "
            f"{sorted(missing_witnesses)}"
        )
        raise HoldoutValidationError(msg)
    for exclusion in split.evaluation_witness_exclusions:
        if exclusion.corpus_role != "evaluation-witness":
            msg = f"{exclusion.witness_id} must remain evaluation-witness"
            raise HoldoutValidationError(msg)

    assigned = train_ids | dev_ids | holdout_ids
    if greek_texts is not None:
        unknown = assigned - set(greek_texts)
        if unknown:
            msg = f"split references unknown Greek unit_ids: {sorted(unknown)[:10]}"
            raise HoldoutValidationError(msg)
        assert_no_train_holdout_greek_ngram_leakage(split, greek_texts)

    if sahidic_texts is not None:
        empty_sahidic = [
            uid for uid in assigned if not sahidic_texts.get(uid, "").strip()
        ]
        if empty_sahidic:
            msg = (
                f"assigned units have empty Sahidic text: {sorted(empty_sahidic)[:10]}"
            )
            raise HoldoutValidationError(msg)


def validate_holdout_artifacts(
    *,
    split_path: Path,
    omission_path: Path,
    divergence_path: Path,
    greek_texts: Mapping[str, str] | None = None,
    sahidic_texts: Mapping[str, str] | None = None,
) -> tuple[NtCorpusSplit, OmissionFixtureSet, FragmentDivergenceDocument]:
    """Validate split, omission, and divergence artifacts."""
    try:
        split = load_nt_split(split_path)
        omissions = load_omission_fixtures(omission_path)
        divergences = load_fragment_divergences(divergence_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HoldoutValidationError(str(exc)) from exc

    divergence_ids = {item.id for item in divergences.divergences}
    if "poxy1-coptic-30-77-division" not in divergence_ids:
        msg = "missing required divergence id poxy1-coptic-30-77-division"
        raise HoldoutValidationError(msg)
    if "P.Oxy. 5575" not in divergences.excluded_witnesses:
        msg = "divergence catalog must exclude P.Oxy. 5575 as non-Thomas"
        raise HoldoutValidationError(msg)

    crosscheck_against_units(
        split,
        omissions,
        greek_texts=greek_texts,
        sahidic_texts=sahidic_texts,
    )
    return split, omissions, divergences


def clean_nt_train_ids(
    split: NtCorpusSplit, *, exclude_synoptic_thomas: bool = True
) -> set[str]:
    """Return NT train unit IDs with Synoptic parallels of P.Oxy. logia excluded."""
    train_ids = _unit_ids_by_role(split, PassageRole.TRAIN)
    if exclude_synoptic_thomas:
        return train_ids - SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
    return train_ids


def all_excluded_nt_ids(
    split: NtCorpusSplit,
    *,
    include_dev: bool = True,
    exclude_synoptic_thomas: bool = True,
) -> set[str]:
    """Return all NT unit IDs forbidden from training (holdout + dev + Synoptic)."""
    excluded = _unit_ids_by_role(split, PassageRole.HOLDOUT)
    if include_dev:
        excluded |= _unit_ids_by_role(split, PassageRole.DEV)
    if exclude_synoptic_thomas:
        excluded |= SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
    return excluded


def assert_clean_training_unit(
    unit_id: str,
    *,
    source_id: str = "",
    excluded_ids: Collection[str] = (),
) -> None:
    """Refuse Thomas, P.Oxy. fragment, or excluded split units during training."""
    uid_low = unit_id.strip().lower()
    if any(uid_low.startswith(pref) for pref in _FORBIDDEN_TRAIN_UNIT_PREFIXES):
        msg = f"forbidden evaluation unit_id in training data: {unit_id!r}"
        raise HoldoutValidationError(msg)
    src_low = source_id.strip().lower()
    if src_low and any(m in src_low for m in _FORBIDDEN_TRAIN_SOURCE_MARKERS):
        msg = (
            f"forbidden evaluation source_id in training data: "
            f"{source_id!r} (unit_id={unit_id!r})"
        )
        raise HoldoutValidationError(msg)
    if unit_id in excluded_ids:
        msg = f"excluded holdout/dev/synoptic unit_id in training data: {unit_id!r}"
        raise HoldoutValidationError(msg)


def assert_clean_training_units(
    units: Iterable[object],
    *,
    excluded_ids: Collection[str] = (),
) -> None:
    """Validate an iterable of units (or unit_id strings) contains zero leakage."""
    excl = set(excluded_ids)
    for item in units:
        if isinstance(item, str):
            assert_clean_training_unit(item, excluded_ids=excl)
        else:
            uid = str(getattr(item, "unit_id", ""))
            src = str(getattr(item, "source_id", ""))
            assert_clean_training_unit(uid, source_id=src, excluded_ids=excl)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_or_lfs_oid(path: Path) -> str:
    """Return content SHA-256, parsing Git LFS pointer ``oid sha256:`` if unsmudged."""
    head = path.read_bytes()[:256]
    if head.startswith(_LFS_POINTER_HEADER):
        for line in head.decode("utf-8", errors="ignore").splitlines():
            if line.startswith("oid sha256:"):
                return line.split("oid sha256:", 1)[1].strip()
    return sha256_file(path)


def load_manifest_sha256(manifest_path: Path) -> str:
    """Read pinned ``sha256:<hex>`` checksum from a ``data/manifests/*.json`` file."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = str(data.get("checksum", "")).strip()
    if raw.startswith("sha256:"):
        return raw.removeprefix("sha256:").lower()
    msg = f"manifest {manifest_path} missing sha256 checksum: {raw!r}"
    raise HoldoutValidationError(msg)


def _hash_unit_ids(unit_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(unit_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_training_provenance(
    *,
    out_dir: Path,
    model_id: str,
    split_path: Path,
    ccl_xml_path: Path,
    nt_train_ids: Iterable[str],
    ot_train_ids: Iterable[str],
    excluded_holdout_ids: Collection[str],
    excluded_dev_ids: Collection[str],
    excluded_synoptic_ids: Collection[str],
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write cryptographic ``training-provenance.json`` auditing corpora & weights."""
    nt_sorted = sorted(set(nt_train_ids))
    ot_sorted = sorted(set(ot_train_ids))
    syn_sorted = sorted(set(excluded_synoptic_ids))
    assert_clean_training_units(
        nt_sorted,
        excluded_ids=set(excluded_holdout_ids)
        | set(excluded_dev_ids)
        | set(syn_sorted),
    )
    assert_clean_training_units(ot_sorted)

    prov_path = out_dir / "training-provenance.json"
    artifact_hashes: dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or p.name == prov_path.name:
            continue
        if p.suffix in {".json", ".safetensors", ".model", ".vocab"}:
            artifact_hashes[p.name] = sha256_or_lfs_oid(p)

    payload: dict[str, object] = {
        "schema_version": "1",
        "model_id": model_id,
        "nt_split_sha256": sha256_file(split_path),
        "ccl_manifest_id": "ccl-ddglc-tei",
        "ccl_xml_sha256": sha256_file(ccl_xml_path),
        "nt_train_unit_count": len(nt_sorted),
        "nt_train_unit_ids_sha256": _hash_unit_ids(nt_sorted),
        "ot_train_unit_count": len(ot_sorted),
        "ot_train_unit_ids_sha256": _hash_unit_ids(ot_sorted),
        "excluded_holdout_count": len(set(excluded_holdout_ids)),
        "excluded_dev_count": len(set(excluded_dev_ids)),
        "excluded_synoptic_thomas_count": len(syn_sorted),
        "excluded_synoptic_thomas_unit_ids": syn_sorted,
        "zero_curated_thomas_idioms": True,
        "zero_ccl_manual_fallbacks": True,
        "artifact_sha256": artifact_hashes,
    }
    if extra:
        payload["extra"] = dict(extra)
    prov_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return prov_path
