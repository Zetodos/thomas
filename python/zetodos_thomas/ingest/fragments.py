# SPDX-License-Identifier: Apache-2.0
"""Build, load, and validate Greek Thomas fragment corpora (JAK-31)."""

from __future__ import annotations

import json
from pathlib import Path

from zetodos_thomas.ingest.dclp_fragments import (
    DCLP_THOMAS_FRAGMENTS,
    EXCLUDED_NON_THOMAS_WITNESSES,
    build_transcription,
)
from zetodos_thomas.schemas.fragment import (
    CorpusRole,
    EvidenceType,
    FragmentCorpus,
)
from zetodos_thomas.schemas.source import SourceManifest

DEFAULT_FRAGMENT_CORPUS_ID = "greek-thomas-fragments-v1"
DEFAULT_FRAGMENT_CORPUS_FILENAME = "greek-thomas-fragments.json"


class FragmentValidationError(ValueError):
    """Fragment corpus failed structural or rights/policy checks."""


def fragments_tei_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "sources" / "fragments" / "dclp"


def fragments_corpus_path(repo_root: Path) -> Path:
    return (
        repo_root / "data" / "sources" / "fragments" / DEFAULT_FRAGMENT_CORPUS_FILENAME
    )


def build_fragment_corpus(tei_dir: Path) -> FragmentCorpus:
    """Parse committed DCLP TEI files into a layered fragment corpus."""
    transcriptions = []
    for meta in DCLP_THOMAS_FRAGMENTS:
        path = tei_dir / meta["filename"]
        if not path.is_file():
            msg = f"missing DCLP TEI: {path}"
            raise FileNotFoundError(msg)
        transcriptions.append(
            build_transcription(
                path,
                source_id=meta["source_id"],
                transcription_id=meta["transcription_id"],
                manuscript_id=meta["manuscript_id"],
                tm_id=meta["tm_id"],
                logia_coverage=meta["logia_coverage"],
            )
        )
    return FragmentCorpus(
        id=DEFAULT_FRAGMENT_CORPUS_ID,
        title=(
            "Greek Gospel of Thomas fragments — P.Oxy. 1, 654, 655 "
            "(DCLP / papyri.info, CC BY 3.0)"
        ),
        evidence_type=EvidenceType.G,
        corpus_role=CorpusRole.EVALUATION_WITNESS,
        transcriptions=transcriptions,
        excluded_witnesses=list(EXCLUDED_NON_THOMAS_WITNESSES),
        notes=(
            "Evidence type G (direct Greek Thomas manuscripts). Labeled "
            "evaluation-witness so JAK-28 Greek–Sahidic alignment training "
            "must not consume these texts as training pairs without an "
            "explicit holdout waiver. P.Oxy. 5575 is excluded (independent "
            "parallel, not a Thomas manuscript). Never-ship: Layton/Attridge "
            "continuous critical Greek; manuscript images."
        ),
    )


def write_fragment_corpus(corpus: FragmentCorpus, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(corpus.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_fragment_corpus(path: Path) -> FragmentCorpus:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FragmentCorpus.model_validate(data)


def validate_fragment_corpus(
    corpus: FragmentCorpus,
    manifests: list[SourceManifest],
) -> list[str]:
    """Return problems: missing licenses, dangling IDs, wrong labels, etc."""
    problems: list[str] = []
    by_id = {m.id: m for m in manifests}

    if corpus.evidence_type is not EvidenceType.G:
        problems.append(
            f"fragment corpus evidence_type must be G (found {corpus.evidence_type})"
        )
    if corpus.corpus_role is not CorpusRole.EVALUATION_WITNESS:
        problems.append(
            "fragment corpus corpus_role must be evaluation-witness "
            f"(found {corpus.corpus_role})"
        )

    if not corpus.transcriptions:
        problems.append("fragment corpus has no transcriptions")

    seen_tx: set[str] = set()
    for tx in corpus.transcriptions:
        if tx.id in seen_tx:
            problems.append(f"duplicate transcription id: {tx.id}")
        seen_tx.add(tx.id)

        if tx.evidence_type is not EvidenceType.G:
            problems.append(f"{tx.id}: evidence_type must be G")
        if tx.corpus_role is not CorpusRole.EVALUATION_WITNESS:
            problems.append(f"{tx.id}: corpus_role must be evaluation-witness")

        manifest = by_id.get(tx.source_id)
        if manifest is None:
            problems.append(f"{tx.id}: dangling source_id {tx.source_id!r}")
            continue

        if not manifest.spdx_or_custom:
            problems.append(f"{manifest.id}: missing spdx_or_custom / license")
        if not manifest.checksum:
            problems.append(f"{manifest.id}: missing checksum")
        if not manifest.manuscript_id:
            problems.append(f"{manifest.id}: missing manuscript_id")
        if manifest.manuscript_id and manifest.manuscript_id != tx.manuscript_id:
            problems.append(
                f"{tx.id}: manuscript_id {tx.manuscript_id!r} != manifest "
                f"{manifest.manuscript_id!r}"
            )
        if not tx.lines:
            problems.append(f"{tx.id}: no lines")
        else:
            kinds = {tok.kind.value for line in tx.lines for tok in line.tokens}
            if "surviving" not in kinds and "restoration" not in kinds:
                problems.append(f"{tx.id}: no surviving/restoration text layers")

        # Guard against mislabeling known non-Thomas parallels.
        mid = (tx.manuscript_id or "").lower().replace(" ", "")
        if "5575" in mid:
            problems.append(
                f"{tx.id}: P.Oxy. 5575 must not be labeled a Thomas manuscript"
            )

    expected_sources = {m["source_id"] for m in DCLP_THOMAS_FRAGMENTS}
    for source_id in expected_sources:
        if source_id not in by_id:
            problems.append(f"missing fragment source manifest: {source_id}")
        else:
            m = by_id[source_id]
            if m.evidence_type != "G":
                problems.append(f"{source_id}: evidence_type must be G")
            if m.corpus_role != CorpusRole.EVALUATION_WITNESS.value:
                problems.append(f"{source_id}: corpus_role must be evaluation-witness")

    return problems


def validate_fragment_corpus_or_raise(
    corpus: FragmentCorpus,
    manifests: list[SourceManifest],
) -> FragmentCorpus:
    problems = validate_fragment_corpus(corpus, manifests)
    if problems:
        raise FragmentValidationError("; ".join(problems))
    return corpus
