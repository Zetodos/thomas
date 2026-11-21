# SPDX-License-Identifier: Apache-2.0
"""``thomas sources`` Typer subcommands."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.ingest.dilley_thomas import (
    EXPECTED_LOGIA,
    INGEST_LEVEL,
    default_corpus_path,
    default_tt_path,
    expected_unit_ids,
    load_thomas_corpus,
    parse_tt,
    validate_thomas_unit_sequence,
    write_thomas_corpus,
)
from zetodos_thomas.ingest.dilley_thomas import (
    SOURCE_ID as DILLEY_SOURCE_ID,
)
from zetodos_thomas.ingest.download import DownloadError, verify_checksum
from zetodos_thomas.ingest.evidence_pilot import (
    EvidencePilotValidationError,
    load_evidence_pilot,
    validate_evidence_pilot_or_raise,
)
from zetodos_thomas.ingest.fragments import (
    FragmentValidationError,
    build_fragment_corpus,
    fragments_corpus_path,
    fragments_tei_dir,
    load_fragment_corpus,
    validate_fragment_corpus_or_raise,
    write_fragment_corpus,
)
from zetodos_thomas.ingest.holdouts import (
    HoldoutValidationError,
    load_omission_fixtures,
    validate_holdout_artifacts,
)
from zetodos_thomas.ingest.manifests import (
    ManifestValidationError,
    load_manifest,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.morphgnt import build_nt_split_v1, pair_horner_morphgnt
from zetodos_thomas.ingest.paths import (
    NT_SPLIT_REL,
    require_repo_root,
    resolve_evidence_pilot_path,
    resolve_fragment_divergences_path,
    resolve_fragments_corpus_path,
    resolve_manifests_dir,
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
    resolve_thomas_path,
)
from zetodos_thomas.ingest.verify_inputs import (
    verify_downloaded_sources,
    verify_fragment_rebuild_inputs,
)
from zetodos_thomas.schemas.holdout import PassageRole
from zetodos_thomas.schemas.source import GATED_DEFAULT_SOURCE_IDS

sources_app = typer.Typer(
    name="sources",
    help="Validate manifests and build annotated corpora offline after setup.",
    no_args_is_help=True,
)


@sources_app.command("validate")
def validate_sources(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (optional; packaged data used if absent)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    manifests_dir: Path | None = typer.Option(
        None,
        "--manifests-dir",
        help="Directory of JSON source manifests",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Validate default manifests, Thomas corpus, fragments, and splits offline."""
    manifests_path = manifests_dir or resolve_manifests_dir(repo_root)
    thomas = resolve_thomas_path(repo_root)
    fragments = resolve_fragments_corpus_path(repo_root)
    evidence_pilot = resolve_evidence_pilot_path(repo_root)

    try:
        manifests = validate_manifests_dir(manifests_path)
    except ManifestValidationError as exc:
        typer.secho(f"Manifest validation failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"OK: {len(manifests)} manifest(s) under {manifests_path}")
    for manifest in manifests:
        typer.echo(f"  - {manifest.id} ({manifest.mode}, {manifest.spdx_or_custom})")

    blocked = sorted(GATED_DEFAULT_SOURCE_IDS & {m.id for m in manifests})
    if blocked:
        typer.secho(
            f"Gated ids present (forbidden on default path): {blocked}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    typer.echo(
        "OK: default path excludes Sahidica / NTAEG / Bohairica "
        f"({len(GATED_DEFAULT_SOURCE_IDS)} gated ids)"
    )

    if thomas.is_file():
        try:
            thomas_corpus = load_thomas_corpus(thomas)
            validate_thomas_unit_sequence(thomas_corpus.units)
        except (OSError, ValueError) as exc:
            typer.secho(f"Thomas corpus validation failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        if thomas_corpus.source_id != DILLEY_SOURCE_ID:
            typer.secho(
                f"Thomas source_id must be {DILLEY_SOURCE_ID}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        morpheme_count = sum(len(u.morphemes) for u in thomas_corpus.units)
        expected = expected_unit_ids()
        typer.echo(
            f"OK: thomas {thomas_corpus.id} - {INGEST_LEVEL} "
            f"({expected[0]}..{expected[-1]}, {EXPECTED_LOGIA} logia + prologue, "
            f"{morpheme_count} morphemes) from {thomas}"
        )
    else:
        typer.echo(f"Note: thomas corpus not found at {thomas} (skipped)")

    if fragments.is_file():
        try:
            frag = validate_fragment_corpus_or_raise(
                load_fragment_corpus(fragments),
                manifests,
            )
        except FragmentValidationError as exc:
            typer.secho(f"Fragment validation failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        line_count = sum(len(tx.lines) for tx in frag.transcriptions)
        word_count = sum(len(tx.words) for tx in frag.transcriptions)
        typer.echo(
            f"OK: fragments {frag.id} - {len(frag.transcriptions)} transcription(s), "
            f"{line_count} lines, {word_count} words, role={frag.corpus_role} "
            f"from {fragments}"
        )
    else:
        typer.echo(f"Note: fragment corpus not found at {fragments} (skipped)")

    if evidence_pilot.is_file() and thomas.is_file() and fragments.is_file():
        try:
            package = validate_evidence_pilot_or_raise(
                load_evidence_pilot(evidence_pilot),
                manifests=manifests,
                thomas_corpus_path=thomas,
                fragments_corpus_path=fragments,
            )
        except EvidencePilotValidationError as exc:
            typer.secho(
                f"Evidence pilot validation failed: {exc}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1) from exc
        typer.echo(
            f"OK: evidence pilot {package.id} - {len(package.sayings)} sayings "
            f"from {evidence_pilot}"
        )

    split_path = resolve_nt_split_path(repo_root)
    omission_path = resolve_omission_fixtures_path(repo_root)
    divergence_path = resolve_fragment_divergences_path(repo_root)
    if split_path.is_file() and omission_path.is_file() and divergence_path.is_file():
        try:
            split, omissions, divergences = validate_holdout_artifacts(
                split_path=split_path,
                omission_path=omission_path,
                divergence_path=divergence_path,
            )
        except HoldoutValidationError as exc:
            typer.secho(f"Holdout validation failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        train_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.TRAIN
        )
        dev_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.DEV
        )
        holdout_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.HOLDOUT
        )
        typer.echo(
            f"OK: split {split.id} (train={train_n}, dev={dev_n}, holdout={holdout_n}, "
            f"omissions={len(omissions.units)}, "
            f"divergences={len(divergences.divergences)})"
        )


@sources_app.command("build")
def build_sources(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Rebuild token-complete Dilley Thomas, word-layered fragments, and NT split."""
    try:
        root = require_repo_root(repo_root)
        manifests_dir = root / "data" / "manifests"
        downloads_dir = root / "data" / "downloads"
        manifests = validate_manifests_dir(manifests_dir)

        # 1. Rebuild Dilley Thomas with full SCRIPTORIUM annotations
        dilley_manifest = load_manifest(manifests_dir / "dilley-thomas.json")
        tt_path = default_tt_path(root)
        if not tt_path.is_file():
            msg = f"missing Dilley TT at {tt_path}; run `thomas setup` first"
            raise FileNotFoundError(msg)
        verify_checksum(tt_path, dilley_manifest.checksum)
        thomas_corpus = parse_tt(tt_path, source_id=DILLEY_SOURCE_ID)
        thomas_out = default_corpus_path(root)
        write_thomas_corpus(thomas_corpus, thomas_out)
        morphemes_n = sum(len(u.morphemes) for u in thomas_corpus.units)
        typer.echo(
            f"Wrote {len(thomas_corpus.units)} Thomas units ({morphemes_n} morphemes) "
            f"to {thomas_out}"
        )

        # 2. Rebuild DCLP fragments with FragmentWord layer
        tei_dir = fragments_tei_dir(root)
        verify_fragment_rebuild_inputs(manifests_dir=manifests_dir, tei_dir=tei_dir)
        frag_corpus = build_fragment_corpus(tei_dir)
        validate_fragment_corpus_or_raise(frag_corpus, manifests)
        frag_out = fragments_corpus_path(root)
        write_fragment_corpus(frag_corpus, frag_out)
        words_n = sum(len(tx.words) for tx in frag_corpus.transcriptions)
        typer.echo(
            f"Wrote {len(frag_corpus.transcriptions)} fragment transcriptions "
            f"({words_n} words) to {frag_out}"
        )

        # 3. Verify downloaded NT/OT/CCL inputs and build full-NT leak-free split v1
        verify_downloaded_sources(
            manifests_dir=manifests_dir, downloads_dir=downloads_dir
        )
        paired = pair_horner_morphgnt(
            sword_root=downloads_dir / "sword",
            morphgnt_dir=downloads_dir / "morphgnt-sblgnt",
        )
        omissions = load_omission_fixtures(resolve_omission_fixtures_path(root))
        legacy_split = root / "data/training/splits/horner-vaticanus-nt-split-v3.json"
        seed_split = legacy_split if legacy_split.is_file() else (root / NT_SPLIT_REL)
        split = build_nt_split_v1(
            paired_verses=paired,
            legacy_split_path=seed_split,
            omission_unit_ids={u.unit_id for u in omissions.units},
        )
        split_out = root / NT_SPLIT_REL
        split_out.parent.mkdir(parents=True, exist_ok=True)
        split_out.write_text(split.model_dump_json() + "\n", encoding="utf-8")
        validate_holdout_artifacts(
            split_path=split_out,
            omission_path=resolve_omission_fixtures_path(root),
            divergence_path=resolve_fragment_divergences_path(root),
            greek_texts={g.unit_id: g.normalized_text for _c, g in paired},
            sahidic_texts={c.unit_id: c.normalized_text for c, _g in paired},
        )
        train_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.TRAIN
        )
        dev_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.DEV
        )
        hold_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.HOLDOUT
        )
        typer.echo(
            f"Wrote {split.id} (train={train_n}, dev={dev_n}, holdout={hold_n}) "
            f"to {split_out}"
        )
    except (
        DownloadError,
        FragmentValidationError,
        HoldoutValidationError,
        ManifestValidationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        typer.secho(f"sources build failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
