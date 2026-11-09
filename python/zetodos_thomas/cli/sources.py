# SPDX-License-Identifier: Apache-2.0
"""``thomas sources`` Typer subcommands."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.align.splits import DEFAULT_SPLITS, load_splits
from zetodos_thomas.align.validate import validate_pilot_corpus
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
from zetodos_thomas.ingest.download import (
    DownloadError,
    download_dilley_thomas,
    download_pilot_sources,
    verify_checksum,
)
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
    validate_holdout_artifacts,
)
from zetodos_thomas.ingest.manifests import (
    ManifestValidationError,
    load_manifest,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.ntvmr_vaticanus import SOURCE_ID as VATICANUS_SOURCE_ID
from zetodos_thomas.ingest.paths import (
    require_repo_root,
    resolve_evidence_pilot_path,
    resolve_fragment_divergences_path,
    resolve_fragments_corpus_path,
    resolve_manifests_dir,
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
    resolve_pilot_path,
    resolve_thomas_path,
)
from zetodos_thomas.ingest.pilot import (
    build_pilot_corpus,
    load_pilot_corpus,
    write_pilot_corpus,
)
from zetodos_thomas.ingest.verify_inputs import (
    verify_fragment_rebuild_inputs,
    verify_pilot_rebuild_inputs,
)
from zetodos_thomas.schemas.holdout import PassageRole
from zetodos_thomas.schemas.source import GATED_DEFAULT_SOURCE_IDS

sources_app = typer.Typer(
    name="sources",
    help="Validate manifests and manage redistributable source imports.",
    no_args_is_help=True,
)


@sources_app.command("validate")
def validate_sources(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (optional; packaged manifests used if absent)",
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
    pilot_path: Path | None = typer.Option(
        None,
        "--pilot",
        help="Optional pilot corpus JSON to validate (omit file to skip)",
        exists=False,
        dir_okay=False,
    ),
    thomas_path: Path | None = typer.Option(
        None,
        "--thomas",
        help="Optional Dilley Thomas corpus JSON to validate (omit file to skip)",
        exists=False,
        dir_okay=False,
    ),
    fragments_path: Path | None = typer.Option(
        None,
        "--fragments",
        help="Optional Greek Thomas fragment corpus JSON (omit file to skip)",
        exists=False,
        dir_okay=False,
    ),
    evidence_pilot_path: Path | None = typer.Option(
        None,
        "--evidence-pilot",
        help="Optional Phase 1 Thomas 7/17/30/77 evidence pilot JSON",
        exists=False,
        dir_okay=False,
    ),
) -> None:
    """Validate default manifests (and optional corpora) without network access.

    Manifests/corpora resolve from the checkout when present; otherwise from
    packaged wheel resources under ``zetodos_thomas.bundled``.
    """
    manifests_path = manifests_dir or resolve_manifests_dir(repo_root)
    pilot = pilot_path or resolve_pilot_path(repo_root)
    thomas = thomas_path or resolve_thomas_path(repo_root)
    fragments = fragments_path or resolve_fragments_corpus_path(repo_root)
    evidence_pilot = evidence_pilot_path or resolve_evidence_pilot_path(repo_root)

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

    if pilot.is_file():
        corpus = load_pilot_corpus(pilot)
        empty_sahidic = sum(
            1 for unit in corpus.units if not unit.sahidic.original_text
        )
        empty_greek = sum(1 for unit in corpus.units if not unit.greek.original_text)
        typer.echo(
            f"OK: pilot {corpus.id} - {len(corpus.units)} units "
            f"({empty_sahidic} empty Sahidic, {empty_greek} empty Greek) "
            f"from {pilot}"
        )
        if corpus.greek_source_id != VATICANUS_SOURCE_ID:
            typer.secho(
                f"Pilot greek_source_id must be {VATICANUS_SOURCE_ID} (JAK-25)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        if corpus.sahidic_source_id != "copsahhorner":
            typer.secho(
                "Pilot sahidic_source_id must be copsahhorner (JAK-14)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        # Structural / normalization / omission checks (JAK-28).
        splits_path = pilot.parent / "splits.json"
        splits = load_splits(splits_path) if splits_path.is_file() else DEFAULT_SPLITS
        report = validate_pilot_corpus(corpus, splits=splits)
        for issue in report.issues:
            if issue.severity == "error":
                typer.secho(
                    f"Pilot validation error [{issue.code}]: {issue.message}",
                    fg=typer.colors.RED,
                )
        if not report.ok:
            raise typer.Exit(code=1)
        typer.echo(
            "OK: pilot structural validation "
            f"({report.nonempty_positive} nonempty positive, "
            f"{report.omission_units} omission)"
        )
    else:
        typer.echo(f"Note: pilot file not found at {pilot} (skipped)")

    if thomas.is_file():
        try:
            thomas_corpus = load_thomas_corpus(thomas)
            validate_thomas_unit_sequence(thomas_corpus.units)
        except (OSError, ValueError) as exc:
            typer.secho(f"Thomas corpus validation failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        if thomas_corpus.source_id != DILLEY_SOURCE_ID:
            typer.secho(
                f"Thomas source_id must be {DILLEY_SOURCE_ID} (JAK-30)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        if INGEST_LEVEL not in thomas_corpus.notes or (
            "not alignment-ready" not in thomas_corpus.notes
        ):
            typer.secho(
                "Thomas corpus notes must label unit-level ingest as "
                "not alignment-ready (JAK-30)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        if thomas_corpus.tokens or thomas_corpus.spans:
            typer.secho(
                "unit-level Thomas corpus must leave tokens[]/spans[] empty",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        empty_orig = sum(1 for u in thomas_corpus.units if not u.original_text)
        empty_norm = sum(1 for u in thomas_corpus.units if not u.normalized_text)
        if empty_orig or empty_norm:
            typer.secho(
                "Thomas corpus has empty original/normalized layers "
                f"(orig={empty_orig}, norm={empty_norm})",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        expected = expected_unit_ids()
        typer.echo(
            f"OK: thomas {thomas_corpus.id} - unit-level ingest "
            f"({expected[0]}..{expected[-1]}, {EXPECTED_LOGIA} logia + prologue); "
            "not token-complete / not alignment-ready "
            f"(tokens={len(thomas_corpus.tokens)}, spans={len(thomas_corpus.spans)}) "
            f"from {thomas}"
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
        typer.echo(
            f"OK: fragments {frag.id} - {len(frag.transcriptions)} transcription(s), "
            f"{line_count} lines, role={frag.corpus_role} from {fragments}"
        )
    else:
        typer.echo(f"Note: fragment corpus not found at {fragments} (skipped)")

    if evidence_pilot.is_file():
        if not thomas.is_file() or not fragments.is_file():
            typer.secho(
                "Evidence pilot requires Dilley Thomas + fragment corpora on disk",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
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
            f"OK: evidence pilot {package.id} - {len(package.sayings)} sayings, "
            f"{len(package.witness_alignments)} witness alignment(s) "
            f"from {evidence_pilot}"
        )
    else:
        typer.echo(f"Note: evidence pilot not found at {evidence_pilot} (skipped)")

    split_path = resolve_nt_split_path(repo_root)
    omission_path = resolve_omission_fixtures_path(repo_root)
    divergence_path = resolve_fragment_divergences_path(repo_root)
    if split_path.is_file() and omission_path.is_file() and divergence_path.is_file():
        pilot_for_holdouts = None
        if pilot.is_file():
            pilot_for_holdouts = load_pilot_corpus(pilot)
        try:
            split, omissions, divergences = validate_holdout_artifacts(
                split_path=split_path,
                omission_path=omission_path,
                divergence_path=divergence_path,
                pilot=pilot_for_holdouts,
            )
        except HoldoutValidationError as exc:
            typer.secho(f"Holdout validation failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        train_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.TRAIN
        )
        holdout_n = sum(
            len(p.unit_ids) for p in split.passages if p.role is PassageRole.HOLDOUT
        )
        typer.echo(
            f"OK: holdouts {split.id} "
            f"(train={train_n}, holdout={holdout_n}, "
            f"omissions={len(omissions.units)}, "
            f"divergences={len(divergences.divergences)}, "
            f"eval-exclusions={len(split.evaluation_witness_exclusions)})"
        )
    else:
        typer.echo("Note: JAK-33 holdout artifacts not found (skipped)")


@sources_app.command("download")
def download_sources(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect from cwd)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even when checksum-verified files already exist",
    ),
) -> None:
    """Download CopSahHorner + NTVMR Vaticanus B* TEI + Dilley Thomas TT.

    Writes only under the checkout (never into the package install). Requires
    network. Throttles NTVMR pulls. Does **not** download manuscript images.
    After this, optionally run ``build-pilot`` / ``build-thomas``.
    """
    try:
        root = require_repo_root(repo_root)
        sword_root, tei_dir, dilley_tt = download_pilot_sources(root, force=force)
    except (DownloadError, FileNotFoundError, OSError) as exc:
        typer.secho(f"Download failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK: CopSahHorner SWORD tree at {sword_root}")
    typer.echo(f"OK: NTVMR Vaticanus B* TEI at {tei_dir}")
    typer.echo(f"OK: Dilley Thomas TT at {dilley_tt}")
    typer.echo("Next (optional): uv run thomas sources build-pilot")
    typer.echo("Next (optional): uv run thomas sources build-thomas")


@sources_app.command("build-pilot")
def build_pilot(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect from cwd)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    sword_root: Path | None = typer.Option(
        None,
        "--sword-root",
        help="Local SWORD tree containing CopSahHorner (under data/downloads/)",
        exists=True,
        file_okay=False,
    ),
    vaticanus_tei_dir: Path | None = typer.Option(
        None,
        "--vaticanus-tei-dir",
        help="Directory of NTVMR Vaticanus teiraw XML files",
        exists=True,
        file_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination JSON for the redistributable pilot excerpt",
    ),
) -> None:
    """Build the pilot excerpt from local downloads after checksum verification."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(f"build-pilot requires a checkout: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    manifests_dir = root / "data" / "manifests"
    downloads_dir = root / "data" / "downloads"
    sword = sword_root or downloads_dir / "sword"
    tei_dir = vaticanus_tei_dir or downloads_dir / "ntvmr-vaticanus"
    dest = (
        output or root / "data" / "sources" / "pilot" / "horner-vaticanus-nt-pilot.json"
    )

    try:
        verify_pilot_rebuild_inputs(
            manifests_dir=manifests_dir,
            sword_root=sword,
            vaticanus_tei_dir=tei_dir,
            downloads_dir=downloads_dir,
        )
        corpus = build_pilot_corpus(sword_root=sword, vaticanus_tei_dir=tei_dir)
    except (DownloadError, FileNotFoundError, OSError, ValueError) as exc:
        typer.secho(f"build-pilot failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    write_pilot_corpus(corpus, dest)
    typer.echo(f"Wrote {len(corpus.units)} parallel units to {dest}")


@sources_app.command("build-thomas")
def build_thomas(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect from cwd)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    tt_path: Path | None = typer.Option(
        None,
        "--tt",
        help="Local Dilley SCRIPTORIUM TreeTagger SGML (*.tt)",
        exists=True,
        dir_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination JSON for the redistributable Thomas corpus",
    ),
    download_if_missing: bool = typer.Option(
        False,
        "--download-if-missing",
        help="Fetch the pinned Dilley TT when the local file is absent",
    ),
) -> None:
    """Build the Dilley Thomas corpus from a local (checksum-verified) *.tt file."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(f"build-thomas requires a checkout: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    manifests_dir = root / "data" / "manifests"
    downloads_dir = root / "data" / "downloads"
    tt = tt_path or default_tt_path(root)
    dest = output or default_corpus_path(root)

    try:
        manifest = load_manifest(manifests_dir / "dilley-thomas.json")
        if not tt.is_file():
            if not download_if_missing:
                msg = (
                    f"missing Dilley TT at {tt}; run "
                    "`thomas sources download` or pass --download-if-missing"
                )
                raise FileNotFoundError(msg)
            tt = download_dilley_thomas(
                downloads_dir, manifests_dir=manifests_dir, force=False
            )
        verify_checksum(tt, manifest.checksum)
        corpus = parse_tt(tt, source_id=DILLEY_SOURCE_ID)
    except (DownloadError, FileNotFoundError, OSError, ValueError) as exc:
        typer.secho(f"build-thomas failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    write_thomas_corpus(corpus, dest)
    typer.echo(
        f"Wrote {len(corpus.units)} units ({EXPECTED_LOGIA} logia + prologue) to {dest}"
    )


@sources_app.command("build-fragments")
def build_fragments(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect from cwd)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    tei_dir: Path | None = typer.Option(
        None,
        "--tei-dir",
        help="Directory of bundled DCLP TEI XML files",
        exists=True,
        file_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination JSON for the layered fragment corpus",
    ),
) -> None:
    """Build layered Greek Thomas fragment corpus from local DCLP TEI (offline)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(f"build-fragments requires a checkout: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    source_tei = tei_dir or fragments_tei_dir(root)
    dest = output or fragments_corpus_path(root)
    manifests_path = root / "data" / "manifests"

    try:
        manifests = validate_manifests_dir(manifests_path)
        # Pin TEI bytes to manifests before parse (same pattern as build-pilot).
        verify_fragment_rebuild_inputs(
            manifests_dir=manifests_path,
            tei_dir=source_tei,
        )
        corpus = build_fragment_corpus(source_tei)
        validate_fragment_corpus_or_raise(corpus, manifests)
    except (
        DownloadError,
        FragmentValidationError,
        ManifestValidationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        typer.secho(f"build-fragments failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    write_fragment_corpus(corpus, dest)
    typer.echo(
        f"Wrote {len(corpus.transcriptions)} transcription(s) "
        f"({sum(len(t.lines) for t in corpus.transcriptions)} lines) to {dest}"
    )
