# SPDX-License-Identifier: Apache-2.0
"""``thomas reconstruct`` Typer subcommands (100% offline, deterministic)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from zetodos_thomas.cli.model_cmd import (
    MODEL_REL,
    MODEL_V2_REL,
    MODEL_V3_REL,
    MODEL_V4_REL,
)
from zetodos_thomas.ingest.dilley_thomas import load_thomas_corpus
from zetodos_thomas.ingest.paths import require_repo_root, resolve_thomas_path
from zetodos_thomas.reconstruct.decoder import KoineBeamDecoder

GENERATED_THOMAS_REL = Path("data/generated/thomas-generated-v1.json")
GENERATED_THOMAS_V2_REL = Path("data/generated/thomas-generated-v2.json")
GENERATED_THOMAS_V3_REL = Path("data/generated/thomas-generated-v3.json")
GENERATED_THOMAS_V4_REL = Path("data/generated/thomas-generated-v4.json")

reconstruct_app = typer.Typer(
    name="reconstruct",
    help="Generate deterministic Category C Koine Greek for Gospel of Thomas.",
    no_args_is_help=True,
)


@reconstruct_app.command("run")
def reconstruct_run(
    saying: str | None = typer.Option(
        None,
        "--saying",
        help="Single Thomas unit_id (e.g. thomas.2)",
    ),
    all_sayings: bool = typer.Option(
        False,
        "--all",
        help="Reconstruct all 115 Thomas units (thomas.0..thomas.114)",
    ),
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output JSON path (default: data/generated/thomas-generated-v*.json)",
    ),
    model: str = typer.Option(
        "v1",
        "--model",
        help="Model track: v1 (classical), v2 (neural), v3, or v4 (faithfulness-first)",
    ),
) -> None:
    """Deterministically reconstruct Koine Greek for Thomas saying(s) offline."""
    if not saying and not all_sayings:
        typer.secho("Specify --saying <id> or --all", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    root = require_repo_root(repo_root)
    m_low = model.lower()
    is_v4 = m_low in {"v4", "model-v4"}
    is_v3 = m_low in {"v3", "model-v3"}
    is_v2 = m_low in {"v2", "model-v2"}
    is_neural = is_v2 or is_v3 or is_v4
    v1_dir = root / MODEL_REL
    model_dir = (
        (root / MODEL_V4_REL)
        if is_v4
        else (root / MODEL_V3_REL)
        if is_v3
        else (root / MODEL_V2_REL)
        if is_v2
        else v1_dir
    )
    marker = "seq2seq-config.json" if is_neural else "decoder-weights.json"
    if not (model_dir / marker).is_file():
        cmd = (
            "thomas model build-v4"
            if is_v4
            else "thomas model build-v3"
            if is_v3
            else "thomas model build-v2"
            if is_v2
            else "thomas model build"
        )
        typer.secho(
            f"Missing frozen model at {model_dir}; run `{cmd}` first",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    corpus = load_thomas_corpus(resolve_thomas_path(root))
    if is_neural:
        from zetodos_thomas.neural.decoder_v2 import NeuralKoineDecoder

        decoder_nn = NeuralKoineDecoder.from_model_dir(
            model_dir, v1_model_dir=v1_dir, device="cpu"
        )
        decoder: object = decoder_nn
    else:
        decoder = KoineBeamDecoder.from_model_dir(model_dir)

    if saying and not all_sayings:
        match = next((u for u in corpus.units if u.unit_id == saying), None)
        if match is None:
            typer.secho(f"Unknown Thomas saying id: {saying!r}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        gen = decoder.decode_unit(match)  # type: ignore[attr-defined]
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    gen.model_dump(mode="json", exclude_defaults=True),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        typer.echo(f"{gen.unit_id}: {gen.greek_text}")
        return

    model_id = (
        "model-v4"
        if is_v4
        else "model-v3"
        if is_v3
        else "model-v2"
        if is_v2
        else "model-v1"
    )
    if is_neural:
        generated = decoder.decode_corpus(corpus.units, model_id=model_id)  # type: ignore[attr-defined]
    else:
        generated = decoder.decode_corpus(corpus.units)  # type: ignore[attr-defined]
    default_rel = (
        GENERATED_THOMAS_V4_REL
        if is_v4
        else GENERATED_THOMAS_V3_REL
        if is_v3
        else GENERATED_THOMAS_V2_REL
        if is_v2
        else GENERATED_THOMAS_REL
    )
    out_path = out or (root / default_rel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = generated.model_dump(mode="json", exclude_defaults=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    if len(serialized.encode("utf-8")) >= 1024 * 1024:
        # Trim lower-priority alternatives so file stays strictly < 1024 KiB
        for u in payload.get("units", []):
            u.pop("clause_alternatives", None)
            u.pop("coptic_text", None)
            for tok in u.get("tokens", []):
                if "alternatives" in tok and len(tok["alternatives"]) > 2:
                    tok["alternatives"] = tok["alternatives"][:2]
        serialized = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    if len(serialized.encode("utf-8")) >= 1024 * 1024:
        # Preserve compact payload if per-token alternatives still exceed 1024 KiB
        for u in payload.get("units", []):
            for tok in u.get("tokens", []):
                tok.pop("alternatives", None)
        serialized = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    out_path.write_text(serialized, encoding="utf-8")
    size_bytes = out_path.stat().st_size
    if size_bytes >= 1024 * 1024:
        typer.secho(
            f"Generated output {out_path} ({size_bytes} bytes) >= 1024 KiB",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    tok_n = sum(len(u.tokens) for u in generated.units)
    typer.echo(
        f"OK: reconstructed {len(generated.units)} Thomas units ({tok_n} Greek tokens, "
        f"{size_bytes} bytes) to {out_path}"
    )


@reconstruct_app.command("inspect")
def reconstruct_inspect(
    saying: str = typer.Option(
        ...,
        "--saying",
        help="Thomas unit_id to inspect (e.g. thomas.7)",
    ),
    model: str = typer.Option(
        "v1",
        "--model",
        help="Model track: v1 (default), v2, v3, or v4 (e.g. --model v4)",
    ),
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Print morpheme-level alignment & completeness audit for one Thomas saying."""
    from zetodos_thomas.neural.constrained_beam import CONTENT_COPTIC_POS

    root = require_repo_root(repo_root)
    m_low = model.lower()
    is_v4 = m_low in {"v4", "model-v4"}
    is_v3 = m_low in {"v3", "model-v3"}
    is_v2 = m_low in {"v2", "model-v2"}
    is_neural = is_v2 or is_v3 or is_v4
    v1_dir = root / MODEL_REL
    model_dir = (
        (root / MODEL_V4_REL)
        if is_v4
        else (root / MODEL_V3_REL)
        if is_v3
        else (root / MODEL_V2_REL)
        if is_v2
        else v1_dir
    )
    corpus = load_thomas_corpus(resolve_thomas_path(root))
    match = next((u for u in corpus.units if u.unit_id == saying), None)
    if match is None:
        typer.secho(f"Unknown Thomas saying id: {saying!r}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if is_neural:
        from zetodos_thomas.neural.decoder_v2 import NeuralKoineDecoder

        dec: object = NeuralKoineDecoder.from_model_dir(
            model_dir, v1_model_dir=v1_dir, device="cpu"
        )
    else:
        dec = KoineBeamDecoder.from_model_dir(model_dir)

    gen = dec.decode_unit(match)  # type: ignore[attr-defined]
    seq_to_toks: dict[int, list[str]] = {}
    for tok in gen.tokens:
        for sq in tok.coptic_seqs:
            seq_to_toks.setdefault(sq, []).append(f"{tok.seq}:{tok.surface}")

    typer.echo(f"=== {saying} ({model_dir.name}) ===")
    typer.echo(f"COPTIC: {match.normalized_text}")
    typer.echo(f"GREEK : {gen.greek_text}")
    typer.echo("")
    unaligned_content: list[str] = []
    for m in match.morphemes:
        mapped = ", ".join(seq_to_toks.get(m.seq, ())) or "-"
        is_c = m.pos in CONTENT_COPTIC_POS
        if is_c and mapped == "-":
            unaligned_content.append(f"{m.seq}:{m.norm}({m.lemma}/{m.pos})")
        typer.echo(
            f"  [{m.seq:2d}] {m.norm:12s} lem={m.lemma:12s} pos={m.pos:8s} -> {mapped}"
        )
    n_c_tot = sum(1 for m in match.morphemes if m.pos in CONTENT_COPTIC_POS)
    typer.echo(
        f"\nContent morphemes: {n_c_tot - len(unaligned_content)}/{n_c_tot} aligned; "
        f"unaligned content: {unaligned_content or 'none'}"
    )
