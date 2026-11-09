# Thomas

Offline-capable, evidence-based reconstruction and reader for the *Gospel of Thomas*.

**Repository:** <https://github.com/Zetodos/thomas>
**Status:** Phase 0 — licensing, docs, and Python toolchain scaffold. Reconstruction feasibility is not yet proven. No corpus import or reconstruction engine ships yet.

## Objective

Develop a reproducible **Koine Greek reconstruction** of the Gospel of Thomas, primarily reconstructing Greek that could underlie the extant Coptic manuscript, and publish:

1. **Generated Greek (C)** — a deterministic, Coptic-derived reconstruction from an approved mapping model.
2. **Critical Greek** — a manually curated reconstruction assessed against Greek fragments, loanwords, NT/LXX parallels, and other early attestations.

A static web reader will expose Coptic, generated and critical Greek, rough English, and ancient evidence per logion, with bidirectional phrase alignment and search.

This project does **not** claim to recover a uniquely certain lost Greek original. It distinguishes attested from reconstructed material and documents disagreements among witnesses.

## Licensing

Project-owned source code is licensed under the Apache License, Version 2.0. Third-party source texts and linguistic datasets, as well as Zetodos's reconstructed texts, translations, and derived mapping datasets, may have separate terms recorded in their respective manifests.

The root [`LICENSE`](LICENSE) covers **project-owned software only** (Python tooling, TypeScript web app, project schemas/tests/config). It is **not** an umbrella license for corpora or editorial datasets. See [`docs/source-licenses.md`](docs/source-licenses.md).

## Documentation

| Doc | Purpose |
| --- | --- |
| [`PLAN.md`](PLAN.md) | Full implementation plan |
| [`docs/source-licenses.md`](docs/source-licenses.md) | Dual code/data policy, setup modes, never-ship list, manifest schema |
| [`docs/licensing-provenance-audit.md`](docs/licensing-provenance-audit.md) | 2026-11-09 provenance audit (updated for Horner decision) |
| [`data/manifests/`](data/manifests/) | Future per-source machine-readable manifests |

## Never ship (summary)

Do **not** bundle or redistribute:

- NA28 / UBS GNT (Deutsche Bibelgesellschaft)
- Rahlfs-Hanhart LXX 2006 (DBG)
- Layton / Attridge continuous critical Thomas Greek
- Crum dictionary OCR dumps
- Brill / DBG full texts used only as scholarly references
- Wells **Sahidica**, **NTAEG**, or **Bohairica** (gated; not default)

Full list and allowlist: [`docs/source-licenses.md`](docs/source-licenses.md#never-ship).

## Text baselines (Phase 0 decisions)

- **Sahidic NT:** **Horner** via CrossWire **`CopSahHorner`** (Public Domain). Do **not** treat Wells Sahidica as the NT baseline. Import/verification: [JAK-24](https://linear.app/jake-danton/issue/JAK-24).
- **Greek NT:** **Westcott–Hort 1881** (PD Robinson transcription) per [JAK-15](https://linear.app/jake-danton/issue/JAK-15). SBLGNT (now CC BY 4.0) is a separate revisit — [JAK-25](https://linear.app/jake-danton/issue/JAK-25); do not flip the default here.
- **LXX / lexicon / critical texts:** Swete LXX, CCL TEI, and CC BY 4.0 for project-authored critical Greek/English — see [`docs/source-licenses.md`](docs/source-licenses.md).

This repo does not yet contain corpus files.

## Python setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --locked
uv run thomas --help
```

### Pre-commit hooks

Install git hooks with one command (after `uv sync --locked`):

```bash
uv run pre-commit install
```

That installs the same suite CI runs (`uv run pre-commit run --all-files`): `uv lock --check`, ruff format/check, `ty`, pytest (fast suite), trailing whitespace / EOF / YAML / TOML checks, large-file guard, and SPDX headers on `python/` and `tests/`. Hook revisions are pinned in [`.pre-commit-config.yaml`](.pre-commit-config.yaml); Dependabot (and a weekly autoupdate workflow) keep them current.

The weekly autoupdate workflow opens a PR only when repository secret `PRE_COMMIT_AUTOUPDATE_TOKEN` is set to a fine-grained PAT or GitHub App token with `contents` + `pull-requests` write (so the PR can trigger CI). Without that secret it still runs `pre-commit autoupdate` and skips PR creation; Dependabot remains the primary update path.

Manual equivalents without hooks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

The `thomas` console script is a Phase 0 Typer scaffold (`--help` only). Planned subcommands live in [`PLAN.md`](PLAN.md); do not invent them ahead of their phases.

Package sources live under `python/zetodos_thomas/`. TypeScript stays under `web/` only (not started yet).

## Offline workflow (planned)

After clone and `uv sync --locked`:

1. Run an explicit one-time **setup** that may download permitted corpora and optional local models (not implemented yet).
2. After setup, CLI commands, site builds, and local website use must work **offline** (no cloud APIs, CDNs, or hidden network fetches).

## Phase 0 scope

This phase covers license/docs, the Python toolchain scaffold (uv, ruff, ty, pytest, Typer, Pydantic), and pre-commit/CI quality gates. Out of scope: web app, corpus downloads, Horner import, and reconstruction.
