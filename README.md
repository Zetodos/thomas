# Thomas

Offline-capable, evidence-based reconstruction and reader for the *Gospel of Thomas*.

**Repository:** <https://github.com/Zetodos/thomas>
**Status:** Phase 0–1 foundation — licensing, Python toolchain, and a redistributable Horner × Vaticanus B* NT pilot import (JAK-24). Reconstruction feasibility is not yet proven.

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
| [`docs/horner-vaticanus-alignment-notes.md`](docs/horner-vaticanus-alignment-notes.md) | CopSahHorner + Vaticanus B* pilot notes (JAK-24) |
| [`data/manifests/`](data/manifests/) | Per-source JSON manifests (Horner, Vaticanus B*) |
| [`data/sources/pilot/`](data/sources/pilot/) | Bundled Horner × Vaticanus B* pilot excerpt |

## Never ship (summary)

Do **not** bundle or redistribute:

- NA28 / UBS GNT (Deutsche Bibelgesellschaft)
- Rahlfs-Hanhart LXX 2006 (DBG)
- Layton / Attridge continuous critical Thomas Greek
- Crum dictionary OCR dumps
- Brill / DBG full texts used only as scholarly references
- Wells **Sahidica**, **NTAEG**, or **Bohairica** (gated; not default)
- Vatican / NTVMR **manuscript images** of Vat. gr. 1209 (transcription CC BY 4.0 ≠ image rights)

Full list and allowlist: [`docs/source-licenses.md`](docs/source-licenses.md#never-ship).

## Text baselines (Phase 0–1 decisions)

- **Sahidic NT:** **Horner** via CrossWire **`CopSahHorner`** v1.5 (Public Domain). Do **not** treat Wells Sahidica as the NT baseline. Verified + pilot-imported: JAK-24.
- **Greek NT:** **INTF/NTVMR Codex Vaticanus GA 03 firsthand (B\*)** (`docID` 20003), CC BY 4.0 transcription (JAK-25 GO; updates the JAK-15 WH default). Do **not** mix Westcott–Hort or SBLGNT into lacunae. Do **not** ship manuscript images.
- **LXX / lexicon / critical texts:** Swete LXX, CCL TEI, and CC BY 4.0 for project-authored critical Greek/English — see [`docs/source-licenses.md`](docs/source-licenses.md).

Bundled pilot: [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](data/sources/pilot/horner-vaticanus-nt-pilot.json). Full downloads stay under gitignored `data/downloads/` (see that README).

## Python setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --locked
uv run thomas --help
uv run thomas sources validate
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

The `thomas` console script currently exposes `sources validate` (offline), `sources download` (explicit setup fetch into `data/downloads/`), and `sources build-pilot` (local rebuild from downloads). Further subcommands live in [`PLAN.md`](PLAN.md).

Package sources live under `python/zetodos_thomas/`. TypeScript stays under `web/` only (not started yet).

## Offline workflow

After clone and `uv sync --locked`:

1. **Offline immediately:** `uv run thomas sources validate` uses committed manifests + the PD pilot only (no network).
2. **Optional setup downloads** (gitignored): `uv run thomas sources download` (see [`data/downloads/README.md`](data/downloads/README.md)); then `uv run thomas sources build-pilot` to regenerate the pilot.
3. After setup, CLI work must remain offline (no cloud APIs, CDNs, or hidden network fetches).

Default CLI/CI never fetches Sahidica, NTAEG, or Bohairica.
