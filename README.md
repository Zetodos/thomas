# Thomas

Offline-capable, evidence-based reconstruction and reader for the *Gospel of Thomas*.

**Repository:** <https://github.com/Zetodos/thomas>
**Status:** Phase 0–1 foundation — licensing, Python toolchain, Horner × Vaticanus B* NT pilot (JAK-24), Dilley SCRIPTORIUM Coptic Thomas (JAK-30), Greek Thomas P.Oxy. fragments (JAK-31), alignment schema (JAK-32), Thomas 7/17/30/77 evidence pilot (JAK-34), and Greek–Sahidic alignment benchmark pilot (JAK-28). Reconstruction feasibility is not yet proven.

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
| [`docs/horner-vaticanus-alignment-notes.md`](docs/horner-vaticanus-alignment-notes.md) | CopSahHorner + Vaticanus B* pilot notes (JAK-24/28) |
| [`docs/alignment-pilot-splits.md`](docs/alignment-pilot-splits.md) | Train/dev/holdout/omission passage splits (JAK-28) |
| [`docs/alignment-pilot-benchmark.md`](docs/alignment-pilot-benchmark.md) | Lexical baseline benchmark + limitations (JAK-28) |
| [`docs/id-scheme.md`](docs/id-scheme.md) | Stable unit/token/span IDs + retokenization rule (JAK-32) |
| [`docs/greek-thomas-fragments.md`](docs/greek-thomas-fragments.md) | P.Oxy. 1/654/655 DCLP ingest notes (JAK-31) |
| [`docs/pilot-evidence-set.md`](docs/pilot-evidence-set.md) | Phase 1 Thomas 7/17/30/77 evidence pilot (JAK-34) |
| [`docs/benchmark-holdouts.md`](docs/benchmark-holdouts.md) | Train/dev/holdout + omission/leakage policy (JAK-33) |
| [`docs/greek-thomas-fragment-divergences.md`](docs/greek-thomas-fragment-divergences.md) | Greek fragment vs Coptic divergences (JAK-33) |
| [`data/manifests/`](data/manifests/) | Per-source JSON manifests (Horner, Vaticanus B*, Dilley, DCLP fragments) |
| [`data/sources/pilot/`](data/sources/pilot/) | Bundled Horner × Vaticanus B* pilot excerpt |
| [`data/sources/thomas/`](data/sources/thomas/) | Unit-level Dilley / SCRIPTORIUM Coptic Thomas import (JAK-30; not alignment-ready) |
| [`data/sources/fragments/`](data/sources/fragments/) | Bundled Greek Thomas fragments (evaluation-witness) |
| [`data/evidence/pilot/`](data/evidence/pilot/) | Thomas 7/17/30/77 evidence inventory (JSON + JSONL) |
| [`data/training/splits/`](data/training/splits/) | Canonical NT train/dev/holdout split v2 (JAK-33; consumed by JAK-28) |
| [`data/sources/alignment/`](data/sources/alignment/) | Provisional gold sample + benchmark JSON (JAK-28) |
| [`docs/statistical-aligner-comparison.md`](docs/statistical-aligner-comparison.md) | eflomal vs fast_align; Jake keep-both (JAK-36) |
| [`docs/lexical-candidates.md`](docs/lexical-candidates.md) | Coptic→Greek lexical candidate layers (JAK-37) |
| [`data/sources/lexicon/`](data/sources/lexicon/) | Lexical candidate table + train-link JSON (JAK-37) |
| [`docs/phrase-loanword-mappings.md`](docs/phrase-loanword-mappings.md) | Statistical constructions + verified loanwords + ablation (JAK-38) |
| [`data/sources/mappings/`](data/sources/mappings/) | Phrase/construction + CCL/DDGLC loanword + ablation JSON (JAK-38) |

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

- **Coptic Thomas (target):** **Dilley / Coptic SCRIPTORIUM** annotated Gospel of Thomas v6.2.0 (**CC BY 4.0**). Unit-level ingest with original + normalized layers (JAK-30); **not** token-complete / **not** alignment-ready yet. Manifest: [`data/manifests/dilley-thomas.json`](data/manifests/dilley-thomas.json).
- **Sahidic NT:** **Horner** via CrossWire **`CopSahHorner`** v1.5 (Public Domain). Do **not** treat Wells Sahidica as the NT baseline. Verified + pilot-imported: JAK-24.
- **Greek NT:** **INTF/NTVMR Codex Vaticanus GA 03 firsthand (B\*)** (`docID` 20003), CC BY 4.0 transcription (JAK-25 GO; updates the JAK-15 WH default). Do **not** mix Westcott–Hort or SBLGNT into lacunae. Do **not** ship manuscript images. Do **not** change this Greek spine in Coptic Thomas ingest work.
- **LXX / lexicon / critical texts:** Swete LXX, CCL TEI, and CC BY 4.0 for project-authored critical Greek/English — see [`docs/source-licenses.md`](docs/source-licenses.md).

Bundled corpora: [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](data/sources/pilot/horner-vaticanus-nt-pilot.json) (NT pilot), [`data/sources/thomas/dilley-thomas-v6.2.0.json`](data/sources/thomas/dilley-thomas-v6.2.0.json) (Dilley Thomas), and [`data/sources/fragments/`](data/sources/fragments/) (Greek Thomas fragments, evidence type G / `evaluation-witness`). Full downloads stay under gitignored `data/downloads/` (see that README).

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

The `thomas` console script exposes `sources validate|download|build-pilot|build-thomas|build-fragments`, `align validate|benchmark|eflomal-probe|aligner-probe|setup-aligners|statistical-eval`, and `lexicon generate-candidates|validate-candidates`. Further subcommands live in [`PLAN.md`](PLAN.md). Statistical aligner notes: [`docs/statistical-aligner-comparison.md`](docs/statistical-aligner-comparison.md). Lexical candidates: [`docs/lexical-candidates.md`](docs/lexical-candidates.md).

Package sources live under `python/zetodos_thomas/`. TypeScript stays under `web/` only (not started yet).

## Offline workflow

After clone and `uv sync --locked`:

1. **Offline immediately:** `uv run thomas sources validate` and `uv run thomas align validate` / `align benchmark` use committed manifests + bundled corpora + pilot gold only (no network).
2. **Optional setup downloads** (gitignored): `uv run thomas sources download` (see [`data/downloads/README.md`](data/downloads/README.md)); then `build-pilot` / `build-thomas` to regenerate imports.
3. After setup, CLI work must remain offline (no cloud APIs, CDNs, or hidden network fetches).

Default CLI/CI never fetches Sahidica, NTAEG, or Bohairica.
