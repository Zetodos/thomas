# Thomas

Offline-capable, evidence-based reconstruction and reader for the *Gospel of Thomas*.

**Repository:** <https://github.com/Zetodos/thomas>  
**Status:** Phase 0 — licensing and repo foundation. Reconstruction feasibility is not yet proven. No CLI, package, or corpus import ships yet.

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

## Planned setup (not implemented)

When tooling exists, the intended workflow is:

1. Clone the repository.
2. Create a local environment with **`uv`** (Python) and install declared dependencies.
3. Run an explicit one-time **setup** that may download permitted corpora and optional local models.
4. After setup, CLI commands, site builds, and local website use must work **offline** (no cloud APIs, CDNs, or hidden network fetches).

There is **no** `thomas` CLI yet. Do not invent commands; follow `PLAN.md` until Phase 1 scaffolding lands.

## Phase 0 scope

This phase adds license, README, and source-policy docs only. Out of scope: Python package, web app, corpus downloads, and Horner import.
