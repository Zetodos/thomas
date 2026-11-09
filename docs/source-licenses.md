# Source-license policy and manifest schema

**Status:** Phase 0 policy (enforce before any corpus import).
**Related:** [PLAN.md](../PLAN.md) §4 · [licensing-provenance-audit.md](licensing-provenance-audit.md) · Linear JAK-12 / JAK-22 / JAK-23

Caveat: not legal advice. Re-check licenses and pin versions before bundling.

## Dual code / data policy

| Content | License |
| --- | --- |
| Project-owned software (Python, TypeScript `web/`, schemas, tests, tooling) | **Apache-2.0** (root [`LICENSE`](../LICENSE)) |
| Imported texts, dictionaries, corpora | **Per-source** terms in manifests — never assumed Apache/CC |
| Mapping models / derived parallel DBs | ShareAlike **only when** the published artifact is a legal adaptation of BY-SA material (JAK-16); otherwise project/data terms as recorded |
| Critical Greek / rough English (project-authored) | **CC BY 4.0** (JAK-17) |

Do **not** put the whole repository under a single Creative Commons or Apache license. README and manifests must state boundaries explicitly.

## Setup modes

Every source manifest must declare exactly one availability mode:

| Mode | Meaning |
| --- | --- |
| `bundled` | Safe to ship in git (with attribution / required notices) |
| `setup-download` | Fetch during explicit setup from a known URL; not committed |
| `manual-import` | User supplies files; repo documents expected paths/checksums only |

**Invariant:** the reproducible working dataset must be present locally *before* an offline run. No hidden data fetching during build or reconstruction. Making a restricted corpus downloadable during setup does **not** by itself authorize redistribution.

## Allowlist (pilot / clean stack)

Adopted redistributable core (JAK-15 Done). Pin versions at ingest. Sahidica is **not** required.

| Family | Typical mode | Notes |
| --- | --- | --- |
| Dilley Thomas (Coptic SCRIPTORIUM) | `bundled` or `setup-download` | CC BY 4.0 |
| P.Oxy. fragments (verify each transcription) | `bundled` (small) | Prefer DCLP (CC BY 3.0) or PD Grenfell/Hunt; Evelyn-White **1920** original is US PD (see audit §2) |
| **Horner Sahidic NT** (`CopSahHorner`) | `setup-download` (+ bundled pilot excerpt) | **Default Sahidic NT** (PD module; JAK-14 Done; ingest JAK-24). Manifest: [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json). Alignment notes: [`horner-wh-alignment-notes.md`](horner-wh-alignment-notes.md) |
| **Westcott–Hort 1881 Greek NT** (PD Robinson transcription) | `setup-download` (+ bundled pilot excerpt) | **Default Greek NT** per JAK-15. Manifest: [`data/manifests/westcott-hort.json`](../data/manifests/westcott-hort.json) |
| SCRIPTORIUM `sahidic.ot` | `setup-download` / `bundled` | CC BY-SA 4.0 (secondary) |
| **Swete LXX** (First1KGreek / nathans/lxx-swete) | `setup-download` | CC BY-SA 4.0 (JAK-18); not Rahlfs-Hanhart; Rahlfs 1935 not preferred |
| **CCL / DDGLC TEI** | `setup-download` / `bundled` | CC BY-SA 4.0 (JAK-19); do not wait on DIOSKOROS; no scrape |
| WEB English passages | `bundled` (passages) | Public domain; trademark care |

### SBLGNT follow-up (do not flip default yet)

Westcott–Hort remains the **current** default Greek NT under JAK-15. SBLGNT’s current **CC BY 4.0** status (since Dec 2022) is tracked separately in JAK-25 (revisit SBLGNT vs WH). Do **not** switch the default mid-stream in this Phase 0 PR.

## ShareAlike policy (JAK-16 Done)

- Apply ShareAlike **only** when published outputs are legally **adaptations** of BY-SA material.
- Do **not** blanket-SA the whole project, all learned statistics, or independent critical texts.
- Keep separable: raw source imports, adapted tables/DBs, independent critical Greek/English, and Apache-2.0 code.
- Manifest each published artifact with license and derivation notes.

## Gated: Wells Sahidica / NTAEG / Bohairica

| Source | Policy |
| --- | --- |
| Sahidica (J. Warren Wells) | **Do not bundle.** Custom license; Thomas training/parallel use is not clearly within the “free electronic editions of the New Testament” grant. |
| NTAEG | **Do not bundle.** Companion Greek; rights not assumed from Sahidica. |
| Bohairica | **Do not bundle.** Copyright / electronic permission unclear. Mode: gated **`manual-import` only** after rights confirmation — **not** `setup-download`. |

Optional user-supplied import may be documented later without distributing files or claiming permission. Default offline operation must not require these texts (JAK-14 Done).

## Never ship

Hard exclusions — do not commit, vendor, or redistribute:

| Edition / work | Reason |
| --- | --- |
| **NA28 / UBS GNT** | © Deutsche Bibelgesellschaft; redistribution requires written permission |
| **Rahlfs-Hanhart LXX 2006** | © DBG |
| **Layton / Attridge** continuous critical Thomas Greek | Brill / NHS copyrighted critical text |
| **Crum, A Coptic Dictionary** OCR dumps | OUP / not safely redistributable wholesale |
| Other **Brill / DBG** full texts used only as scholarly references | Copyrighted; cite, do not ship |
| **Wells Sahidica, NTAEG, Bohairica** | Gated custom/unclear rights — excluded from default artifacts |

Reference-only citation and short fair-use quotation policy may apply in scholarly notes; that is not permission to package continuous text.

**Note:** Evelyn-White’s 1920 *The Sayings of Jesus from Oxyrhynchus* is **US public domain** and is **not** in this never-ship list. Later copyrighted editorial work (e.g. Attridge/Layton) remains excluded.

## Manifest schema

Machine-readable manifests live under [`data/manifests/`](../data/manifests/) as **JSON** (one file per source). Typed model: `zetodos_thomas.schemas.source.SourceManifest`. Validate offline with `uv run thomas sources validate`.

Full CrossWire / Robinson downloads stay under gitignored `data/downloads/`. A redistributable PD pilot excerpt is committed at [`data/sources/pilot/horner-wh-nt-pilot.json`](../data/sources/pilot/horner-wh-nt-pilot.json).

### Required / expected fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable canonical source ID (e.g. `copsahhorner`, `westcott-hort`) |
| `title` | string | Human-readable title / edition name |
| `version` | string | Edition or module version / date pin |
| `url` | string | Provenance / download URL |
| `spdx_or_custom` | string | SPDX id (e.g. `Apache-2.0`, `CC-BY-4.0`, `CC0-1.0`) or `custom:` + short label |
| `attribution` | string | Required credit / copyright notice text |
| `mode` | enum | `bundled` \| `setup-download` \| `manual-import` |
| `checksum` | string | Content hash of the ingested artifact (algorithm prefixed, e.g. `sha256:…`) |
| `notes` | string | Restrictions, dialect, known issues, open questions |

### Optional fields (recommended)

| Field | Type | Description |
| --- | --- | --- |
| `editor` | string | Editor / publisher |
| `language` | string | ISO or project language tag |
| `dialect` | string | e.g. Sahidic, Bohairic |
| `license_url` | string | Primary license page |
| `local_path` | string | Expected path after setup |
| `redistributable` | boolean | Explicit project judgment |
| `gate` | string | If gated: reason / Linear issue |

### Live example

See [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json) (checksum pinned for `CopSahHorner.zip`) and [`data/manifests/westcott-hort.json`](../data/manifests/westcott-hort.json) (git commit + `JOH.WH` checksum).

## Decision status

| Issue | Topic | Status | Adopted policy |
| --- | --- | --- | --- |
| JAK-13 | Code license | **Done** | Apache-2.0 for project-owned software only |
| JAK-14 | Sahidic NT / Wells | **Done** (decision) | Horner/`CopSahHorner` default; do not bundle Sahidica/NTAEG/Bohairica without explicit rights. Implementation: JAK-24 |
| JAK-15 | Pilot corpus | **Done** | Redistributable stack above; Greek NT = Westcott–Hort (PD Robinson). SBLGNT revisit: JAK-25 |
| JAK-16 | ShareAlike outputs | **Done** | SA only on legal adaptations of BY-SA material; no blanket SA |
| JAK-17 | Critical Greek/English | **Done** | Project-authored critical Greek + rough English: **CC BY 4.0** |
| JAK-18 | LXX edition | **Done** | Default Swete via First1KGreek / nathans/lxx-swete (CC BY-SA 4.0) |
| JAK-19 | Lexicon / loanwords | **Done** | Default CCL TEI incl. DDGLC (CC BY-SA 4.0); not DIOSKOROS scrape |
