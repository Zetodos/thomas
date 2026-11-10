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
| **Dilley Thomas (Coptic SCRIPTORIUM)** | `setup-download` (+ bundled **unit-level** import) | **Primary Coptic Thomas target** (JAK-30). **CC BY 4.0**; attribute Dilley / SCRIPTORIUM. Manifest: [`data/manifests/dilley-thomas.json`](../data/manifests/dilley-thomas.json). Import: [`data/sources/thomas/dilley-thomas-v6.2.0.json`](../data/sources/thomas/dilley-thomas-v6.2.0.json) — **unit-level only** (orig+norm TextUnits); **not** token-complete / **not** alignment-ready. Full SCRIPTORIUM token/group data deferred; use JAK-32 ID grammar when it lands. |
| P.Oxy. fragments (verify each transcription) | `bundled` (small) | **JAK-31 imported:** DCLP TEI for P.Oxy. **1 / 654 / 655** — each verified **CC BY 3.0** (do not assume PD). Manifests `dclp-poxy-*`. Evidence type **G**, `corpus_role=evaluation-witness`. Prefer DCLP or PD Grenfell/Hunt / Evelyn-White **1920** (see audit §2). **Do not** label P.Oxy. 5575 as a Thomas manuscript. |
| **Horner Sahidic NT** (`CopSahHorner`) | `setup-download` (+ bundled pilot excerpt) | **Default Sahidic NT** (PD module; JAK-14 Done; ingest JAK-24). Manifest: [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json). Alignment notes: [`horner-vaticanus-alignment-notes.md`](horner-vaticanus-alignment-notes.md) |
| **NTVMR Vaticanus GA 03 B\*** (INTF TEI, `docID` 20003) | `setup-download` (+ bundled pilot excerpt) | **Default Greek NT** (JAK-25 GO; updates JAK-15 WH default). **CC BY 4.0** transcription; attribute INTF/NTVMR. Firsthand only; empty firsthand = omission; **no** corrector fill; **no** WH/SBLGNT lacuna fill; **do not** ship images. Manifest: [`data/manifests/ntvmr-vaticanus-b.json`](../data/manifests/ntvmr-vaticanus-b.json) |
| Westcott–Hort 1881 (Robinson) | not default | Documented unused fallback only; **not** in default manifests/CLI |
| SCRIPTORIUM `sahidic.ot` | `setup-download` / `bundled` | CC BY-SA 4.0 (secondary) |
| **Swete LXX** (First1KGreek / nathans/lxx-swete) | `setup-download` | CC BY-SA 4.0 (JAK-18); not Rahlfs-Hanhart; Rahlfs 1935 not preferred |
| **CCL / DDGLC TEI** | `setup-download` / `bundled` | CC BY-SA 4.0 (JAK-19); do not wait on DIOSKOROS; no scrape |
| WEB English passages | `bundled` (passages) | Public domain; trademark care |

### Attribution — Dilley / SCRIPTORIUM Coptic Thomas (CC BY 4.0)

Required credit for the Coptic Thomas import: **Paul C. Dilley** (edition and primary annotations); **Coptic SCRIPTORIUM**; annotation/metadata contributions by Caroline T. Schroeder, Lydia Bremer-McCollum, and Amir Zeldes; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Pin release **v6.2.0**, URL, and checksum in [`data/manifests/dilley-thomas.json`](../data/manifests/dilley-thomas.json). Original (manuscript-oriented) and normalized analysis layers are separate recoverable representations of one edition — not competing Coptic texts. Do **not** substitute Layton/Attridge or Bethge continuous critical Greek. Facsimile images are not part of this grant.

### Attribution — NTVMR / INTF Vaticanus transcription (CC BY 4.0)

Required credit for the Greek TEI and derived pilot Greek: **Institut für neutestamentliche Textforschung (INTF / NTVMR)**; Codex Vaticanus GA 03 / Vat. gr. 1209 transcription; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Pin retrieval URL, date, and checksums in the source manifest. Site image restrictions are separate — **never** package Vatican/NTVMR manuscript images.

### SBLGNT / Westcott–Hort

Default Greek NT is **Vaticanus B\*** (JAK-25). SBLGNT (CC BY 4.0) and Westcott–Hort remain non-default and must not be mixed into empty B\* slots.

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
| **Vatican / NTVMR images** of Vat. gr. 1209 | Image rights restricted; CC BY 4.0 covers the **transcription**, not images |

Reference-only citation and short fair-use quotation policy may apply in scholarly notes; that is not permission to package continuous text.

**Note:** Evelyn-White’s 1920 *The Sayings of Jesus from Oxyrhynchus* is **US public domain** and is **not** in this never-ship list. Later copyrighted editorial work (e.g. Attridge/Layton) remains excluded.

## Manifest schema

Machine-readable manifests live under [`data/manifests/`](../data/manifests/) as **JSON** (one file per source). Typed model: `zetodos_thomas.schemas.source.SourceManifest`. Validate offline with `uv run thomas sources validate`.

Full CrossWire / NTVMR TEI / Dilley TT downloads stay under gitignored `data/downloads/`. Redistributable imports are committed at [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](../data/sources/pilot/horner-vaticanus-nt-pilot.json), [`data/sources/thomas/dilley-thomas-v6.2.0.json`](../data/sources/thomas/dilley-thomas-v6.2.0.json), and [`data/sources/fragments/`](../data/sources/fragments/) — see [`greek-thomas-fragments.md`](greek-thomas-fragments.md).

### Required / expected fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable canonical source ID (e.g. `copsahhorner`, `ntvmr-vaticanus-b`) |
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

See [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json), [`data/manifests/ntvmr-vaticanus-b.json`](../data/manifests/ntvmr-vaticanus-b.json), [`data/manifests/dilley-thomas.json`](../data/manifests/dilley-thomas.json), and [`data/manifests/dclp-poxy-1.json`](../data/manifests/dclp-poxy-1.json).

## Decision status

| Issue | Topic | Status | Adopted policy |
| --- | --- | --- | --- |
| JAK-13 | Code license | **Done** | Apache-2.0 for project-owned software only |
| JAK-14 | Sahidic NT / Wells | **Done** (decision) | Horner/`CopSahHorner` default; do not bundle Sahidica/NTAEG/Bohairica without explicit rights. Implementation: JAK-24 |
| JAK-15 | Pilot corpus | **Done** (Greek default updated by JAK-25) | Redistributable stack; **Greek NT = Vaticanus B\*** (INTF/NTVMR CC BY 4.0). WH no longer default. |
| JAK-16 | ShareAlike outputs | **Done** | SA only on legal adaptations of BY-SA material; no blanket SA |
| JAK-17 | Critical Greek/English | **Done** | Project-authored critical Greek + rough English: **CC BY 4.0** |
| JAK-18 | LXX edition | **Done** | Default Swete via First1KGreek / nathans/lxx-swete (CC BY-SA 4.0) |
| JAK-19 | Lexicon / loanwords | **Done** | Default CCL TEI incl. DDGLC (CC BY-SA 4.0); not DIOSKOROS scrape |
| JAK-30 | Dilley Coptic Thomas ingest | **In progress** | SCRIPTORIUM Dilley v6.2.0 CC BY 4.0; orig+norm layers; 114 logia |
