# Source-license policy and manifest schema

**Status:** Phase 0 policy (enforce before any corpus import).  
**Related:** [PLAN.md](../PLAN.md) §4 · [licensing-provenance-audit.md](licensing-provenance-audit.md) · Linear [JAK-12](https://linear.app/jake-danton/issue/JAK-12) / [JAK-22](https://linear.app/jake-danton/issue/JAK-22) / [JAK-23](https://linear.app/jake-danton/issue/JAK-23)

Caveat: not legal advice. Re-check licenses and pin versions before bundling.

## Dual code / data policy

| Content | License |
| --- | --- |
| Project-owned software (Python, TypeScript `web/`, schemas, tests, tooling) | **Apache-2.0** (root [`LICENSE`](../LICENSE)) |
| Imported texts, dictionaries, corpora | **Per-source** terms in manifests — never assumed Apache/CC |
| Mapping models / derived parallel DBs | Separate derivation-rights decision (see open items) |
| Critical Greek / rough English (project-authored) | Separate editorial license decision — **not decided** |

Do **not** put the whole repository under a single Creative Commons or Apache license. README and manifests must state boundaries explicitly.

## Setup modes

Every source manifest must declare exactly one availability mode:

| Mode | Meaning |
| --- | --- |
| `bundled` | Safe to ship in git (with attribution / required notices) |
| `setup-download` | Fetch during explicit setup from a known URL; not committed |
| `manual-import` | User supplies files; repo documents expected paths/checksums only |

**Invariant:** the reproducible working dataset must be present locally *before* an offline run. No hidden data fetching during build or reconstruction. Making a restricted corpus downloadable during setup does **not** by itself authorize redistribution.

## Provisional allowlist (clean stack)

Preferred redistributable core for an offline open repo (provisional; pin versions at ingest):

| Family | Typical mode | Notes |
| --- | --- | --- |
| Dilley Thomas (Coptic SCRIPTORIUM) | `bundled` or `setup-download` | CC BY |
| P.Oxy. via DCLP or PD Grenfell/Hunt | `bundled` (small) | CC BY 3.0 / PD |
| **Horner Sahidic NT** (`CopSahHorner`) | `setup-download` (later) | **Default Sahidic NT baseline** (PD module; JAK-14) |
| SBLGNT (+ MorphGNT morph) | `bundled` or `setup-download` | CC BY / morph BY-SA |
| Swete LXX (First1KGreek) | `setup-download` | BY-SA digitization; lean preference pending JAK-18 |
| SCRIPTORIUM `sahidic.ot` | `setup-download` / `bundled` | CC BY-SA 4.0 |
| CCL / DDGLC TEI | `setup-download` / `bundled` | CC BY-SA 4.0; lean preference pending JAK-19 |
| WEB English passages | `bundled` (passages) | Public domain; trademark care |

**Pilot note (JAK-15 open):** a legally clean pilot can use **Horner + SBLGNT** (plus Dilley Thomas, OT/CCL as needed) without Wells corpora.

## Gated: Wells Sahidica / NTAEG / Bohairica

| Source | Policy |
| --- | --- |
| Sahidica (J. Warren Wells) | **Do not bundle.** Custom license; Thomas training/parallel use is not clearly within the “free electronic editions of the New Testament” grant. |
| NTAEG | **Do not bundle.** Companion Greek; rights not assumed from Sahidica. |
| Bohairica | **Do not bundle.** Copyright / electronic permission unclear. |

Optional user-supplied import may be documented later without distributing files or claiming permission. Default offline operation must not require these texts ([JAK-14](https://linear.app/jake-danton/issue/JAK-14) Done).

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

## Manifest schema (draft)

Machine-readable manifests will live under [`data/manifests/`](../data/manifests/) (YAML or JSON). No corpus files are committed in Phase 0.

### Required / expected fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable canonical source ID (e.g. `copsahhorner`, `sblgnt`) |
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

### Example stub (illustrative — not a live corpus)

```yaml
id: copsahhorner
title: Sahidic Coptic New Testament (Horner) — CrossWire CopSahHorner
version: "1.5"
url: https://www.crosswire.org/sword/modules/ModInfo.jsp?modName=CopSahHorner
spdx_or_custom: "custom:Public-Domain"  # CrossWire DistributionLicense=Public Domain
attribution: "G. W. Horner; CrossWire CopSahHorner module"
mode: setup-download
checksum: ""  # fill at ingest
notes: "Default Sahidic NT baseline (JAK-14). Pin module; verify verse map vs Greek spine."
redistributable: true
```

## Open decisions (do not treat as resolved)

Recorded briefly; provisional leans from the audit only:

| Issue | Topic | Status |
| --- | --- | --- |
| [JAK-15](https://linear.app/jake-danton/issue/JAK-15) | Pilot corpus packing | Open — note Horner + SBLGNT clean stack |
| [JAK-16](https://linear.app/jake-danton/issue/JAK-16) | ShareAlike on derived outputs | Open |
| [JAK-17](https://linear.app/jake-danton/issue/JAK-17) | Critical Greek / English license | Open |
| [JAK-18](https://linear.app/jake-danton/issue/JAK-18) | Swete vs Rahlfs 1935 | Open — audit leans Swete |
| [JAK-19](https://linear.app/jake-danton/issue/JAK-19) | CCL TEI vs DIOSKOROS | Open — audit leans CCL |

**Decided:** code Apache-2.0 ([JAK-13](https://linear.app/jake-danton/issue/JAK-13)); Sahidic NT baseline Horner / exclude Wells default ([JAK-14](https://linear.app/jake-danton/issue/JAK-14)).
