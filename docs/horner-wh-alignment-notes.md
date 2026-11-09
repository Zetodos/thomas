# Horner × Westcott–Hort pilot alignment notes (JAK-24)

**Status:** Lightweight Phase 1 lexical/phrase pilot — not a neural aligner.
**Greek spine:** Westcott–Hort 1881 (Robinson PD transcription) per [JAK-15](https://linear.app/jake-danton/issue/JAK-15) — **not** SBLGNT.
**Sahidic:** CrossWire `CopSahHorner` v1.5 (Public Domain) per [JAK-14](https://linear.app/jake-danton/issue/JAK-14).
**Pilot file:** [`data/sources/pilot/horner-wh-nt-pilot.json`](../data/sources/pilot/horner-wh-nt-pilot.json)
**Manifests:** [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json), [`data/manifests/westcott-hort.json`](../data/manifests/westcott-hort.json)
**Audit:** [`licensing-provenance-audit.md`](licensing-provenance-audit.md) §4 · Policy: [`source-licenses.md`](source-licenses.md)

## What was verified

| Check | Result |
| --- | --- |
| CrossWire module | `CopSahHorner` **v1.5** (`SwordVersionDate=2009-11-01`) |
| Distribution license | **Public Domain** on ModInfo, copyright page, and `copsahhorner.conf` |
| Download artifact | `CopSahHorner.zip` from CrossWire ftpmirror; `sha256:4f074891e60cab8e59cb8ebaa64cb207ae990ae78270df5be290fb0a8e7d2132` |
| Encoding / characters | UTF-8 Coptic in ztext export; combining overlines on nomina sacra preserved in **original** layer |
| Word boundaries | Space-separated tokens in module export; no SCRIPTORIUM-style gold tokenization |
| Textual coverage | Full NT book list present; empty slots for John 7:53–8:11 and Mark 16:9–20 |
| Greek counterpart | Robinson WH `textonly/*.WH` at commit `91473892…`; beta→Unicode in ingest |

## Pilot sample

| Passage | Units | Role |
| --- | --- | --- |
| John 1:1–18 | 18 | High-signal prologue (λόγος / ⲡϣⲁϫⲉ) |
| Mark 1:1–8 | 8 | Gospel opening + loanword density |
| John 8:1–11 | 11 | Documented **empty Sahidic** NRSV slots vs Greek tradition |

Each unit keeps **original** and **normalized** forms for both languages (Coptic normalization strips combining overlines/binders and collapses whitespace; Greek is NFC + whitespace).

## Initial lexical / phrase observations

These are qualitative notes from inspecting the pilot — not scored alignment metrics.

1. **John 1:1** — Clear lexical anchors: ⲡϣⲁϫⲉ ↔ ὁ λόγος; ⲡⲛⲟⲩⲧⲉ ↔ τὸν θεόν / θεός; ϩⲛ̅ⲧⲉϩⲟⲩⲉ͡ⲓⲧⲉ ↔ ἐν ἀρχῇ. Coptic uses second-tense / preterite morphology (`ⲛⲉϥϣⲟⲟⲡ`) where Greek uses ἦν — phrase-level, not 1:1 token map.
2. **Nomina sacra** — Sahidic `ⲓ̅ⲥ̅`, `ⲡⲉⲭ̅ⲥ̅` retain overlines in original text; normalized form drops combining marks for matching. Greek WH textonly has no NS contractions (full Ιησους / Χριστος forms appear where present).
3. **Loanwords** — Mark 1 opening shows expected Greek loans in Sahidic (e.g. ⲉⲩⲁⲅⲅⲉⲗⲓⲟⲛ) that should align strongly once lexicon (CCL/DDGLC) is wired.
4. **Particle / article asymmetry** — Greek articles and particles often lack direct Coptic counterparts; expect many-to-one and null alignments in any automatic word aligner.

## Textual-version mismatch classes

| Class | Example in pilot | Handling |
| --- | --- | --- |
| **Tradition omission** | John 8:1–11 empty in `CopSahHorner`; Greek WH has the pericope | Keep empty Sahidic `original_text=""`; record note on the parallel unit; do not invent Coptic |
| **Longer Mark ending** | Mark 16:9–20 empty in module (outside this pilot range) | Same empty-slot policy when those verses are imported |
| **NRSV versification force** | Module `Versification=NRSV` | Verse IDs follow NRSV slots; Sahidic boundaries may merge/split vs print Horner |
| **Edition mismatch (WH vs Sahidic Vorlage)** | Eclectic Horner MSS ≠ WH Alexandrian print | Flag as textual-version mismatch when glosses diverge despite good lexical overlap; do not treat as aligner failure alone |
| **Historical misnumbering** | Fixed in module v1.5; residual risk vs older dumps | Pin v1.5 + zip checksum; refuse unlabeled mirrors |

## Wells corpora

Sahidica / NTAEG / Bohairica remain **out of default CLI/CI**. `thomas sources validate` rejects those source ids if present under `data/manifests/`. Defer any Wells permission ask until after further Horner quality measurement (JAK-14 / JAK-24).

## Offline after setup

1. Download `CopSahHorner.zip` into `data/downloads/` and unpack to `data/downloads/sword/` (gitignored).
2. Fetch Robinson WH `textonly/` books into `data/downloads/westcott-hort/textonly/` (gitignored).
3. Optional regenerate: `uv run thomas sources build-pilot` (requires `pysword`, via `--extra sword` or the dev group).
4. Offline validate: `uv run thomas sources validate` — uses committed manifests + pilot only; **no network**.
