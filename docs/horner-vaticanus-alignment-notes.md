# Horner × Vaticanus B* pilot alignment notes (JAK-24 / JAK-25 / JAK-28)

**Status:** Phase 1–2 lexical/phrase alignment pilot with quantified baseline (JAK-28).
**Greek spine:** INTF/NTVMR Codex Vaticanus GA 03 (`docID` 20003) **firsthand / B\*** only (JAK-25 GO). Westcott–Hort and SBLGNT are **not** used and must not fill lacunae.
**Sahidic:** CrossWire `CopSahHorner` v1.5 (Public Domain) per JAK-14.
**Corpus file:** [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](../data/sources/pilot/horner-vaticanus-nt-pilot.json) (`horner-vaticanus-nt-pilot-v3`; JAK-43)
**Canonical splits (JAK-43):** [`eval-data-readiness.md`](eval-data-readiness.md) · [`data/training/splits/horner-vaticanus-nt-split-v3.json`](../data/training/splits/horner-vaticanus-nt-split-v3.json)
**Archival Phase 1:** [`horner-vaticanus-nt-pilot-v2.json`](../data/sources/pilot/horner-vaticanus-nt-pilot-v2.json) · [`horner-vaticanus-nt-split-v2.json`](../data/training/splits/horner-vaticanus-nt-split-v2.json)
**Consumer notes / benchmark:** [`alignment-pilot-splits.md`](alignment-pilot-splits.md) · [`alignment-pilot-benchmark.md`](alignment-pilot-benchmark.md)
**Manifests:** [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json), [`data/manifests/ntvmr-vaticanus-b.json`](../data/manifests/ntvmr-vaticanus-b.json)

## What was verified

| Check | Result |
| --- | --- |
| CopSahHorner | v1.5 PD module; zip pin unchanged from prior JAK-24 work |
| NTVMR API | `transcript/get` teiraw for `docID=20003` without auth |
| B\* selection | `rdg[@type='orig'][@hand='firsthand']` only; empty firsthand = omission |
| Unclear letters | Preserved in original as `[…]` (e.g. John 1:15) |
| Horner editorial restorations | HTML entities unescaped; `<>` kept on original, stripped in normalized |
| Images | **Not** downloaded or packaged |
| License (transcription) | **CC BY 4.0** INTF (TEI `publicationStmt`) |

## Pilot sample (JAK-28)

| Passage | Units | Role |
| --- | --- | --- |
| John 1:1–18 | 18 | train — prologue |
| Mark 1:1–8 | 8 | train — opening + loanwords; Mark 1:1 heading paratext |
| Matthew 5:14–16 | 3 | train — light sayings |
| Matthew 7:7–8 | 2 | train — imperatives |
| Luke 6:20–21 | 2 | dev — beatitudes |
| Matthew 13:31–32 | 2 | **holdout** — mustard-seed parable |
| John 8:1–11 | 11 | **omission** fixture (empty both sides) |

**v3 (JAK-43):** 407 units · 396 nonempty positive · 11 omission.
**v2 archival:** 46 units · 35 nonempty · 11 omission.

## Lexical / phrase observations (illustrative — not scores)

1. **John 1:1** — ⲡϣⲁϫⲉ ↔ ὁ λόγος; ⲡⲛⲟⲩⲧⲉ ↔ `θν` / `θς` (diplomatic nomina sacra retained). Encoded in human gold; scored only via denominators in the benchmark report.
2. **John 1:4** — B\* ends at `το φως`; do **not** backfill from WH.
3. **Mark 1:1** — Sahidic book heading `ⲙⲁⲣⲕⲟⲥ` is **paratext**; Greek `υιου θυ` absent in Horner.
4. **Matthew 7:7** — loanword `ⲁⲓⲧⲉⲓ`; Horner `<ⲟⲩ>` restoration on open-verb form.
5. **John 8:1–11** — dual empty slots document shared absence, not an aligner failure.

## Textual-version mismatch classes

| Class | Example | Handling |
| --- | --- | --- |
| Tradition omission (Sahidic) | Empty Horner NRSV slots | Empty Sahidic `original_text` |
| Vaticanus lacuna / PA | John 8:1–11 empty B\* | Empty Greek; **no WH/SBLGNT fill** |
| Firsthand omission vs corrector | John 1:4 corrector1 adds words | Keep empty firsthand; never adopt corrector |
| Eclectic Horner vs B diplomatic | Mark 1:1 plus; Luke 6:20 heavens/`θυ` | Flag; do not treat as token-aligner bug alone |
| Paratext | Mark 1:1 `ⲙⲁⲣⲕⲟⲥ` | Exclude from lexical scoring |

## Offline after setup

1. `uv run thomas sources download` → gitignored `data/downloads/` (Horner zip + NTVMR teiraw; throttled; **no images**).
2. Optional: `uv run thomas sources build-pilot`.
3. Offline: `uv run thomas sources validate` · `uv run thomas align validate` · `uv run thomas align benchmark` (all against the canonical JAK-33 split).
