# Horner × Vaticanus B* pilot alignment notes (JAK-24 / JAK-25)

**Status:** Lightweight Phase 1 lexical/phrase pilot — not a neural aligner.
**Greek spine:** INTF/NTVMR Codex Vaticanus GA 03 (`docID` 20003) **firsthand / B\*** only (JAK-25 GO). Westcott–Hort and SBLGNT are **not** used and must not fill lacunae.
**Sahidic:** CrossWire `CopSahHorner` v1.5 (Public Domain) per JAK-14.
**Pilot file:** [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](../data/sources/pilot/horner-vaticanus-nt-pilot.json)
**Manifests:** [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json), [`data/manifests/ntvmr-vaticanus-b.json`](../data/manifests/ntvmr-vaticanus-b.json)
**Feasibility:** project store `docs/ntvmr-vaticanus-feasibility.md` · Policy: [`source-licenses.md`](source-licenses.md)

## What was verified

| Check | Result |
| --- | --- |
| CopSahHorner | v1.5 PD module; zip pin unchanged from prior JAK-24 work |
| NTVMR API | `transcript/get` teiraw for `docID=20003` without auth |
| B\* selection | `rdg[@type='orig'][@hand='firsthand']` only; empty firsthand = omission |
| Unclear letters | Preserved in original as `[…]` (e.g. John 1:15) |
| Images | **Not** downloaded or packaged |
| License (transcription) | **CC BY 4.0** INTF (TEI `publicationStmt`) |

## Pilot sample

| Passage | Units | Role |
| --- | --- | --- |
| John 1:1–18 | 18 | Prologue; B\* omits corrector supplies (e.g. John 1:4 lacks `των ανθρωπων`) |
| Mark 1:1–8 | 8 | Gospel opening + loanword density |
| John 8:1–11 | 11 | **Empty on both sides** (PA absent in Horner and Vaticanus) |

## Initial lexical / phrase observations

1. **John 1:1** — ⲡϣⲁϫⲉ ↔ ὁ λόγος; ⲡⲛⲟⲩⲧⲉ ↔ `θν` / `θς` (diplomatic nomina sacra retained).
2. **John 1:4** — B\* ends at `το φως`; WH historically had `των ανθρωπων` via a different edition — we do **not** backfill.
3. **Nomina sacra** — Greek keeps abbreviated overlined forms; Coptic keeps overlines in original / strips in normalized.
4. **John 8:1–11** — dual empty slots document shared absence, not an aligner failure.

## Textual-version mismatch classes

| Class | Example | Handling |
| --- | --- | --- |
| Tradition omission (Sahidic) | Empty Horner NRSV slots | Empty Sahidic `original_text` |
| Vaticanus lacuna / PA | John 8:1–11 empty B\* | Empty Greek; **no WH/SBLGNT fill** |
| Firsthand omission vs corrector | John 1:4 corrector1 adds words | Keep empty firsthand; never adopt corrector |
| Eclectic Horner vs B diplomatic | Expected edition mismatch | Flag; do not treat as token-aligner bug alone |

## Offline after setup

1. `uv run thomas sources download` → gitignored `data/downloads/` (Horner zip + NTVMR teiraw; throttled; **no images**).
2. Optional: `uv run thomas sources build-pilot`.
3. Offline: `uv run thomas sources validate`.

## Token / span IDs (JAK-32)

Verse pairs above remain `PilotCorpus` / `ParallelUnit`. Phrase-level stubs that
mint stable token/span/parallel ids (including discontinuous ⲡϣⲁϫⲉ ↔ λογος and
empty John 8:1 spans) are under `tests/fixtures/alignment/`. See
[`id-scheme.md`](id-scheme.md).
