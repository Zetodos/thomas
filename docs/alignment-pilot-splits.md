# Alignment pilot passage splits (JAK-28 consumer notes)

**Canonical owner:** **JAK-33** — [`data/training/splits/`](../data/training/splits/) (see [`docs/benchmark-holdouts.md`](benchmark-holdouts.md) on #11).
**Do not** maintain a second competing policy at `data/sources/pilot/splits.json`.
**Corpus:** [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](../data/sources/pilot/horner-vaticanus-nt-pilot.json) (`horner-vaticanus-nt-pilot-v2`)
**Greek:** NTVMR Vaticanus B\* only (GA 03 / `docID` 20003) — never WH/SBLGNT
**Sahidic:** CopSahHorner v1.5 (Public Domain)

Agreed whole-passage roles for the v2 46-unit pilot (encoded in the JAK-33 artifact once updated):

| Role | Passage | Units | Why |
| --- | --- | --- | --- |
| **train** | John 1:1–18 | 18 | Prologue; nomina sacra; sayings-like density |
| **train** | Mark 1:1–8 | 8 | Gospel opening; loanwords; Mark 1:1 heading **paratext** |
| **train** | Matthew 5:14–16 | 3 | Light/city sayings (Thomas-adjacent imagery) |
| **train** | Matthew 7:7–8 | 2 | Imperatives ask/seek/knock (Thomas 2/92/94 zone) |
| **dev** | Luke 6:20–21 | 2 | Beatitudes; pronouns; clause variation (Thomas 54 zone) |
| **holdout** | Matthew 13:31–32 | 2 | Mustard-seed parable whole-passage holdout (Thomas 20 zone) |
| **omission** (separate catalog) | John 8:1–11 | 11 | Dual-empty PA; negative/omission fixture only |

**Totals:** 46 units · 35 nonempty positive pairs · 11 omission slots.

## Textual / versification notes

| Issue | Handling |
| --- | --- |
| Mark 1:1 Sahidic `ⲙⲁⲣⲕⲟⲥ` book heading | Paratext; not a Greek verse token |
| Mark 1:1 Greek `υιου θυ` vs shorter Horner | Edition mismatch / null on Sahidic — not an aligner bug alone |
| Matt 7:7 Horner `<ⲟⲩ>` restoration | Kept on `original_text`; stripped in `normalized_text` |
| John 1:15 Greek `[η]` unclear | Editorial mark on original; normalized drops wrappers |
| John 8:1–11 empty both sides | Omission fixture catalog (JAK-33); never positive training/eval denominator |
| Luke 6:20 heavens vs `θυ` | Documented tradition difference |

## Leakage controls

- Greek Thomas fragment witnesses (notably logia **7 / 17 / 30 / 77**) are **not** in this NT stack (JAK-31 / JAK-33 / JAK-34).
- Coptic Thomas (Dilley) is **not** absorbed here (JAK-30).
- Holdout Matthew 13:31–32 must not be used when fitting the lexical baseline.
- JAK-33 `PilotSplits`/`NtCorpusSplit` rejects overlapping unit assignments and validates `pilot_corpus_id`.

## Rebuild / integrated check

```bash
uv run thomas sources download
uv run thomas sources build-pilot
uv run thomas sources validate
uv run thomas align validate
uv run thomas align benchmark
```
