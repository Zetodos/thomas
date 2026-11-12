# Alignment passage splits (JAK-43 consumer notes)

**Canonical owner:** **JAK-43** — [`data/training/splits/horner-vaticanus-nt-split-v3.json`](../data/training/splits/horner-vaticanus-nt-split-v3.json)
(see also [`docs/eval-data-readiness.md`](eval-data-readiness.md)).
**Archival Phase 1:** [`horner-vaticanus-nt-split-v2.json`](../data/training/splits/horner-vaticanus-nt-split-v2.json) (JAK-33) +
[`horner-vaticanus-nt-pilot-v2.json`](../data/sources/pilot/horner-vaticanus-nt-pilot-v2.json).
**Do not** maintain a second competing policy at `data/sources/pilot/splits.json`.
**Corpus:** [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](../data/sources/pilot/horner-vaticanus-nt-pilot.json) (`horner-vaticanus-nt-pilot-v3`)
**Greek:** NTVMR Vaticanus B\* only (GA 03 / `docID` 20003) — never WH/SBLGNT
**Sahidic:** CopSahHorner v1.5 (Public Domain)

## v3 role summary (407 units)

| Role | Passages (summary) | Nonempty units |
| --- | --- | --- |
| **train** | Gospel blocks minus holdout leakage counterparts | **304** |
| **dev** | Luke 6:20–49; John 3:1–21 | **51** |
| **holdout** | Matt 10:26–33; Luke 8:4–15; Luke 12:2–3; Synoptic/phrase-leakage counterparts (Mark 4 / Matt 5·7·13 verses) | **41** |
| **omission** (separate catalog) | John 8:1–11 | 11 dual-empty |

**Totals:** 407 units · 396 nonempty positive pairs · 11 omission slots (~11× Phase 1 nonempty).

## Leakage controls

- Train and holdout share **no** exact Greek five-token sequences (Synoptic parallel / phrase leakage regression in `tests/test_holdouts.py` + `assert_no_train_holdout_greek_ngram_leakage`).

- Greek Thomas fragment witnesses (P.Oxy. 654 / 1 / 655; logia **7 / 17 / 30 / 77**) are **not** in this NT stack.
- Coptic Thomas (Dilley) is **not** absorbed here.
- Holdout passages must not be used when fitting lexical / statistical aligners.
- Blind Thomas 7 pack: [`data/evidence/eval/thomas-7-blinded-v1.json`](../data/evidence/eval/thomas-7-blinded-v1.json).

## Coptic segmentation

Use `zetodos_thomas.align.segment_coptic` (`bound-group-v1`) for Horner NT aligner
prep. Default Dice baseline remains whitespace for Phase 1 gold comparability.

## Rebuild / integrated check

```bash
uv run thomas sources download
uv run thomas sources build-pilot
uv run thomas sources validate
uv run thomas align validate
uv run thomas align benchmark
```
