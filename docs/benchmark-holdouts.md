# Benchmark holdouts and leakage policy (JAK-33)

**Status:** Written train / dev / holdout policy for the Horner × Vaticanus B\* pilot.
**Greek NT:** Vaticanus B\* (`ntvmr-vaticanus-b`) only — no Westcott–Hort / SBLGNT lacuna fill.
**Sahidic NT:** `copsahhorner`.
**Consumers:** JAK-28 (alignment benchmarks). This PR owns the split files; JAK-28 must not invent a competing NT split path.

## Artifacts (offline)

| Path | Role |
| --- | --- |
| [`data/training/splits/horner-vaticanus-nt-split-v1.json`](../data/training/splits/horner-vaticanus-nt-split-v1.json) | Whole-passage train / dev / holdout + evaluation-witness exclusions |
| [`data/evidence/omissions/john-8-pa-omission-fixture-v1.json`](../data/evidence/omissions/john-8-pa-omission-fixture-v1.json) | Empty/omission fixtures **separate** from positive lexical pairs |
| [`data/evidence/divergences/greek-thomas-fragment-divergences-v1.json`](../data/evidence/divergences/greek-thomas-fragment-divergences-v1.json) | Greek Thomas fragment vs Coptic sequence/edition divergences |
| [`docs/greek-thomas-fragment-divergences.md`](greek-thomas-fragment-divergences.md) | Human-readable divergence notes |

Schema models: `zetodos_thomas.schemas.holdout`. Validation runs offline via `uv run thomas sources validate` (and `tests/test_holdouts.py`).

## NT split (whole passages / books)

| Role | Passage | Unit ids | Notes |
| --- | --- | --- | --- |
| **train** | John 1:1–18 | `john.1.1`–`john.1.18` | Nonempty pilot pairs for mapping observations |
| **dev** | *(empty / reserved)* | — | Present in schema; no units until JAK-28 adds a whole new nonempty passage |
| **holdout** | Mark 1:1–8 | `mark.1.1`–`mark.1.8` | Whole-passage reconstruction holdout |

**Not in train/dev/holdout:** John 8:1–11 — see omission fixtures.

**Secondary OT:** not used in v1 (`secondary_ot_used: false`).

Holdouts are **entire passages**, never random isolated words. When evaluating recovery, exclude holdout unit ids (and evaluation witnesses below) from training mappings / phrase tables.

## Omission fixtures (separate catalog)

John 8:1–11 are dual-empty on Horner and Vaticanus B\* (pericope adulterae absent). They live only in [`john-8-pa-omission-fixture-v1.json`](../data/evidence/omissions/john-8-pa-omission-fixture-v1.json):

- Do **not** count toward positive lexical/phrase precision or recall denominators.
- Do **not** treat as aligner failure; they document shared absence.
- Do **not** fill empty firsthand Greek with another edition.

## Evaluation-witness exclusions (Greek Thomas fragments)

Coordinate with JAK-28 leakage controls — do **not** train Greek–Sahidic mappings on these witnesses when they are intended for withheld reconstruction evaluation (notably Thomas 7 / 17 / 30 / 77 via JAK-34):

| Witness id | Source id | Manuscript |
| --- | --- | --- |
| `poxy-654-dclp` | `dclp-poxy-654` | P.Oxy. 654 |
| `poxy-1-dclp` | `dclp-poxy-1` | P.Oxy. 1 |
| `poxy-655-dclp` | `dclp-poxy-655` | P.Oxy. 655 |

Ids match JAK-31 corpus naming. This file remains valid offline as policy even before JAK-30/31 merge; it does not vendor their TEI/ingest trees.

## Ownership vs JAK-28

- **JAK-33** owns `data/training/splits/` policy JSON and the omission / divergence catalogs above.
- **JAK-28** consumes them for benchmarks; if JAK-28 later expands the nonempty pilot, bump the split schema / id rather than silently reassigning Mark 1 or John 1 roles.
- Full token/phrase alignment benchmarks remain JAK-28 scope (not duplicated here).

## Jake decisions

1. **Dev empty in v1** — with only two nonempty pilot passages, Mark 1:1–8 is holdout; confirm whether an early JAK-28 expansion should fill `dev` first.
2. **Fragment source ids** — confirm `dclp-poxy-*` / `dilley-thomas` naming once JAK-30/31 land (placeholders must not silently rename).
