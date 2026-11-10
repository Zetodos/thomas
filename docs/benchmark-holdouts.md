# Benchmark holdouts and leakage policy (JAK-33)

**Status:** Canonical train / dev / holdout policy for the Horner × Vaticanus B\* **v2** pilot (46 units).
**Greek NT:** Vaticanus B\* (`ntvmr-vaticanus-b`) only — no Westcott–Hort / SBLGNT lacuna fill.
**Sahidic NT:** `copsahhorner`.
**Consumers:** JAK-28 (alignment benchmarks on PR #13). **JAK-33 owns this file**; JAK-28 must not keep a second policy under `data/sources/pilot/splits.json`.

## Canonical artifact (offline)

| Path | Role |
| --- | --- |
| [`data/training/splits/horner-vaticanus-nt-split-v2.json`](../data/training/splits/horner-vaticanus-nt-split-v2.json) | **Authoritative** whole-passage train / dev / holdout + evaluation-witness exclusions |
| [`data/evidence/omissions/john-8-pa-omission-fixture-v1.json`](../data/evidence/omissions/john-8-pa-omission-fixture-v1.json) | Empty/omission fixtures **separate** from positive lexical pairs |
| [`data/evidence/divergences/greek-thomas-fragment-divergences-v1.json`](../data/evidence/divergences/greek-thomas-fragment-divergences-v1.json) | Greek Thomas fragment vs Coptic sequence/edition divergences |
| [`docs/greek-thomas-fragment-divergences.md`](greek-thomas-fragment-divergences.md) | Human-readable divergence notes |

Schema: `zetodos_thomas.schemas.holdout.NtCorpusSplit`. Validation: `uv run thomas sources validate` (+ `tests/test_holdouts.py`).

`pilot_corpus_id` is pinned to **`horner-vaticanus-nt-pilot-v2`** (46 units / 35 nonempty / 11 omission).

## NT split v2 (whole passages)

| Role | Passage | Unit ids |
| --- | --- | --- |
| **train** | John 1:1–18 | `john.1.1`–`john.1.18` |
| **train** | Mark 1:1–8 | `mark.1.1`–`mark.1.8` |
| **train** | Matthew 5:14–16 | `matthew.5.14`–`matthew.5.16` |
| **train** | Matthew 7:7–8 | `matthew.7.7`–`matthew.7.8` |
| **dev** | Luke 6:20–21 | `luke.6.20`–`luke.6.21` |
| **holdout** | Matthew 13:31–32 | `matthew.13.31`–`matthew.13.32` |

**Not in train/dev/holdout:** John 8:1–11 — omission fixtures only.

**Secondary OT:** not used (`secondary_ot_used: false`).

## Omission fixtures (separate catalog)

John 8:1–11 dual-empty PA slots live only in the omission catalog. Do not score as positive lexical/phrase pairs; do not assign them to train/dev/holdout.

## Evaluation-witness exclusions (Greek Thomas fragments)

| Witness id | Source id | Manuscript |
| --- | --- | --- |
| `poxy-654-dclp` | `dclp-poxy-654` | P.Oxy. 654 |
| `poxy-1-dclp` | `dclp-poxy-1` | P.Oxy. 1 |
| `poxy-655-dclp` | `dclp-poxy-655` | P.Oxy. 655 |

## Ownership vs JAK-28 (#13)

1. **Read** [`horner-vaticanus-nt-split-v2.json`](../data/training/splits/horner-vaticanus-nt-split-v2.json) (and the omission / divergence catalogs).
2. **Delete** any duplicate `data/sources/pilot/splits.json` policy once this lands.
3. Map `passages[].role` + `unit_ids` / book–verse ranges into the aligner partitioner; treat the omission catalog as the sole John 8 PA negative set.
4. Run `thomas sources validate` and `thomas align benchmark --splits <canonical-path>` (or default to this path).

## Jake decisions

1. Confirm Luke 6:20–21 as **dev** and Matthew 13:31–32 as **holdout** (adopted from #13 design).
2. Confirm Mark stays in **train** (not holdout) for the v2 pilot.
