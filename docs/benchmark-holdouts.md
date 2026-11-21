# Benchmark Holdouts and Zero-Leakage Policy

**Active split (`v1`):** Whole-chapter/passage NT partition over **Horner CopSahHorner × MorphGNT SBLGNT** (`nt-split-v1.json`: `7,782` `train` verses, `51` `dev` verses, `41` `holdout` verses), plus P.Oxy. 1 / 654 / 655 DCLP witness exclusions (`EVALUATION_WITNESS`).

- **Greek NT spine:** `morphgnt-sblgnt` (all 27 NT books, polytonic Koine, gold lemmas + 8-slot morphology).
- **Sahidic NT:** `copsahhorner`.
- **Omission fixtures:** [`data/evidence/omissions/john-8-pa-omission-fixture-v3.json`](../data/evidence/omissions/john-8-pa-omission-fixture-v3.json) (John 8:1–11 Pericope Adulterae excluded from training/scoring).
- **Gated corpora:** Wells `Sahidica`, `NTAEG`, `Bohairica`, and `UD_Coptic` are never downloaded or used.

## Canonical Evaluation & Split Artifacts

| Path | Role |
| --- | --- |
| [`data/training/splits/nt-split-v1.json`](../data/training/splits/nt-split-v1.json) | Authoritative whole-passage NT `train` / `dev` / `holdout` + evaluation-witness exclusions (`N=7874`: `train=7782`, `dev=51`, `holdout=41`) |
| [`data/evidence/fragment-clause-map.json`](../data/evidence/fragment-clause-map.json) | Evaluation-only span alignments (`FragmentClauseMap`) mapping Dilley Coptic morphemes to DCLP P.Oxy. 1/654/655 `FragmentWord` ranges (29 spans across 20 overlapping units; convergent vs `divergent_witness: true`) |
| [`data/evidence/divergences/greek-thomas-fragment-divergences-v1.json`](../data/evidence/divergences/greek-thomas-fragment-divergences-v1.json) | Documented structural/redactional divergences in P.Oxy. vs NHC II |
| [`data/evidence/omissions/john-8-pa-omission-fixture-v3.json`](../data/evidence/omissions/john-8-pa-omission-fixture-v3.json) | John 8:1–11 empty-Coptic omission fixture catalog |
| [`data/generated/eval-report-v1.json`](../data/generated/eval-report-v1.json) | Machine-readable NT holdout + lacuna-aware P.Oxy. evaluation report |
| [`docs/benchmark-report.md`](benchmark-report.md) | Human-readable benchmark tables and side-by-side P.Oxy. comparisons |

## Strict Zero-Leakage Enforcements

Enforced automatically in code and CI (`tests/test_leakage_boundaries.py`, `tests/test_holdouts.py`):

1. **Fragment Blindness:** Packages `morphology`, `lexicon`, `align`, and `reconstruct` may never import `eval` or `fragments`, and may never reference `greek-thomas-fragments`, `fragment-clause-map`, or `dclp/`.
2. **NT Split & Synoptic Scrub:** All 41 `holdout` verses, 51 `dev` verses, AND any `train` verse sharing a Greek 5-gram with `holdout` or `dev` Synoptic passages are excluded from alignment, inflection table building, idiom/valency extraction, and Koine LM training (`model build`).
3. **Morphology Holdout:** `CopticAnalyzer` trains strictly on `sahidic.ot_TT.zip` + `CCL v1.2` (zero Thomas, holding out `Ruth` and `Jonah` for intrinsic segmentation/lemmatization scoring).
4. **Decoder Tuning:** `tune_weights_on_dev` tunes log-linear weights exclusively on the 51-verse `dev` partition.
5. **Lacuna Honesty:** Purely editorial `[lacuna]` restorations (`surviving_text == ""`) contribute zero gold lemma or character counts in `eval/benchmark.py`; truncated words (`is_truncated=True`) score only surviving letters.

Validation commands:

```bash
uv run thomas sources validate
uv run thomas eval run
uv run pytest tests/test_leakage_boundaries.py tests/test_holdouts.py tests/test_eval_lacuna.py
```
