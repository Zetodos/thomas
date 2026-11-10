# Greek–Sahidic alignment pilot benchmark (JAK-28)

**Status:** Baseline measurement for PLAN.md Phases 1–2 — **no pass/fail threshold**.
**Machine report:** [`data/sources/alignment/benchmark-report.json`](../data/sources/alignment/benchmark-report.json) (regenerate with `uv run thomas align benchmark`)
**Splits:** Canonical artifact owned by **JAK-33** under [`data/training/splits/`](../data/training/splits/) (see [`docs/benchmark-holdouts.md`](benchmark-holdouts.md) once #11 is present). Do **not** maintain a second `data/sources/pilot/splits.json`.
**Reference gold:** [`data/sources/alignment/reference/horner-vaticanus-gold-sample-v1.json`](../data/sources/alignment/reference/horner-vaticanus-gold-sample-v1.json) — **provisional** (not adjudicated human review).
**Greek spine:** NTVMR Vaticanus B\* only (GA 03 / `docID` 20003). **Never** mix WH/SBLGNT.

## What was measured

| Item | Value |
| --- | --- |
| Pilot units | 46 |
| Nonempty positive pairs | 35 |
| Omission fixtures (John 8:1–11) | 11 (excluded from positive denominators) |
| Algorithm | Verse-level **type** Dice (once per type per verse) + mild positional prior |
| Scoring coverage | 1-best for every left token (`min_dice=0`); only gold-annotated Sahidic left ids enter precision |
| Provisional gold links scored | 43 (token links; phrase spans expand Cartesian) |
| Predicted links on annotated lefts | 29 |
| True positives | 16 |
| Precision | ≈0.552 (= 16 / 29) |
| Recall | ≈0.372 (= 16 / 43) |
| F1 | ≈0.444 |
| Pass threshold | **None** (not invented) |

When both prediction and gold denominators are nonzero and TP=0, F1 is **0.0** (not null). When there are no predictions, precision/F1 stay undefined (`null`).

Reproduce:

```bash
uv run thomas sources validate
uv run thomas align validate
uv run thomas align benchmark
```

## Gold provenance (provisional)

The reference file is an **agent-authored draft** tagged `provisional-draft-jak28-unreviewed`. It is **not** Human-checked / Human-verified: no named reviewer or adjudication record is attached.

Draft annotation rules:

1. Whitespace tokenize Sahidic/Greek as in `zetodos_thomas.align.tokenize`.
2. Lexical/loanword parallels are 1:1 token spans when lengths match.
3. Multi-token phrase parallels are recorded as spans and scored via **Cartesian link expansion** (a link-expansion metric — **not** independent phrase fidelity).
4. Mark 1:1 `ⲙⲁⲣⲕⲟⲥ` heading is paratext (no lexical gold).
5. Dual-empty John 8:1 is shared-absence.
6. Matthew 13:31 is holdout **smoke-test** gold only (7 links — not a generalizable F1).

Source versions: `ntvmr-vaticanus-b` + `copsahhorner` on pilot `horner-vaticanus-nt-pilot-v2`.

## Null alignments

Scoring emits a 1-best for every alignable left token (`min_dice=0`). Therefore `forced_prediction_coverage=1.0` is a property of the scoring regime, **not** evidence the model learned nulls. The report also gives `null_rate_at_abstention_threshold` (default 0.05) for abstention-style nulls.

## Error categories observed

| Category | Example / note |
| --- | --- |
| Bag co-occurrence false friends | Dice links frequent function words across the verse |
| Bound Coptic groups | Whitespace tokens fuse morphs (`ⲛϭⲓⲡϣⲁϫⲉ`) |
| Punctuation glued to tokens | e.g. `ⲡϣⲁϫⲉ.` vs `λογος` |
| Word-order / null alignments | Mark 1:1 Greek plus vs Horner; Luke kingdom heavens/`θυ` |
| Editorial marks | Must stay on `original_text`; stripped in normalized |
| Paratext | Mark heading must not score as a Greek content word |
| Partial / provisional gold | Unannotated tokens excluded from precision denominator |

## eflomal

`eflomal` is **not implemented** in this PR. Probe-only: `uv run thomas align eflomal-probe`. Native build remains unavailable; not on the default path.

**Jake decision:** whether to add a native eflomal toolchain pin later for Phase 2 comparison (still no LLM on the deterministic path).

## Limitations (Phase 2 gate)

- Provisional gold; small sample; annotated-left coverage only.
- Phrase F1 is Cartesian link-expansion, not phrase fidelity.
- Holdout Matt 13:31 = 7 gold links (smoke test).
- Forced `min_dice=0` coverage ≠ learned null rate.
- No morphology / lemma layer yet.
- Canonical splits owned by JAK-33; this PR consumes them.
- Sibling ingest (JAK-30 / JAK-31) is not absorbed.

## Validation / CI

Offline checks cover unit/source identity, normalization, omission/paratext, and (once wired) JAK-33 split cross-check + `thomas sources validate` / `thomas align benchmark` against the single canonical artifact. See `tests/test_alignment_pilot.py`.
