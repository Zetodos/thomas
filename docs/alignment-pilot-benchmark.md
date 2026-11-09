# Greek–Sahidic alignment pilot benchmark (JAK-28)

**Status:** Baseline measurement for PLAN.md Phases 1–2 — **no pass/fail threshold**.
**Machine report:** [`data/sources/alignment/benchmark-report.json`](../data/sources/alignment/benchmark-report.json) (regenerate with `uv run thomas align benchmark`)
**Splits:** [`docs/alignment-pilot-splits.md`](alignment-pilot-splits.md)
**Human gold:** [`data/sources/alignment/reference/horner-vaticanus-gold-sample-v1.json`](../data/sources/alignment/reference/horner-vaticanus-gold-sample-v1.json)
**Greek spine:** NTVMR Vaticanus B\* only (GA 03 / `docID` 20003). **Never** mix WH/SBLGNT.

## What was measured

| Item | Value |
| --- | --- |
| Pilot units | 46 |
| Nonempty positive pairs | 35 |
| Omission fixtures (John 8:1–11) | 11 (excluded from positive denominators) |
| Algorithm | Verse-bag co-occurrence Dice + mild positional prior |
| Human gold links scored | 43 (token links expanded from lexical/phrase/loanword parallels) |
| Predicted links on gold units | 36 |
| True positives | 9 |
| Precision | 0.25 (= 9 / 36) |
| Recall | ≈0.209 (= 9 / 43) |
| F1 | ≈0.228 |
| Pass threshold | **None** (not invented) |

Reproduce:

```bash
uv run thomas align validate
uv run thomas align benchmark
```

## Human reference sample (not experimental theatre)

Gold encodes inspected correspondences such as:

- John 1:1 — ⲡϣⲁϫⲉ ↔ λογος (phrase); εν αρχη ↔ ϩⲛⲧⲉϩⲟⲩⲉⲓⲧⲉ
- Mark 1:1 — heading `ⲙⲁⲣⲕⲟⲥ` marked **paratext** (no lexical gold); ευαγγελιον loanword; Greek `υιου θυ` null on Horner
- Matthew 7:7 — αιτει / ask–seek–knock; Horner `<ⲟⲩ>` editorial restoration preserved on original
- Luke 6:20 — beatitude / μαθητης loanword; heavens vs `θυ` mismatch noted
- Matthew 13:31 — **holdout** gold only (mustard-seed phrase); never used to fit the baseline
- John 8:1 — shared-absence omission fixture

Qualitative pairs must not be reported as if they were system scores; scores above use explicit denominators.

## Error categories observed

| Category | Example / note |
| --- | --- |
| Bag co-occurrence false friends | Dice links frequent function words across the verse |
| Bound Coptic groups | Whitespace tokens fuse morphs (`ⲛϭⲓⲡϣⲁϫⲉ`) |
| Punctuation glued to tokens | e.g. `ⲡϣⲁϫⲉ.` vs `λογος` |
| Word-order / null alignments | Mark 1:1 Greek plus vs Horner; Luke kingdom heavens/`θυ` |
| Editorial marks | Must stay on `original_text`; stripped in normalized |
| Paratext | Mark heading must not score as a Greek content word |

## eflomal

`eflomal` (PyPI 2.0.0) **failed to build** here (`make -C src eflomal`). It is **not** selected for the deterministic default path. Probe: `uv run thomas align eflomal-probe`.

**Jake decision:** whether to add a native eflomal toolchain pin later for Phase 2 comparison (still no LLM on the deterministic path).

## Limitations

- Small human gold sample; not exhaustive verse annotation.
- No morphology / lemma layer yet.
- Holdout metrics only appear when holdout units are in the gold file (Matthew 13:31 is).
- Thomas Greek fragments 7/17/30/77 are outside this NT training stack (JAK-31/33/34).
- Sibling ingest work (JAK-30 Dilley, JAK-31 fragments) is intentionally not absorbed.

## Validation / CI

Offline checks cover unit/source identity, ordering/uniqueness, normalization recomputation, omission emptiness + notes, editorial-mark stripping, and Mark 1:1 paratext warning. See `tests/test_alignment_pilot.py`.
