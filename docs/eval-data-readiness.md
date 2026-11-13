# Evaluation-data readiness (JAK-43)

**Goal:** Independently reviewed evaluation materials so Phase 2 reconstruction
([JAK-40](https://linear.app/jake-danton/issue/JAK-40), [JAK-41](https://linear.app/jake-danton/issue/JAK-41))
can score without improvised labels.

**Corpus / split:** [`horner-vaticanus-nt-pilot-v3`](../data/sources/pilot/horner-vaticanus-nt-pilot.json) +
[`horner-vaticanus-nt-split-v3.json`](../data/training/splits/horner-vaticanus-nt-split-v3.json)
**Inventory:** [`horner-vaticanus-nt-v3-inventory.json`](../data/training/horner-vaticanus-nt-v3-inventory.json)
**Blind Thomas 7 pack:** [`thomas-7-blinded-v1.json`](../data/evidence/eval/thomas-7-blinded-v1.json)
**Evidence-type span:** [`thomas-7-17-30-77`](../data/evidence/pilot/thomas-7-17-30-77.json) (JAK-34)

Greek NT spine remains **Vaticanus B\*** only. Sahidic is **CopSahHorner** (PD).

## Review-status vocabulary

| Status | Meaning |
| --- | --- |
| `provisional-draft` | Machine- or author-drafted; **not** human-checked |
| `ready-for-independent-review` | Package complete enough for a reviewer; seals/leakage controls in place |
| `human-checked` | Independent reviewer verified labels / surviving-letter targets |
| `sealed-holdout` | Held out of training/rule selection until scoring |

Do **not** treat `provisional-draft` gold (e.g. JAK-28
`horner-vaticanus-gold-sample-v1`) as human-checked.


## Independent-review follow-up (blocking for JAK-40/41)

The Thomas 7 blinded pack ships at `ready-for-independent-review`. That status
means the package is sealed and checklist-complete — **not** that an independent
reviewer has verified surviving-letter vs restoration scoring decisions.

**JAK-43 formal follow-up (no new Linear task):** obtain and record an actual
independent review (reviewer, date, inspected witness/source version, scoring
decisions), then set `review_status` to `human-checked`. Until then, scoring
entry points for JAK-40/41 **must** call
`zetodos_thomas.eval.review_gate.assert_eval_package_human_checked`, which
raises if the pack is still unreviewed. Do **not** relabel as `human-checked`
without a real review.

## Coptic segmentation (aligner prep)

| Path | Module | Notes |
| --- | --- | --- |
| Horner NT | `zetodos_thomas.align.segment_coptic` (`bound-group-v1`, `ALIGNMENT_READY`) | Peels Sahidic proclitics with a stem/lexicon guard (protects `ⲛⲟⲩⲧⲉ` / `ⲡⲉϫⲉ`); see regression tests |
| Dilley Thomas | SCRIPTORIUM per-token orig↔norm (JAK-30) | **Do not** run the Horner peeler on Dilley units |
| Default lexical baseline | whitespace | Keeps Phase 1 Dice / gold comparable; JAK-36 should use `bound-group-v1` |

## Leakage controls (Thomas 7)

- P.Oxy. **654** (`poxy-654-dclp`) stays `corpus_role=evaluation-witness` on split v3.
- Blind pack seals Greek fragment text until JAK-40 scoring.
- Do not train or select reconstruction rules from Thomas-7 Greek, direct
  Thomas-specific reconstructions, or fragment-derived manual mappings.
- General NT-learned correspondences that happen to predict the same Greek word
  remain permissible (JAK-35 safeguard).

## NT holdout readiness

Split v3 holdouts (**41** nonempty units after Synoptic/phrase-leakage regrouping):

| Passage cluster | Why |
| --- | --- |
| Matthew 10:26–33; Luke 12:2–3 | Hidden/revealed + confess (Synoptic parallels co-held) |
| Luke 8:4–15; Mark 4:4/8/19; Matt 13 sower/kingdom verses | Sower + kingdom parallels co-held so train shares no exact Greek 5-grams |
| Matthew 13:31–33 + selected kingdom verses | Mustard-seed / kingdom zone (Thomas 20) |
| Selected Matt 5 / 7 formula verses | Exact 5-gram overlap with holdout sealed phrases |

Train/holdout must share **no** exact Greek five-token sequences
(`assert_no_train_holdout_greek_ngram_leakage` in holdout validation).

Gold links for these holdouts may begin as `provisional-draft` and must be
promoted to `human-checked` before JAK-41 gate claims. Surviving-letter
protocol for Greek Thomas fragments is separate (blind pack above).

## Corpus size after expansion

| Metric | Phase 1 (v2) | Phase 2 (v3) |
| --- | --- | --- |
| Unit slots | 46 | **407** |
| Nonempty pairs | 35 | **396** |
| Train nonempty | ~31 | **304** |
| Dev nonempty | 2 | **51** |
| Holdout nonempty | 2 | **41** |
| Omission (John 8:1–11) | 11 | **11** |

**Adequacy for JAK-36:** ~11× Phase 1 nonempty pairs in Gospel-first blocks is
enough to **begin** bidirectional statistical aligner learning curves on this
canonical split. Sufficiency / plateau measurement remains JAK-36 — this issue
does not claim full-NT coverage.

## Rebuild / validate

```bash
uv run thomas sources download
uv run thomas sources build-pilot
uv run thomas sources validate
uv run thomas align validate
uv run thomas align benchmark
```

Blinded reconstruction / surviving-letter harness (JAK-40):
[`docs/blinded-reconstruction.md`](blinded-reconstruction.md).
