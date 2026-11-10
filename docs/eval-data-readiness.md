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

## Coptic segmentation (aligner prep)

| Path | Module | Notes |
| --- | --- | --- |
| Horner NT | `zetodos_thomas.align.segment_coptic` (`bound-group-v1`) | Peels common Sahidic proclitics from **normalized** Horner text |
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

Split v3 holdouts (whole passages):

| Passage | Units | Why |
| --- | --- | --- |
| Matthew 13:31–32 | 2 | Mustard-seed (retained from v2 / Thomas 20 zone) |
| Matthew 10:26–33 | 8 | Hidden/revealed + confess sayings |
| Luke 8:4–15 | 12 | Parable of the sower |

Gold links for these holdouts may begin as `provisional-draft` and must be
promoted to `human-checked` before JAK-41 gate claims. Surviving-letter
protocol for Greek Thomas fragments is separate (blind pack above).

## Corpus size after expansion

| Metric | Phase 1 (v2) | Phase 2 (v3) |
| --- | --- | --- |
| Unit slots | 46 | **407** |
| Nonempty pairs | 35 | **396** |
| Train nonempty | ~31 | **323** |
| Dev nonempty | 2 | **51** |
| Holdout nonempty | 2 | **22** |
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
