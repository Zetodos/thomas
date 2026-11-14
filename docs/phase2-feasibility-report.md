# Phase 2 feasibility report and decision gate (JAK-41)

**Status:** Advisory gate input for Jake — **do not** auto-start Phase 3.
**Recommendation:** `pause_and_improve`
**Evaluation-ready:** `false`
**Machine metrics:** [`data/reconstructions/jak41/feasibility-metrics-v1.json`](../data/reconstructions/jak41/feasibility-metrics-v1.json)
**Inputs:** JAK-40 blinded reconstructions on identical Coptic (Thomas 7 + NT holdouts).

## Decision gate

- Evaluation-ready=false — Thomas 7 review_status='ready-for-independent-review' is not human-checked.; Official Thomas 7 surviving-letter scores are unavailable.; NT holdout gold remains 'provisional-draft'.
- Official Thomas 7 surviving-letter scores are gate-blocked (review_status='ready-for-independent-review'); cannot claim G-witness feasibility on a completed human-checked gold.
- NT holdout gold remains 'provisional-draft' — useful for relative decoder-vs-baseline comparison, not as a final OKR.
- NT holdout macro mean letter F1 for decoder≈0.2654 (baseline≈0.1972; delta≈+0.0682) — descriptive only; no approved numeric pass threshold.
- Lemma/inflection accuracy is unmeasured and is not used as an independent gate signal.
- Jake Phase 2 note: corpus/morphology gaps are still the likely limiting factors; absolute recovery is far from production-scale.
- Ollama (JAK-42) stays deferred: limitation is evidence/eval readiness and morphology/corpus depth, not a demonstrated need for LLM fill-in.
- No invented numeric decision cutoffs (PLAN.md / JAK-41); Proceed requires evaluation-ready inputs plus approved criteria.

### Evaluation-ready checklist

- Thomas 7 review_status='ready-for-independent-review' is not human-checked.
- Official Thomas 7 surviving-letter scores are unavailable.
- NT holdout gold remains 'provisional-draft'.

Jake chooses **Proceed / Pause and improve / Reassess**. This report does not invent a numeric OKR or merge Phase 3 work. `proceed` requires evaluation-ready inputs plus approved criteria.

## Official Thomas 7 surviving-letter gate

| Field | Value |
| --- | --- |
| Eval package | `thomas-7-blinded-eval-v1` |
| `review_status` | `ready-for-independent-review` |
| Gate blocked | **true** |
| Freeze present | true |
| Freeze verified | true |
| Official scores | false |

Independent review (reviewer, date, witness/source version, surviving-vs-restoration decisions) remains a **blocking** follow-up under JAK-43 eval readiness. Do not invent `human-checked`. Freeze file existence alone does not make official scores true.

## Identical-input comparison (NT holdouts)

Decoder (JAK-39/40) vs elementary word-for-word train-cooccurrence argmax (JAK-28 lineage) on the **same** Sahidic holdout surfaces. Gold: Vaticanus B* at eval time (`provisional-draft`). Alignment protocol: `token-edit-alignment-v1`.

| Metric | Decoder | Word-for-word |
| --- | --- | --- |
| Units | 41 | 41 |
| Mean letter F1 | 0.2654 | 0.1972 |
| Mean letter precision | 0.3704 | 0.2365 |
| Mean letter recall | 0.2178 | 0.1806 |
| Mean surface-exact recall | 0.2313 | 0.1800 |
| Lemma/inflection accuracy | unmeasured | unmeasured |
| Mean surface-LCS recall | 0.2468 | 0.2053 |
| Mean token coverage | 0.6957 | 0.7513 |
| Mean span coverage | 0.6953 | 0.7513 |
| Mean token abstention rate | 0.3043 | 0.2487 |
| Total additions / omissions / substitutions | 55 / 254 / 369 | 54 / 208 / 449 |
| Construction spans (unlocalized occurrence hits) | 3 (1) | 0 (0) |
| Phrase fidelity | unavailable | unavailable |
| Letter-F1 unit wins | 34 | 7 |

Primary committed ablation arm: `nt_aligner_plus_verified_loanwords`.

## Ablation arms (diagnostic only — not L-evidence)

Regenerated post-gold A/B arms are **uninformative** for the verified-loanword decision unless independently frozen provenance-matched outputs exist. Equal NT metrics alone cannot show whether any holdout exercised a loanword.

| Arm | Mean letter F1 | Δ vs wfw | Units | Informative? | A/B hyp changes | Thomas7 token cov |
| --- | --- | --- | --- | --- | --- | --- |
| `nt_aligner_only` | 0.2654 | 0.0682 | 41 | false | 1 | 0.5952 |
| `nt_aligner_plus_verified_loanwords` | 0.2654 | 0.0682 | 41 | false | 1 | 0.6190 |

## Metric definitions

- **letter_agreement:** Letter F1/precision/recall on edit-aligned token pairs (token-edit-alignment-v1); abstention gaps preserved before alignment. Global LCS is diagnostic only.
- **surface_exact:** Exact surface matches among edit-aligned pairs after letter-fold (casefold + strip diacritics + letters only). Recall = exact hits / gold tokens. Strict index-positional diagnostics are separate.
- **lemma_proxy:** Lemma/inflection accuracy unmeasured: JAK-37 infer_greek_lemma only strips combining marks and duplicates greek_letters_only / surface_exact; cannot detect παραβολη vs παραβολην. Not an independent gate signal.
- **lemma_lcs:** Longest common subsequence length of letter-folded surfaces / gold token count — order-preserving content recovery (not lemma morphology).
- **near_miss_tokens:** Edit-aligned pairs with shared letters but not exact surface match (weak near-miss signal; not morphology accuracy).
- **word_order:** Order-sensitive: edit alignment charges substitutions/insertions/deletions; reordering that loses position lowers aligned surface scores even when surface-LCS stays high.
- **phrase_fidelity:** Actual source-span-localized phrase fidelity is unavailable. construction_phrase_hits reports an unlocalized whole-verse phrase occurrence diagnostic only (not mapping fidelity).
- **additions_omissions:** From token-level edit alignment: insertions (unsupported hyp tokens), deletions (omitted gold tokens), and substitutions. Not length-only sequence-length differences.
- **uncertainty:** Decoder span abstention rate + mean top-1 candidate uncertainty when candidates exist. Coverage is source-token coverage from input_span.token_ids; span_coverage is reported separately.
- **surviving_letter_thomas7:** Official Thomas 7 surviving-letter agreement requires ``review_status=human-checked`` + freeze pin whose digest matches the frozen reconstruction (JAK-40). Gate-blocked status is reported honestly; not invented.

## Failure-mode taxonomy (examples)

### `additions_omissions`

- **matthew.10.28** [decoder]: Lowest letter-F1 holdout unit — edit-alignment insertions/deletions/substitutions.

### `coverage_vs_fidelity`

- **(nt_holdouts)**: Word-for-word baseline covers more Coptic tokens but does not beat decoder letter F1 — coverage ≠ fidelity.

### `false_equivalence`

- **mark.4.4** [decoder]: Frequent Greek function word predicted where gold differs (bag co-occurrence / NT prior risk).

### `fluent_greek_not_success`

- **thomas.7** [decoder]: Fluent or NT-like Greek must not be treated as reconstruction success; surviving-letter / multi-level metrics are authoritative.

### `fragment_divergence`

- **thomas.7**: P.Oxy. 654 Thomas 7 lines mix sparse surviving letters with restorations; restorations stay out of gold. Official agreement deferred until independent review.

### `morphology_underdetermination`

- **(corpus)**: Lemma/inflection accuracy is unmeasured: JAK-37 lemma proxies duplicate surface letter-fold and cannot detect inflection errors (e.g. παραβολη vs παραβολην).

### `nt_overcopy_risk`

- **luke.8.9** [decoder]: High Coptic token coverage with low letter F1 — emitting familiar NT Greek that does not match this holdout.

### `thomas7_side_by_side`

- **thomas.7**: Blinded Thomas 7 decoder vs word-for-word on identical Dilley Coptic; official surviving-letter comparison deferred.

## Leakage safeguards

- Safeguards confirmed: **true** (requires validated packages + complete holdout membership).
- Blinded reconstruct did not consult P.Oxy. 654 / holdout Greek.
- Train/dev not scored as holdout; omission fixtures excluded.
- Dual aligners kept (`eflomal` + `fast_align`); `adopted_aligner=null`.
- No LLM / network on the reconstruction or metrics path.
- Restorations excluded from surviving-letter gold policy (JAK-40).

## Reproduce offline

```bash
uv run thomas eval feasibility-report
uv run thomas eval feasibility-report --skip-ablation-arms \
  -o data/reconstructions/jak41/feasibility-metrics-v1.json \
  --write-doc docs/phase2-feasibility-report.md
```

## Out of scope

Phase 3 architecture; adopting Ollama (JAK-42 deferred); inventing numeric pass thresholds; merging without Jake.
