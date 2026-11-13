# Phase 2 feasibility report and decision gate (JAK-41)

**Status:** Advisory gate input for Jake — **do not** auto-start Phase 3.
**Recommendation:** `pause_and_improve`
**Machine metrics:** [`data/reconstructions/jak41/feasibility-metrics-v1.json`](../data/reconstructions/jak41/feasibility-metrics-v1.json)
**Inputs:** JAK-40 blinded reconstructions on identical Coptic (Thomas 7 + NT holdouts).

## Decision gate

- Official Thomas 7 surviving-letter scores are gate-blocked (review_status='ready-for-independent-review'); cannot claim G-witness feasibility on a completed human-checked gold.
- NT holdout gold remains 'provisional-draft' — useful for relative decoder-vs-baseline comparison, not as a final OKR.
- NT holdout macro mean letter F1 for decoder≈0.1187 (baseline≈0.1029; delta≈+0.0158).
- Lemma-proxy recall remains low (decoder≈0.0567; baseline≈0.0518).
- Jake Phase 2 note: corpus/morphology gaps are still the likely limiting factors; absolute recovery is far from production-scale.
- Decoder is only marginally better than the elementary word-for-word baseline on identical inputs — not yet a clear architecture win.
- Ollama (JAK-42) stays deferred: limitation is evidence/eval readiness and morphology/corpus depth, not a demonstrated need for LLM fill-in.

Jake chooses **Proceed / Pause and improve / Reassess**. This report does not invent a numeric OKR or merge Phase 3 work.

## Official Thomas 7 surviving-letter gate

| Field | Value |
| --- | --- |
| Eval package | `thomas-7-blinded-eval-v1` |
| `review_status` | `ready-for-independent-review` |
| Gate blocked | **true** |
| Freeze present | true |
| Official scores | false |

Independent review (reviewer, date, witness/source version, surviving-vs-restoration decisions) remains a **blocking** follow-up under JAK-43 eval readiness. Do not invent `human-checked`.

## Identical-input comparison (NT holdouts)

Decoder (JAK-39/40) vs elementary word-for-word train-cooccurrence argmax (JAK-28 lineage) on the **same** Sahidic holdout surfaces. Gold: Vaticanus B* at eval time (`provisional-draft`).

| Metric | Decoder | Word-for-word |
| --- | --- | --- |
| Units | 41 | 41 |
| Mean letter F1 | 0.1187 | 0.1029 |
| Mean letter precision | 0.1584 | 0.1174 |
| Mean letter recall | 0.0949 | 0.0920 |
| Mean surface-exact recall | 0.0567 | 0.0518 |
| Mean lemma-proxy recall | 0.0567 | 0.0518 |
| Mean lemma-LCS recall | 0.2468 | 0.2053 |
| Mean coverage | 0.6922 | 0.7513 |
| Mean abstention rate | 0.3047 | 0.2487 |
| Total additions / omissions | 21 / 220 | 32 / 186 |
| Construction spans (hits) | 3 (1) | 0 (0) |
| Letter-F1 unit wins | 21 | 20 |

Primary committed ablation arm: `nt_aligner_plus_verified_loanwords`.

## Ablation arms (decoder only)

| Arm | Mean letter F1 | Δ vs word-for-word | Units |
| --- | --- | --- | --- |
| `nt_aligner_only` | 0.1187 | 0.0158 | 41 |
| `nt_aligner_plus_verified_loanwords` | 0.1187 | 0.0158 | 41 |

## Metric definitions

- **letter_agreement:** Token-positional Greek letter F1/precision/recall (``token-positional-letter-v1``). Global LCS is diagnostic only.
- **surface_exact:** Positional exact match after letter-fold (casefold + strip diacritics + letters only). Recall = exact hits / gold tokens.
- **lemma_proxy:** Positional match of JAK-37 ``infer_greek_lemma`` proxies (NFD strip combining marks). Not MorphGNT lemmas.
- **lemma_lcs:** Longest common subsequence length of lemma-proxy sequences / gold token count — order-preserving content recovery beyond strict index.
- **near_miss_tokens:** Positional pairs with shared letters but not exact surface match (weak morphology / near-miss signal under the letter-fold proxy).
- **word_order:** Captured jointly by positional lemma_proxy recall and lemma_lcs recall; fluent reordering that loses position lowers positional scores even when LCS stays high.
- **phrase_fidelity:** For multi-token JAK-38 construction spans: whether the predicted Greek surface sequence appears as a contiguous gold subsequence (letter-folded). Baseline is always singleton 1:1.
- **additions_omissions:** additions = unpaired hypothesis tokens; omissions = unpaired gold tokens (length mismatch after flattening non-null predictions).
- **uncertainty:** Decoder abstention rate + mean top-1 candidate uncertainty when candidates exist. Baseline uncertainty is coverage-only (null = unseen).
- **surviving_letter_thomas7:** Official Thomas 7 surviving-letter agreement requires ``review_status=human-checked`` + freeze pin (JAK-40). Gate-blocked status is reported honestly; not invented.

## Failure-mode taxonomy (examples)

### `additions_omissions`

- **luke.8.4** [decoder]: Lowest letter-F1 holdout unit — length/content mismatch.

### `coverage_vs_fidelity`

- **(nt_holdouts)**: Word-for-word baseline covers more Coptic surfaces but does not beat decoder letter F1 — coverage ≠ fidelity.

### `false_equivalence`

- **mark.4.4** [decoder]: Frequent Greek function word predicted where gold differs (bag co-occurrence / NT prior risk).

### `fluent_greek_not_success`

- **thomas.7** [decoder]: Fluent or NT-like Greek must not be treated as reconstruction success; surviving-letter / multi-level metrics are authoritative.

### `fragment_divergence`

- **thomas.7**: P.Oxy. 654 Thomas 7 lines mix sparse surviving letters with restorations; restorations stay out of gold. Official agreement deferred until independent review.

### `morphology_underdetermination`

- **(corpus)**: JAK-37 lemma proxies are accent-stripped surfaces, not MorphGNT lemmas; surface_exact and lemma_proxy often coincide, so inflection errors are under-measured.

### `nt_overcopy_risk`

- **mark.4.4** [decoder]: High Coptic coverage with low letter F1 — emitting familiar NT Greek that does not match this holdout.

### `thomas7_side_by_side`

- **thomas.7**: Blinded Thomas 7 decoder vs word-for-word on identical Dilley Coptic; official surviving-letter comparison deferred.

## Leakage safeguards (confirmed)

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
