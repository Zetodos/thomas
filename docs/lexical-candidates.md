# Coptic→Greek lexical candidates (JAK-37)

**Inputs:** JAK-36 dual-aligner train alignments (`eflomal` + `fast_align`) on
`horner-vaticanus-nt-pilot-v3` / `horner-vaticanus-nt-split-v3` with
`bound-group-v1` Coptic segmentation. Jake decision: keep both aligners
(`phase2_active_aligners=["eflomal","fast_align"]`, `adopted_aligner=null`).

**Artifact:** [`data/sources/lexicon/coptic-greek-lexical-candidates-v1.json`](../data/sources/lexicon/coptic-greek-lexical-candidates-v1.json)
(`columnar-v1` for size; expands to rich models on load).

## Layer separation

| Layer | What it is | What it is not |
| --- | --- | --- |
| `observed_wordform` | Attested surfaces from train symmetrized links | Lemmas or scores |
| `inferred_lemma` | Proxy lemmas via documented heuristics | Morph analyzer output |
| `confidence_estimate` | Separate uncertainty over wordform pairs | An observation count |

Consumers must read the `layer` / `applies_to_layer` fields and
`inference_method` labels. Collapsing layers is a schema violation.

### Inference methods (proxies)

- Coptic: `normalized_surface_as_lemma_proxy_v1` (identity after bound-group tokens)
- Greek: `nfd_strip_combining_marks_v1` (strip accents/breathings only)

Vaticanus B* pilot Greek is largely undiacriticized, so many inferred rows
currently mirror observed surfaces — still a **separate** layer with method
labels for downstream morph upgrades.

### Confidence method

`relative_frequency_aligner_agreement_support_v1` averages available
components only (relative frequency among Coptic candidates, dual-aligner
agreement, observation support, optional JAK-28 Dice). Missing components
stay null; counts are never invented.

## Dual-aligner provenance

Each observed row carries `per_aligner_observation_counts` and
`contributing_aligners`. Headline `observation_count = max(per_aligner)` so
overlapping verse links are not double-counted by summing aligners.

Train-link inputs are also committed beside the table:

- `eflomal-train-links-v1.json`
- `fast_align-train-links-v1.json`

## Leakage controls

| Forbidden | Enforcement |
| --- | --- |
| Dev / holdout / omission as observations | Train-role assert on every link sentence |
| Thomas 7 / P.Oxy. 654 | Unit-id guards + split v3 exclusions |
| Fabricated attestation counts | Validation rejects inconsistent / summed counts |

Optional Dice values reuse the JAK-28 `fit_lexical_baseline` on the **same**
train units as a secondary metric — not a redo of the elementary baseline as
the candidate table.

## Reproduce / validate

```bash
uv run thomas align setup-aligners   # once, if regenerating links
uv run thomas lexicon generate-candidates --seed 36
uv run thomas lexicon validate-candidates
```

Offline validation does not require aligner binaries when the committed table
is present.

No LLM on this path. Out of scope: JAK-38 phrases/loanwords, decoder, Ollama.
