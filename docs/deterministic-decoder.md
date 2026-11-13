# Minimal deterministic decoder (JAK-39)

Phase 2 feasibility decoder: ranked multi-candidate Greek hypotheses from
frozen JAK-37/38 tables, with explicit uncertainty and abstention. No network.
No LLM on the decode path.

## Inputs

| Surface | Source |
| --- | --- |
| Coptic span ids + token ids + surfaces | JAK-32 / JAK-38 construction examples |
| Optional `construction_id` | JAK-38 statistical constructions (**hint**, not exclusive override) |
| Ablation arm | JAK-38 `nt_aligner_only` or `nt_aligner_plus_verified_loanwords` |

Request schema: `zetodos_thomas.schemas.decode.DecodeRequest`.

`construction_id` must exist and match the span’s Coptic pattern or the span
abstains (`construction_id_unknown` / `construction_pattern_mismatch`). A
valid hint **boosts** that construction in ranking but still gathers all other
applicable constructions, observed wordforms, and (arm B) loanword clues.

Request / span / source / token IDs must agree. P.Oxy. fragment handles are
rejected at every level (`forbidden_query_unit` / `input_id_inconsistent`).

## Evidence (do not collapse)

| Kind | Evidence class | Artifact | `form_kind` |
| --- | --- | --- | --- |
| `statistical_construction` | `statistical` | phrase constructions JSON | `attested_surface` |
| `lexical_observed_wordform` | — (lexical layer) | lexical candidates JSON | `attested_surface` |
| `independently_verified_loanword` | `independently_verified` | CCL/DDGLC loanwords (arm B) | `lemma_clue` |

Aligner-derived pairs are never reclassified as verified loanwords. Loanword
`illustrative_nt_span_ids` are not training exemplars and are not consumed.

Dual aligners stay active: `phase2_active_aligners=["eflomal","fast_align"]`,
`adopted_aligner=null`.

Load path fails closed: JAK-37/38 validators + shared provenance
(`pilot_corpus_id`, `split_id`, source IDs, `coptic_segmentation`,
`bitext_checksum_sha256`) must agree. Decode output records immutable
`artifact_digests` and `evidence_provenance`.

## Ranking (`jak39-stable-ranking-v1`)

Identical inputs → identical outputs. Ascending key:

1. `construction_id` hint boost
2. Evidence tier: construction < **observed wordform** < verified loanword lemma
3. Dual-aligner agreement preferred
4. Higher `observation_count`
5. Higher `heuristic_ranking_score` when present
6. Shorter Greek surface sequences
7. Lexicographic Greek surfaces
8. Lexicographic primary evidence `ref_id`

Candidates are **deduplicated by normalized Greek surface sequence** (one
hypothesis per distinct Greek output); independent evidence refs are
aggregated.

Verified loanwords are **lemma-level L-clues** (`form_kind=lemma_clue`) and do
**not** automatically outrank attested Greek surfaces.

Print rules offline: `uv run thomas decode ranking-rules`.

## Evidence sufficiency / abstention

Observed / statistical rows require **dual-aligner** agreement before emission.
Singleton one-aligner links (e.g. Coptic `ⲁⲛⲟⲛ` → `πληρωματος`) abstain with
`insufficient_evidence` rather than asserting unsupported Greek. Independently
verified loanword L-clues are not gated by aligner corroboration.

| Reason | Meaning |
| --- | --- |
| `empty_span` | No Coptic surfaces |
| `no_attested_evidence` | No construction / observed wordform / allowed loanword |
| `insufficient_evidence` | Only weak/uncorroborated statistical hits |
| `construction_id_unknown` | Hint id missing from the statistical table |
| `construction_pattern_mismatch` | Hint id’s Coptic pattern ≠ span surfaces |
| `ablation_excludes_loanwords` | Only verified loanword evidence would apply under arm A |
| `forbidden_query_unit` | P.Oxy. fragment handle in request/span/source/token |
| `input_id_inconsistent` | Mismatched unit/span/source/token IDs |

## Leakage controls

| Forbidden | Enforcement |
| --- | --- |
| Holdout / dev / omission as training evidence | JAK-37/38 validators on load |
| Thomas 7 / P.Oxy. 654 Greek as evidence | Validators + fragment query guards |
| Provenance / split mismatch across tables | Fail closed on load |
| LLM / network on decode path | Provenance `llm=false`, `network=false` |

Querying Coptic `thomas.*` units (blinded reconstruction targets) is allowed.

## CLI

```bash
uv run thomas decode run --ablation nt_aligner_plus_verified_loanwords \
  --surface ⲙⲁⲕⲁⲣⲓⲟⲥ --unit-id thomas.7 -o /tmp/decode.json
uv run thomas decode run --request path/to/DecodeRequest.json -o /tmp/decode.json
uv run thomas decode validate /tmp/decode.json
uv run thomas decode ranking-rules
```

## Out of scope

JAK-40 blinded reconstruction scoring, Phase 3 approved-model freeze, inferred
lemma opt-in (removed from v1 request schema), full morphology generation,
alternatives UI, website export, Ollama on this path.
