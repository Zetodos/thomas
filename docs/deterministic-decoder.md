# Minimal deterministic decoder (JAK-39)

Phase 2 feasibility decoder: ranked multi-candidate Greek hypotheses from
frozen JAK-37/38 tables, with explicit uncertainty and abstention. No network.
No LLM on the decode path.

## Inputs

| Surface | Source |
| --- | --- |
| Coptic span ids + token ids + surfaces | JAK-32 / JAK-38 construction examples |
| Optional `construction_id` | JAK-38 statistical constructions |
| Ablation arm | JAK-38 `nt_aligner_only` or `nt_aligner_plus_verified_loanwords` |

Request schema: `zetodos_thomas.schemas.decode.DecodeRequest`.

## Evidence (do not collapse)

| Kind | Evidence class | Artifact |
| --- | --- | --- |
| `statistical_construction` | `statistical` | `statistical-phrase-constructions-v1.json` |
| `independently_verified_loanword` | `independently_verified` | `verified-loanwords-ccl-ddglc-v1.json` (arm B only) |
| `lexical_observed_wordform` | — (lexical layer) | `coptic-greek-lexical-candidates-v1.json` |

Aligner-derived pairs are never reclassified as verified loanwords. Loanword
`illustrative_nt_span_ids` are not training exemplars and are not consumed.

Dual aligners stay active: `phase2_active_aligners=["eflomal","fast_align"]`,
`adopted_aligner=null`.

## Ranking (`jak39-stable-ranking-v1`)

Identical inputs → identical outputs. Ascending key:

1. Evidence tier: construction < verified loanword < observed wordform
2. Dual-aligner agreement preferred
3. Higher `observation_count`
4. Higher `heuristic_ranking_score` when present
5. Shorter Greek surface sequences
6. Lexicographic Greek surfaces
7. Lexicographic evidence `ref_id`

Reported `score` is a deterministic display transform of that key — **not** a
calibrated correctness probability. `uncertainty` is explicit and separate.

Print rules offline: `uv run thomas decode ranking-rules`.

## Abstention

The decoder **never invents** unsupported Greek. A span abstains when:

| Reason | Meaning |
| --- | --- |
| `empty_span` | No Coptic surfaces |
| `no_attested_evidence` | No construction / observed wordform / allowed loanword |
| `construction_id_unknown` | Hint id missing from the statistical table |
| `construction_pattern_mismatch` | Hint id’s Coptic pattern ≠ span surfaces |
| `ablation_excludes_loanwords` | Only verified loanword evidence would apply under arm A |
| `forbidden_query_unit` | Decode unit embeds a P.Oxy. Greek fragment handle |

Simple grammatical notes (article / negation / DET+N family labels) annotate
candidates; they do not generate new morphology.

## Leakage controls

| Forbidden | Enforcement |
| --- | --- |
| Holdout / dev / omission as training evidence | Tables already train-only (JAK-37/38); decoder only reads those |
| Thomas 7 / P.Oxy. 654 Greek as evidence | Evidence `ref_id` leakage guards on validate; no fragment Greek tables loaded |
| LLM / network on decode path | Provenance `llm=false`, `network=false`; offline CLI |

Querying Coptic `thomas.*` units (blinded reconstruction targets) is allowed.
P.Oxy. unit handles as decode units are refused.

## CLI

```bash
uv run thomas decode run --ablation nt_aligner_plus_verified_loanwords \
  --surface ⲙⲁⲕⲁⲣⲓⲟⲥ --unit-id thomas.7 -o /tmp/decode.json
uv run thomas decode run --request path/to/DecodeRequest.json -o /tmp/decode.json
uv run thomas decode validate /tmp/decode.json
uv run thomas decode ranking-rules
```

## Out of scope

JAK-40 blinded reconstruction scoring, Phase 3 approved-model freeze, full
morphology generation, alternatives UI, website export, Ollama on this path.
