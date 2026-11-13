# Phrase/construction mappings and verified loanwords (JAK-38)

Minimal Phase 2 mapping layer for the feasibility experiment. **Statistical**
correspondences (train-split aligner observations) stay separate from
**independently verified** CCL/DDGLC loanword evidence (Jake revision).

## Artifacts

| Path | Evidence class | Role |
| --- | --- | --- |
| [`data/sources/mappings/statistical-phrase-constructions-v1.json`](../data/sources/mappings/statistical-phrase-constructions-v1.json) | `statistical` | Negation / article / pronoun / verbal patterns from dual-aligner train links |
| [`data/sources/mappings/verified-loanwords-ccl-ddglc-v1.json`](../data/sources/mappings/verified-loanwords-ccl-ddglc-v1.json) | `independently_verified` | Spot-checked CCL/DDGLC subset (CC BY-SA 4.0) |
| [`data/sources/mappings/ablation-nt-aligner-only-v1.json`](../data/sources/mappings/ablation-nt-aligner-only-v1.json) | — | Ablation arm A: NT aligner stack only |
| [`data/sources/mappings/ablation-nt-aligner-plus-loanwords-v1.json`](../data/sources/mappings/ablation-nt-aligner-plus-loanwords-v1.json) | — | Ablation arm B: arm A + verified loanwords |
| [`data/manifests/ccl-ddglc-tei.json`](../data/manifests/ccl-ddglc-tei.json) | — | CCL TEI v1.2 pin + attribution (JAK-19) |

## Layer separation (do not collapse)

| Class | Source | Use |
| --- | --- | --- |
| **statistical** | Train-only symmetrized `eflomal` + `fast_align` links | Soft phrase/construction candidates |
| **independently_verified** | CCL TEI / DDGLC, manually spot-checked | Stronger L-evidence / decoder constraints |

Aligner-derived pairs are **not** verified loanword facts. Verified rows are
**not** mixed into statistical training observations.

## Dual aligners (JAK-36 keep-both)

`phase2_active_aligners = ["eflomal", "fast_align"]`, `adopted_aligner = null`.
Statistical constructions require dual-aligner agreement.

## Span IDs (JAK-32)

Each statistical example mints:

- Coptic span ids on the bound-group **v2** token layer
- Greek span ids on the whitespace **v1** token layer
- `v1:parallel/…` parallel ids

These handles are the JAK-39 decoder input surface (construction_id + spans).

## Leakage controls

| Forbidden | Enforcement |
| --- | --- |
| Dev / holdout / omission as observations | Train-role + train membership checks |
| Thomas 7 / P.Oxy. 654 Greek | Unit-id guards; verified illustrative spans reject `thomas.` / `poxy` |
| Treating statistical rows as verified | Distinct `evidence_class` + validation |

Pilot sayings **7 / 17 / 30 / 77** may prioritize which constructions to keep
(negation, articles, pronouns, future/verbal) but must not supply hidden Greek
for training or verification shortcuts. Thomas 7 `ⲙⲁⲕⲁⲣⲓⲟⲥ` motivates the
verified makarios row as an **L clue only** — P.Oxy. 654 is not consulted.

## Ablation

1. **Arm A (`nt_aligner_only`)** — lexical candidates (JAK-37) + statistical
   constructions; verified loanwords omitted.
2. **Arm B (`nt_aligner_plus_verified_loanwords`)** — arm A plus the CCL/DDGLC
   verified subset as separate constraints.

## Reproduce / validate

```bash
uv run thomas mappings write-loanwords
uv run thomas mappings generate-phrases   # uses committed JAK-37 train-links
uv run thomas mappings build-ablation
uv run thomas mappings validate
```

Offline validation does not require aligner binaries when committed artifacts
are present. No LLM on this path. Decoder consumer: [`docs/deterministic-decoder.md`](deterministic-decoder.md) (JAK-39). Out of scope here: full Phase 3
construction grammar, DIOSKOROS, website.
