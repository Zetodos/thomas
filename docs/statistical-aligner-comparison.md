# Statistical aligner comparison (JAK-36)

**Corpus / split:** `horner-vaticanus-nt-pilot-v3` + `horner-vaticanus-nt-split-v3`
**Segmentation:** Coptic `bound-group-v1` (token id scheme **v2**); Greek whitespace **v1**
**Greek NT:** Vaticanus B\* only (`ntvmr-vaticanus-b`)
**Machine report:** [`data/sources/alignment/statistical/statistical-aligner-eval-v1.json`](../data/sources/alignment/statistical/statistical-aligner-eval-v1.json)

## Goal

Evaluate a local **bidirectional** statistical word aligner on the expanded v3
train/dev/holdout stack. **eflomal first**, compared with **fast_align** on
identical partitions. Measure **learning curves / corpus sufficiency**, not
merely whether a binary runs.

## Setup (explicit; offline afterward)

```bash
# Needs network + gcc/g++/make/cmake/libomp/python3-dev once:
uv run thomas align setup-aligners
uv run thomas align aligner-probe
```

Binaries land in gitignored `data/local/aligners/`. Adapters live under
`zetodos_thomas.integrations.aligners` and refuse to invent scores when the
tool is absent.

## Leakage controls

| Forbidden as positive training pairs | Enforcement |
| --- | --- |
| Holdout units | Learning-curve train subsets drawn only from `SplitRole.TRAIN` |
| John 8:1–11 PA omission | Bitext builder emits zero omission sentences |
| Thomas 7 / P.Oxy. 654 (and other P.Oxy. evaluation witnesses) | Split v3 exclusions + train unit_id guards |

## Protocol

1. Tokenize nonempty train/dev/holdout with bound-group Coptic + whitespace Greek.
2. For each train fraction `{0.25, 0.5, 0.75, 1.0}`, jointly align
   `train_subset ∪ dev` (unsupervised) and score **dev only**.
3. Full-train comparison: same protocol on **dev** and **holdout**.
4. Control: lexical Dice baseline with the same Coptic segmentation.
5. Export inspectable JSON alignments (forward / reverse / symmetrized) with
   provenance (corpus id, split id, bitext sha256, seed, config).

**Primary intrinsic metric:** mean forward↔reverse agreement
`|fwd ∩ rev| / |fwd ∪ rev|` on eval sentences (no gold required).

**Secondary:** symmetrized density / source fertility; soft norm-form F1 against
provisional v1 gold (bridges whitespace gold ↔ v2 segmented tokens). Gold remains
`provisional-draft` — not decisive.

## Learning-curve / sufficiency notes (seed=36 committed run)

Primary intrinsic metric = mean forward↔reverse agreement on **dev** (51 units),
joint unsupervised protocol. Train nonempty = **304**. Exact floats live in
`statistical-aligner-eval-v1.json` (eflomal varies slightly across runs).

| Train fraction | Train n | eflomal agreement | fast_align agreement |
| --- | ---: | ---: | ---: |
| 0.25 | 76 | ~0.50 | ~0.42 |
| 0.50 | 152 | ~0.53 | ~0.42 |
| 0.75 | 228 | ~0.50 | ~0.42 |
| 1.00 | 304 | ~0.53 | ~0.42 |

Full-train **dev**: eflomal ~**0.51** vs fast_align ~**0.42** (Δ ≈ +0.09).
Full-train **holdout** (41): eflomal ~**0.49** vs fast_align ~**0.41**.

**Sufficiency:** eflomal stays above fast_align on this intrinsic at every curve
point; the 25%→100% slope is weak / noisy (Gibbs variance), not a clean plateau
proof. v3 is adequate to **rank** tools and begin Phase 2 lexical work; more
Gospel/NT pairs and human-checked gold still matter more for JAK-37/40 than
another bake-off.

**Burden / reproducibility:** eflomal needs OpenMP + Python headers; fast_align
needs cmake/C++ only. fast_align EM is deterministic for fixed flags; eflomal's
Gibbs sampler uses OS entropy (seed recorded, not fully pinning).

## Jake decision (do not silent-lock)

The report’s `jake_decision` block leaves `adopted_aligner: null` and asks Jake
to **adopt eflomal**, **adopt fast_align**, or **keep both** for Phase 2 lexical
table work. Thin intrinsic margins (< ~0.02 agreement) or missing binaries force
`needs-jake-decision`.

## Reproduce

```bash
uv run thomas align statistical-eval --seed 36
# artifacts under data/sources/alignment/statistical/
```

No LLM on this path.
