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

Binaries land in gitignored `data/local/aligners/`. Upstream commits are
**pinned** in `zetodos_thomas.integrations.aligners.pins` and written to
`data/local/aligners/versions.json` + report `aligner_pins`. Adapters refuse
to invent scores when the tool is absent.

## Leakage controls

| Forbidden as positive training pairs | Enforcement |
| --- | --- |
| Holdout / dev units in fit bitext | OOS: fit on `SplitRole.TRAIN` only; apply to unseen score sets |
| John 8:1–11 PA omission | Bitext builder emits zero omission sentences |
| Thomas 7 / P.Oxy. 654 (and other P.Oxy. evaluation witnesses) | Split v3 exclusions + train unit_id guards |

## Protocol

1. Tokenize nonempty train/dev/holdout with bound-group Coptic + whitespace Greek.
   Each bound-group token carries a **parent-v1 id** via original char spans.
2. For each train fraction `{0.25, 0.5, 0.75, 1.0}`, **fit on train_subset only**
   and **apply to the fixed unseen dev** set:
   - eflomal: train → Dirichlet priors → dev-only align with frozen priors
   - fast_align: train → frozen co-occurrence table → greedy apply + symmetrize
3. Full-train **dev** and **holdout**: same OOS apply path for both aligners.
   Export the exact `AlignmentLinks` used for full-train→dev (no second draw).
4. Control: lexical Dice baseline with the same Coptic segmentation (also OOS;
   preds projected to parent-v1 before gold scoring).
5. Regression guards: fit∩score unit_ids must be empty; repeated Coptic `ⲡ`
   cannot cross-match via substring.

**Primary intrinsic metric:** mean forward↔reverse agreement
`|fwd ∩ rev| / |fwd ∪ rev|` on eval sentences (no gold required).

**Secondary:** provisional gold F1 via **positional parent-v1 identity**
`(unit_id, left_token_id, right_norm)` after char-span projection — **not**
substring membership. Gold remains `provisional-draft` — not decisive.

## Learning-curve / sufficiency notes (seed=36 committed run)

Primary intrinsic metric = mean forward↔reverse agreement on **dev** (51 units),
**out-of-sample** protocol. Train nonempty = **304**. Exact floats live in
`statistical-aligner-eval-v1.json` (eflomal varies slightly across runs).

| Train fraction | Train n | eflomal agreement | fast_align agreement |
| --- | ---: | ---: | ---: |
| 0.25 | 76 | ~0.49 | ~0.14 |
| 0.50 | 152 | ~0.51 | ~0.13 |
| 0.75 | 228 | ~0.51 | ~0.13 |
| 1.00 | 304 | ~0.52 | ~0.13 |

Full-train **dev** (OOS): eflomal ~**0.54** vs fast_align ~**0.13** (frozen
lexical apply). **Holdout** (41, OOS): eflomal ~**0.60**; fast_align ~**0.14**.

**Sufficiency:** eflomal leads clearly under the shared OOS intrinsic; the
25%→100% eflomal slope is still mild/noisy (Gibbs variance), not a plateau
proof. fast_align’s frozen-table apply is much weaker than prior joint-EM
dev numbers — treat those as superseded. v3 remains adequate to **rank** tools
and begin Phase 2 lexical work; more Gospel/NT pairs and human-checked gold
still matter more for JAK-37/40 than another bake-off.

**Burden / reproducibility:** eflomal needs OpenMP + Python headers; fast_align
needs cmake/C++ only. fast_align EM is deterministic for fixed flags; eflomal's
Gibbs sampler uses OS entropy (seed recorded, not fully pinning). Upstream SHAs
are pinned.

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
