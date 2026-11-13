# Blinded reconstruction + surviving-letter eval (JAK-40)

Phase 2 feasibility run: reconstruct **Thomas 7** (Dilley Coptic) and **NT
holdout** Sahidic verses with the JAK-39 decoder + JAK-37/38 evidence, **without
consulting** P.Oxy. 654 / holdout Greek during decode. Score surviving Greek
letters only at evaluation time.

No LLM. Dual aligners stay active (`eflomal` + `fast_align`;
`adopted_aligner=null`).

## Blinding protocol

| Stage | Allowed inputs | Forbidden |
| --- | --- | --- |
| Rule / table selection | Train-split JAK-37/38 tables | Thomas-7 Greek, P.Oxy. 654, holdout/dev as training, fragment-derived manual maps |
| Blinded reconstruct | Dilley `thomas.7` Coptic; holdout **Sahidic** only | Opening `greek-thomas-fragments.json`; reading ParallelUnit.greek into the decoder |
| Surviving-letter score | P.Oxy. 654 surviving layer (lines `main.40`–`main.42`); Vaticanus B* for NT holdouts | Counting editorial **restorations** in the gold denominator |

General NT-learned correspondences that happen to predict the same Greek word
as the fragment are permissible (Jake Phase 2 safeguard).

## Review gate

Official Thomas 7 scores call
`zetodos_thomas.eval.review_gate.assert_eval_package_human_checked` on
[`thomas-7-blinded-v1.json`](../data/evidence/eval/thomas-7-blinded-v1.json).
While `review_status` is `ready-for-independent-review`, official scoring is
**blocked** (JAK-43 follow-up). Blinded reconstruction artifacts remain valid.
Use `--allow-unreviewed` only for harness smoke (`official=false`), or
`--gate-report` to write a gate-blocked report without inventing
`human-checked`.

## Artifacts

| Path | Role |
| --- | --- |
| [`data/reconstructions/jak40/blinded-reconstruction-v1.json`](../data/reconstructions/jak40/blinded-reconstruction-v1.json) | Thomas 7 blinded decode + word-for-word baseline |
| [`data/reconstructions/jak40/nt-holdout-blinded-reconstruction-v1.json`](../data/reconstructions/jak40/nt-holdout-blinded-reconstruction-v1.json) | NT holdout decode (top-1; split for ≤1024 KiB hook) |
| [`data/reconstructions/jak40/thomas-7-blinded-freeze-v1.json`](../data/reconstructions/jak40/thomas-7-blinded-freeze-v1.json) | Pre-unseal digest pin for official Thomas 7 scoring |
| [`data/reconstructions/jak40/thomas-7-surviving-letter-eval-v1.json`](../data/reconstructions/jak40/thomas-7-surviving-letter-eval-v1.json) | Thomas 7 surviving-letter report (or gate-blocked stub) |
| [`data/reconstructions/jak40/nt-holdout-letter-eval-v1.json`](../data/reconstructions/jak40/nt-holdout-letter-eval-v1.json) | NT holdout letter agreement (provisional gold) |

Each reconstruction unit records ranked candidates / abstentions from JAK-39,
source token provenance (`input_span` + real Dilley/Horner ids), covering spans
(including multi-token JAK-38 constructions when matched), `artifact_digests`,
and `blinding.greek_consulted_during_reconstruction=false`.

NT holdouts use train-parity `bound-group-v1` (v2 ids); Dilley Thomas uses
`dilley-normalized-whitespace-v1`. Whitespace-unsplit Mark 4:4 groups such as
`ⲛⲧⲉⲣⲉϥϫⲟ,` and `ⲛϭⲓⲛϩⲁⲗⲁⲧⲉ` are therefore split before decode.

## Metrics (elementary)

Thomas 7 uses **gap-aware ordered-run** letter agreement (restorations / unclear
as gaps). NT holdouts use **token-positional** letter agreement. Global LCS is
retained only as `incidental_global_lcs` (diagnostic — not gate precision/F1).
Unclear papyrus letters are excluded from primary gold by default (sensitivity
reported separately). No invented pass/fail threshold (Jake). Multi-level lemma /
order / phrase metrics and the decoder-vs-baseline comparison gate live in
**JAK-41**.

## CLI

```bash
# Blinded reconstruct (Coptic only; offline) — writes Thomas + NT sibling files
uv run thomas eval reconstruct

# Thomas 7 surviving-letter score (opens P.Oxy. 654 here only)
uv run thomas eval score-thomas7 --gate-report \
  -r data/reconstructions/jak40/blinded-reconstruction-v1.json

# Protocol smoke (unofficial) once review is still pending:
uv run thomas eval score-thomas7 --allow-unreviewed \
  -r data/reconstructions/jak40/blinded-reconstruction-v1.json \
  -o /tmp/thomas7-unofficial.json

# NT holdouts vs Vaticanus at eval time
uv run thomas eval score-nt-holdouts \
  -r data/reconstructions/jak40/nt-holdout-blinded-reconstruction-v1.json

uv run thomas eval validate-reconstruction \
  data/reconstructions/jak40/blinded-reconstruction-v1.json
uv run thomas eval validate-reconstruction \
  data/reconstructions/jak40/nt-holdout-blinded-reconstruction-v1.json
```

## Leakage controls

| Control | Where |
| --- | --- |
| Train/dev not scored as holdout | Split v3 role filter |
| PA omission fixtures excluded | Omission catalog separate from holdout |
| P.Oxy. query units rejected | JAK-39 `forbidden_query_unit` |
| Fragment Greek absent from reconstruct JSON | `validate-reconstruction` + tests |
| Review gate before official Thomas 7 score | `assert_eval_package_human_checked` |
| Keep-both aligners | Package provenance |

## Out of scope

JAK-41 feasibility report / decision gate, Phase 3 model freeze, Ollama,
critical edition, using restored letters as gold.
