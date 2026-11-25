# Thomas 7 surviving-letter score

This is the restored JAK-40 surviving-letter freeze. It is its own score. It is not the model-v3 papyrus table in [`benchmark-report-v3.md`](benchmark-report-v3.md), and it is not the exploratory fragment span `thomas.7.poxy654.a` in that table.

The committed score is gate-blocked. There is no official surviving-letter F1.

| Field | Value |
| --- | --- |
| Score file | [`data/reconstructions/jak40/thomas-7-surviving-letter-eval-v1.json`](../data/reconstructions/jak40/thomas-7-surviving-letter-eval-v1.json) |
| `official` | `false` |
| `review_status` | `ready-for-independent-review` |
| Gate blocked | `true` |
| `scores` | empty |
| `gold` | `null` |
| Greek loaded at scoring | `false` |
| Freeze | [`data/reconstructions/jak40/thomas-7-blinded-freeze-v1.json`](../data/reconstructions/jak40/thomas-7-blinded-freeze-v1.json) |
| Frozen reconstruction | [`data/reconstructions/jak40/blinded-reconstruction-v1.json`](../data/reconstructions/jak40/blinded-reconstruction-v1.json) |
| Reconstruction sha256 | `f3fd90a0b0af143a2150224dda975ae0e3ed5648a03cd6d8276853acaa092d9c` |
| Eval package | [`data/evidence/eval/thomas-7-blinded-v1.json`](../data/evidence/eval/thomas-7-blinded-v1.json) |

The freeze was sealed before any P.Oxy. / fragment load (`sealed_before_fragment_load: true`). The predictions in the reconstruction package are that frozen artifact. This restore does not edit them and does not relabel the eval package `human-checked`. `ready-for-independent-review` is packaged readiness only.

The letter-score file records `freeze_verified: false` because official scoring returns before the pin is applied. The pin itself matches the reconstruction digest. The recovered JAK-41 record reports that pin check separately: `freeze_verified: true`, `official_surviving_letter_scores: false`, `evaluation_ready: false`, `gate_recommendation: pause_and_improve` ([`docs/phase2-feasibility-report.md`](phase2-feasibility-report.md)). The NT holdout letter figures in that record are not this score.

Re-check the gate-blocked report without rewriting predictions:

```bash
uv run thomas eval score-thomas7 --gate-report
```

`--fail-closed` exits non-zero while the review gate is closed. `--allow-unreviewed` is an unofficial protocol smoke (`official: false`). It is not this score, and it does not change the frozen reconstruction or the eval package.
