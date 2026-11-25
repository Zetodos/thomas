# JAK-40 Thomas 7 surviving-letter freeze

Recovered from commit `2608302`. The reconstruction is the frozen blinded decode. It is not recomputed from P.Oxy. or from model-v3.

| File | Role |
| --- | --- |
| `thomas-7-blinded-freeze-v1.json` | Pre-unseal digest pin |
| `blinded-reconstruction-v1.json` | Frozen Thomas 7 reconstruction the pin covers |
| `thomas-7-surviving-letter-eval-v1.json` | Gate-blocked surviving-letter score (`official: false`) |

`review_status` stays `ready-for-independent-review`. Do not relabel it `human-checked`.

This score is separate from `docs/benchmark-report-v3.md`. See [`docs/thomas-7-surviving-letter-score.md`](../../../docs/thomas-7-surviving-letter-score.md).

Re-check the gate-blocked report:

```bash
uv run thomas eval score-thomas7 --gate-report
```
