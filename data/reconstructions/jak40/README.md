# JAK-40 blinded reconstruction artifacts

| File | Role |
| --- | --- |
| `blinded-reconstruction-v1.json` | Blinded Thomas 7 + NT holdout decode + word-for-word baseline |
| `thomas-7-surviving-letter-eval-v1.json` | Surviving-letter report (gate-blocked until `human-checked`) |
| `nt-holdout-letter-eval-v1.json` | NT holdout letter agreement vs Vaticanus (provisional gold) |

Rebuild:

```bash
uv run thomas eval reconstruct -o data/reconstructions/jak40/blinded-reconstruction-v1.json
uv run thomas eval score-thomas7 --gate-report \
  -r data/reconstructions/jak40/blinded-reconstruction-v1.json \
  -o data/reconstructions/jak40/thomas-7-surviving-letter-eval-v1.json
uv run thomas eval score-nt-holdouts \
  -r data/reconstructions/jak40/blinded-reconstruction-v1.json \
  -o data/reconstructions/jak40/nt-holdout-letter-eval-v1.json
```

See [`docs/blinded-reconstruction.md`](../../../docs/blinded-reconstruction.md).
