# JAK-40 blinded reconstruction artifacts

Committed packages are **split** so each JSON stays under the
`check-added-large-files` 1024 KiB limit.

| File | Role |
| --- | --- |
| `blinded-reconstruction-v1.json` | Thomas 7 blinded decode + word-for-word baseline (full multi-candidate) |
| `nt-holdout-blinded-reconstruction-v1.json` | NT holdout Sahidic decode (top-1; compact JSON ≤1024 KiB) |
| `thomas-7-blinded-freeze-v1.json` | Pre-unseal digest pin for official Thomas 7 scoring |
| `thomas-7-surviving-letter-eval-v1.json` | Surviving-letter report (gate-blocked until `human-checked`) |
| `nt-holdout-letter-eval-v1.json` | NT holdout letter agreement vs Vaticanus (provisional gold) |

Rebuild:

```bash
uv run thomas eval reconstruct
uv run thomas eval score-thomas7 --gate-report \
  -r data/reconstructions/jak40/blinded-reconstruction-v1.json \
  -o data/reconstructions/jak40/thomas-7-surviving-letter-eval-v1.json
uv run thomas eval score-nt-holdouts \
  -r data/reconstructions/jak40/nt-holdout-blinded-reconstruction-v1.json \
  -o data/reconstructions/jak40/nt-holdout-letter-eval-v1.json
```

See [`docs/blinded-reconstruction.md`](../../../docs/blinded-reconstruction.md).
