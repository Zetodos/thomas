# JAK-41 Phase 2 feasibility artifacts

| File | Role |
| --- | --- |
| `feasibility-metrics-v1.json` | Multi-level decoder vs word-for-word metrics + gate recommendation |

Written report: [`docs/phase2-feasibility-report.md`](../../../docs/phase2-feasibility-report.md)

Rebuild (offline):

```bash
uv run thomas eval feasibility-report
```

Inputs are the committed JAK-40 packages under `../jak40/`. Official Thomas 7
surviving-letter scores remain gated on `human-checked` (see JAK-40 / JAK-43).
