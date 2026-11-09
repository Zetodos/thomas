# Thomas evidence packages

Machine-readable evidence inventories and cross-refs (PLAN.md `data/evidence/`).

| Path | Contents |
| --- | --- |
| [`pilot/thomas-7-17-30-77.json`](pilot/thomas-7-17-30-77.json) | Phase 1 pilot set for Thomas **7, 17, 30, 77** (JAK-34) |
| [`pilot/thomas-7-17-30-77.jsonl`](pilot/thomas-7-17-30-77.jsonl) | Same sayings, one JSON object per line |

Schema: `zetodos_thomas.schemas.evidence.PilotEvidenceSet`.

Validate offline (with Dilley + fragment corpora present):

```bash
uv run thomas sources validate
```

Docs: [`docs/pilot-evidence-set.md`](../../docs/pilot-evidence-set.md).
P.Oxy. 1 vs Coptic 30/77 divergence detail beyond this pilot: **JAK-33**.
