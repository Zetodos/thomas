# Evidence catalogs

Thomas-specific evidence inventories and policy catalogs (PLAN.md `data/evidence/`).

| Path | Contents | Owner |
| --- | --- | --- |
| [`pilot/thomas-7-17-30-77.json`](pilot/thomas-7-17-30-77.json) | Phase 1 pilot set for Thomas **7, 17, 30, 77** | JAK-34 |
| [`pilot/thomas-7-17-30-77.jsonl`](pilot/thomas-7-17-30-77.jsonl) | Same sayings, one JSON object per line | JAK-34 |
| [`omissions/john-8-pa-omission-fixture-v1.json`](omissions/john-8-pa-omission-fixture-v1.json) | John 8:1–11 dual-empty omission fixtures | JAK-33 |
| [`divergences/greek-thomas-fragment-divergences-v1.json`](divergences/greek-thomas-fragment-divergences-v1.json) | Greek fragment vs Coptic sequence/edition divergences | JAK-33 |

Schema (pilot): `zetodos_thomas.schemas.evidence.PilotEvidenceSet`.

Validate offline (with Dilley + fragment corpora present):

```bash
uv run thomas sources validate
```

Docs: [`docs/pilot-evidence-set.md`](../../docs/pilot-evidence-set.md),
[`docs/benchmark-holdouts.md`](../../docs/benchmark-holdouts.md).
Fragment TEI / Dilley Thomas text remain under `data/sources/` (JAK-30 / JAK-31).
