# Source manifests

Per-source machine-readable **JSON** manifests (stable `id`).

- **Schema / policy:** [`docs/source-licenses.md`](../../docs/source-licenses.md#manifest-schema)
- **Typed model:** `zetodos_thomas.schemas.source.SourceManifest`
- **Validate:** `uv run thomas sources validate`

## Current manifests (JAK-24)

| File | Source |
| --- | --- |
| [`copsahhorner.json`](copsahhorner.json) | CrossWire Horner Sahidic NT (default Sahidic) |
| [`westcott-hort.json`](westcott-hort.json) | Robinson WH Greek NT (default Greek) |

Do **not** add Sahidica, NTAEG, or Bohairica to this default set.
