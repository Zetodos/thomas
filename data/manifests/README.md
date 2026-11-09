# Source manifests

Per-source machine-readable **JSON** manifests (stable `id`).

- **Schema / policy:** [`docs/source-licenses.md`](../../docs/source-licenses.md#manifest-schema)
- **Typed model:** `zetodos_thomas.schemas.source.SourceManifest`
- **Validate:** `uv run thomas sources validate`

## Current manifests (JAK-24)

| File | Source |
| --- | --- |
| [`copsahhorner.json`](copsahhorner.json) | CrossWire Horner Sahidic NT (default Sahidic) |
| [`ntvmr-vaticanus-b.json`](ntvmr-vaticanus-b.json) | INTF/NTVMR Vaticanus GA 03 B* (default Greek) |

Do **not** add Sahidica, NTAEG, or Bohairica to this default set.
Westcott–Hort is not part of the default Greek spine (JAK-25).
