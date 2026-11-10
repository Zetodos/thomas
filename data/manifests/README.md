# Source manifests

Per-source machine-readable **JSON** manifests (stable `id`).

- **Schema / policy:** [`docs/source-licenses.md`](../../docs/source-licenses.md#manifest-schema)
- **Typed model:** `zetodos_thomas.schemas.source.SourceManifest`
- **Validate:** `uv run thomas sources validate`

## Current manifests

| File | Source |
| --- | --- |
| [`copsahhorner.json`](copsahhorner.json) | CrossWire Horner Sahidic NT (default Sahidic; JAK-24) |
| [`ntvmr-vaticanus-b.json`](ntvmr-vaticanus-b.json) | INTF/NTVMR Vaticanus GA 03 B* (default Greek; JAK-24/25) |
| [`dilley-thomas.json`](dilley-thomas.json) | Dilley / SCRIPTORIUM Coptic Thomas (primary target; JAK-30) |
| [`dclp-poxy-1.json`](dclp-poxy-1.json) | DCLP P.Oxy. 1 Greek Thomas (CC BY 3.0; JAK-31) |
| [`dclp-poxy-654.json`](dclp-poxy-654.json) | DCLP P.Oxy. 654 Greek Thomas (CC BY 3.0; JAK-31) |
| [`dclp-poxy-655.json`](dclp-poxy-655.json) | DCLP P.Oxy. 655 Greek Thomas (CC BY 3.0; JAK-31) |

Do **not** add Sahidica, NTAEG, or Bohairica to this default set.
Westcott–Hort is not part of the default Greek spine (JAK-25).
Do **not** add P.Oxy. 5575 as a Thomas manuscript (independent parallel).
Fragment notes: [`docs/greek-thomas-fragments.md`](../../docs/greek-thomas-fragments.md).
