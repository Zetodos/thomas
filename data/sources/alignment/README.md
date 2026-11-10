# Alignment pilot artifacts (JAK-28)

| Path | Role |
| --- | --- |
| `reference/horner-vaticanus-gold-sample-v1.json` | Provisional token/phrase `AlignmentDocument` (unreviewed draft) |
| `benchmark-report.json` | Machine benchmark with denominators (no pass bar) |
| `../pilot/horner-vaticanus-nt-pilot.json` | Verse-paired Horner × Vaticanus B\* corpus (v2) |
| `../../training/splits/` (JAK-33) | Canonical train/dev/holdout — **not** duplicated here |

Docs: `docs/alignment-pilot-splits.md` (consumer notes), `docs/alignment-pilot-benchmark.md`, and JAK-33 `docs/benchmark-holdouts.md`.

```bash
uv run thomas sources validate
uv run thomas align validate
uv run thomas align benchmark
```
