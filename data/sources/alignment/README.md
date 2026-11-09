# Alignment pilot artifacts (JAK-28)

| Path | Role |
| --- | --- |
| `reference/horner-vaticanus-gold-sample-v1.json` | Human-checked token/phrase `AlignmentDocument` |
| `benchmark-report.json` | Machine benchmark with denominators (no pass bar) |
| `../pilot/splits.json` | Whole-passage train/dev/holdout/omission splits |
| `../pilot/horner-vaticanus-nt-pilot.json` | Verse-paired Horner × Vaticanus B\* corpus |

Docs: `docs/alignment-pilot-splits.md`, `docs/alignment-pilot-benchmark.md`.

```bash
uv run thomas align validate
uv run thomas align benchmark
```
