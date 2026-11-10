# Training artifacts

Greek–Coptic training policy and (later) observed alignments.

| Path | Contents | Owner |
| --- | --- | --- |
| [`splits/horner-vaticanus-nt-split-v3.json`](splits/horner-vaticanus-nt-split-v3.json) | Canonical whole-passage train/dev/holdout for Horner × Vaticanus B\* **v3** (407 units; 396 nonempty) | **JAK-43** (consumed by JAK-36 / align CLI) |
| [`horner-vaticanus-nt-v3-inventory.json`](horner-vaticanus-nt-v3-inventory.json) | Slot / nonempty / split counts after expansion | **JAK-43** |
| [`splits/horner-vaticanus-nt-split-v2.json`](splits/horner-vaticanus-nt-split-v2.json) | Archival Phase 1 split (46 units) | **JAK-33** |

Do **not** place a competing NT split JSON under `data/sources/pilot/`. See [`docs/eval-data-readiness.md`](../../docs/eval-data-readiness.md) and [`docs/benchmark-holdouts.md`](../../docs/benchmark-holdouts.md).
