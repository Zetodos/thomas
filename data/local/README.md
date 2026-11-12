# Local / restricted imports (gitignored)

User-supplied or gated corpora (if ever permitted) go here. Contents except this README are **not** committed.

Wells **Sahidica**, **NTAEG**, and **Bohairica** are **not** part of the default offline path. Do not place them under `data/sources/` or `data/manifests/` without an explicit rights decision and `manual-import` gate. See [`docs/source-licenses.md`](../../docs/source-licenses.md#gated-wells-sahidica--ntaeg--bohairica).

## Statistical aligner binaries (JAK-36)

After explicit setup (`uv run thomas align setup-aligners`), native binaries are
copied to `data/local/aligners/` (`eflomal`, `fast_align`, optional `atools`).
That directory is gitignored; CI stays green without them via soft probes and
skipped smoke tests. See [`docs/statistical-aligner-comparison.md`](../../docs/statistical-aligner-comparison.md).
