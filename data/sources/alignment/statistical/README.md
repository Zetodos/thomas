# Statistical aligner artifacts (JAK-36)

Inspectable outputs from `uv run thomas align statistical-eval`:

| File | Role |
| --- | --- |
| `statistical-aligner-eval-v1.json` | Comparison + learning curves + Jake decision scaffold |
| `eflomal-train-dev-links-v1.json` | Compact bidirectional + symmetrized Pharaoh links (train+dev) |
| `fast_align-train-dev-links-v1.json` | Same for fast_align |

Committed link JSON is **compact** (unit ids + index pairs + lengths; token
strings recoverable from the pilot via bitext sha256). Provenance includes
pilot/split ids, segmentation version, seed, and aligner config. Binaries stay
under gitignored `data/local/aligners/` after `thomas align setup-aligners`.

See [`docs/statistical-aligner-comparison.md`](../../../../docs/statistical-aligner-comparison.md).
