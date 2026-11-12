# Lexical candidate artifacts (JAK-37)

Inspectable Coptic→Greek lexical candidates built from **train-only**
symmetrized alignments of the Phase 2 active aligners (`eflomal`, `fast_align`;
`adopted_aligner=null` per Jake / JAK-36).

| File | Role |
| --- | --- |
| `coptic-greek-lexical-candidates-v1.json` | Three-layer candidate table (`columnar-v1`) |
| `eflomal-train-links-v1.json` | Compact train-only Pharaoh links (eflomal) |
| `fast_align-train-links-v1.json` | Compact train-only Pharaoh links (fast_align) |

## Layers (do not collapse)

1. **observed_wordform** — attested Coptic↔Greek surfaces with observation
   counts, per-aligner provenance, source unit IDs, and example unit refs.
2. **inferred_lemma** — lemma *proxies* derived from observed wordforms
   (`inference_method` required; not MorphGNT/SCRIPTORIUM lemmas).
3. **confidence_estimate** — separate confidence/uncertainty rows keyed to
   observed wordforms; null when evidence is insufficient.

`observation_count` is `max(per_aligner counts)` — never a fabricated sum of
overlapping aligner views.

## Commands

```bash
# Needs aligners once (or pass --eflomal-links / --fast-align-links):
uv run thomas lexicon generate-candidates --seed 36
uv run thomas lexicon validate-candidates
```

Holdout / PA omission / Thomas 7 / P.Oxy. 654 never feed observations.
Greek NT = Vaticanus B* only. No LLM on this path.
