# Oracle accent-restorer ceiling on the NT dev split

This note is an oracle ceiling and error analysis of `KoineAccentRestorer` on the 51-verse NT dev split. It measures what the unaccented decode path can recover when the letters are already the bare gold.

The published model's `unaccented_target: false` was not chosen by this comparison. The acute row is the MorphGNT gold scored against itself, and the other row is that gold after stripping and restoration, so the gap is the restorer's miss rate. Paired seq2seq checkpoints were not trained, and this note does not choose a training target.

`model-v3` stays published with `unaccented_target: false` (`data/mappings/model-v3/seq2seq-config.json`). The rerank weights in `data/mappings/model-v3/rerank-weights.json` stay the existing ordered-LCS dev-grid point.

The scored set is John 3:1–21 and Luke 6:20–49 in `data/training/splits/nt-split-v1.json`. The restorer is built with those 51 verses and the 41 holdout verses excluded. The papyrus corpus is unused. The published holdout tables in `docs/benchmark-report-v3.md` and `data/generated/eval-report-v3.json` are unchanged.

## What was scored

The objective is the one in `tune_rerank_weights_on_dev`: `0.60 * ordered content lemma F1 + 0.40 * ordered surface F1`, micro-averaged over the 51 verses. Content POS is `{N-, V-, A-, D-}`. Hypothesis lemmas come from the committed model-v3 Greek analyzer. Gold lemmas and surfaces come from MorphGNT.

Both rows start from those MorphGNT surfaces.

The acute row uses the surfaces as the hypothesis. Surface F1 is 1.0000 because the hypothesis is the reference. Ordered content lemma F1 is 0.9239. The committed analyzer was fit without these 51 verses, so its reading of the correct surfaces has ordered content lemma LCS 455 against 519 gold lemmas (predicted 466).

The restored row strips the same surfaces and passes them through `KoineAccentRestorer`, built the way `thomas model build-v3 --unaccented-target` builds it: MorphGNT train, plus Swete, CCL lemmas, and the patristic Koine sentences. A MorphGNT-train unigram wins over an auxiliary count. The gap below is that restorer missing on gold letters. A trained model can land below either row.

## Ceiling

| Hypothesis | Ordered content lemma F1 | Ordered surface F1 | Objective |
| --- | ---: | ---: | ---: |
| Acute MorphGNT surface | 0.9239 | 1.0000 | 0.9543 |
| Bare gold + `KoineAccentRestorer` | 0.9207 | 0.9771 | 0.9433 |

Full-precision objectives are 0.9543147208121827 and 0.9432818963985686. The restorer moves lemma LCS from 455 to 453 and surface LCS from 961 to 939. It matches 939 of 961 dev surfaces (0.9771). The 22 misses are accent, breathing, or iota subscript. Two change a content lemma: `ὤν` to `ὧν` in John 3:4, and `οἰκτίρμων` to `οἰκτιρμῶν` in Luke 6:36.

A restorer built on MorphGNT train alone matches 924 of 961 surfaces (surface F1 0.9615, objective 0.9370, the same lemma F1 0.9207). Swete, CCL lemmas, and the patristic sentences close 15 of those 37 misses. That row is the same kind of ceiling.

The published NT holdout ordered content lemma F1 0.6334 and chrF++ 0.4747 stay the published polytonic model's holdout scores. This comparison leaves `unaccented_target: false` as it was published.
