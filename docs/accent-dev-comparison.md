# Accent target on the NT dev split

`model-v3` is published with `unaccented_target: false` (`data/mappings/model-v3/seq2seq-config.json`). The rerank weights in `data/mappings/model-v3/rerank-weights.json` are already a point on the ordered-LCS dev grid. This note records the accent-flag comparison that grid did not cover.

The scored set is the 51-verse NT dev split in `data/training/splits/nt-split-v1.json`: John 3:1–21 and Luke 6:20–49. The accent restorer is built with those 51 verses and the 41 holdout verses excluded. The papyrus corpus is unused. No seq2seq weights were trained or replaced. The published holdout tables in `docs/benchmark-report-v3.md` and `data/generated/eval-report-v3.json` are unchanged.

## Objective

The objective is the one in `tune_rerank_weights_on_dev`: `0.60 * ordered content lemma F1 + 0.40 * ordered surface F1`, micro-averaged over the 51 verses. Content POS is `{N-, V-, A-, D-}`. Hypothesis lemmas come from the committed model-v3 Greek analyzer. Gold lemmas and surfaces come from MorphGNT.

The polytonic row is the acute-only normalized MorphGNT surface, which is the direct training target. The unaccented row strips that surface and restores it with `KoineAccentRestorer`, built the way `thomas model build-v3 --unaccented-target` builds it: MorphGNT train, plus Swete, CCL lemmas, and the patristic Koine sentences. A MorphGNT-train unigram wins over an auxiliary count. These are realized targets. They are the strings a model would be scored on if it emitted its target exactly.

## Result

| Setting | Ordered content lemma F1 | Ordered surface F1 | Objective |
| --- | ---: | ---: | ---: |
| Polytonic (`unaccented_target: false`) | 0.9239 | 1.0000 | **0.9543** |
| Unaccented + `KoineAccentRestorer` | 0.9207 | 0.9771 | 0.9433 |

Dev selects **polytonic** (`unaccented_target: false`). The objective gap is 0.0110, from full-precision scores 0.9543147208121827 and 0.9432818963985686.

The analyzer was fit without these 51 verses, so its reading of the correct surfaces has ordered content lemma LCS 455 against 519 gold lemmas (predicted 466). Both rows use that analyzer. The accent flag moves lemma LCS from 455 to 453 and surface LCS from 961 to 939. The restorer matches 939 of 961 dev surfaces (0.9771). The 22 misses are accent, breathing, or iota subscript. Two change a content lemma: `ὤν` to `ὧν` in John 3:4, and `οἰκτίρμων` to `οἰκτιρμῶν` in Luke 6:36.

Building the restorer on MorphGNT train alone still selects polytonic (objective 0.9370, surface F1 0.9615, the same lemma F1 0.9207).

The published NT holdout ordered content lemma F1 0.6334 and chrF++ 0.4747 are the scores of this polytonic setting. They were left as published.
