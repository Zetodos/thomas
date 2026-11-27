# Thomas

Offline, deterministic, auditable **Koine Greek reconstruction (`Generated Greek`, Category C)** and witness-separated evidence benchmark for the *Gospel of Thomas*.

**Repository:** <https://github.com/Zetodos/thomas>
**Architecture:** The default track is the deterministic two-stage bilingual Coptic→Koine clause-transfer and beam-search pipeline (`model-v1`), offline after setup. Published beside it: from-scratch neural `model-v2`, Greekified + Patristic neural `model-v3`, and faithfulness-first coverage + constrained-beam neural `model-v4`. `thomas reconstruct run` and `thomas eval run` default to `model-v1`.

## Objective

Reconstruct grammatically coherent Koine Greek for all 115 units (`thomas.0`–`thomas.114`) of the Sahidic Coptic *Gospel of Thomas* while enforcing strict provenance boundaries:

1. **Attested Greek Fragments (Category G)** — P.Oxy. 1 / 654 / 655 DCLP EpiDoc TEI witnesses preserving surviving letters, unclear ink, lacunae, abbreviations, and editorial restorations as separate layers (`evaluation-witness` only — never leaked into training or decoding).
2. **Generated Greek (Category C)** — deterministic, offline reconstructions from Coptic. The default track is `model-v1`, with morpheme-level provenance (`L` loanword, `N` NT bitext, `P` OT/LXX bitext, `C` grammatical transfer). `model-v2`, `model-v3`, and `model-v4` are optional neural tracks (`--model v2`, `--model v3`, `--model v4`).
3. **Critical Greek** — Separately authored scholarly edition (not automated output).

The pipeline does **not** claim to recover a lost archetype with certainty; convergent and redactionally divergent P.Oxy. clauses are reported separately in [`docs/benchmark-report.md`](docs/benchmark-report.md).

## Corpora & Licensing

Project-owned Python code under `python/` and tests under `tests/` are licensed under Apache 2.0 (see [`LICENSE`](LICENSE)). Third-party corpora keep their upstream terms recorded under [`data/manifests/`](data/manifests/):

| Role | Source | License |
| --- | --- | --- |
| **Coptic Thomas (target)** | **Dilley / Coptic SCRIPTORIUM** (`dilley-thomas-v6.2.0.json`, 115 units / 6,516 morphemes) | CC BY 4.0 |
| **Sahidic NT** | **Horner** via CrossWire `CopSahHorner` | Public Domain |
| **Greek NT spine** | **MorphGNT / SBLGNT** (`morphgnt-sblgnt`, 27 books, gold lemmas + 8-slot parses) | CC BY 4.0 / CC BY-SA 3.0 |
| **Sahidic OT + Coptic morphology** | **Coptic SCRIPTORIUM Sahidic OT** (`sahidic.ot_TT.zip`) | CC BY-SA 4.0 |
| **LXX secondary bitext** | **Swete LXX** (`lxx-swete`) | Public Domain / MIT |
| **Loanwords & compounds** | **Comprehensive Coptic Lexicon v1.2 / DDGLC** | CC BY-SA 4.0 |
| **Strong's Greek** | **morphgnt/strongs-dictionary-xml** (`strongs-greek`) | CC0 1.0 |
| **Apostolic Fathers** | **Open Apostolic Fathers** (`apostolic-fathers`; CC BY-SA text already inside `model-v3` weights) | CC BY-SA 4.0 |
| **First1K Koine prose** | **First1KGreek** Epictetus and Philo (`first1k-koine-prose`; CC BY-SA text already inside `model-v3` weights) | CC BY-SA 4.0 |
| **Apophthegmata Patrum** | **Coptic SCRIPTORIUM** (`scriptorium-apophthegmata`) | CC BY 4.0 |
| **Greek Thomas fragments** | **DCLP P.Oxy. 1 / 654 / 655** (`greek-thomas-fragments.json`, `evaluation-witness`) | CC BY 4.0 / CC BY-SA 3.0 |

Gated corpora (Wells **Sahidica**, **NTAEG**, **Bohairica**, `UD_Coptic`) are **never** fetched, read, or bundled. Ordinary tracked and generated files stay strictly `< 1024 KiB` (`--maxkb=1024`). Neural weights for `model-v2`, `model-v3`, and `model-v4` (`data/mappings/model-v[234]/` `*.safetensors` and `*.model`) are Git LFS and sit outside that cap (`.gitattributes`, `.pre-commit-config.yaml`).

## Pipeline Overview

1. **Bidirectional Morphology (`python/zetodos_thomas/morphology/`):**
   - `CopticAnalyzer`: Finite-state + lexicon + trigram POS Viterbi bound-group segmenter & lemmatizer trained strictly on `sahidic.ot_TT.zip` + `CCL v1.2` (zero Thomas).
   - `GreekInflector`: Empirical MorphGNT paradigms + pure-Python regular Koine declensions/conjugations + MIT `greek-accentuation` + Koine phonological sandhi.
2. **Loanwords, Idioms & Alignment (`lexicon/`, `align/`):**
   - `CclLexicon`: Full DDGLC Greek loanword & light-verb compound (`ⲣ-`/`ϯ-`/`ϫⲓ-`) index (`363/363` Dilley `<lang="Greek">` coverage).
   - `LexicalTranslationModel`: Seeded bidirectional `eflomal` (3× NT domain weight over OT/Swete, excluding holdout, dev & Synoptic 5-grams) with count-smoothed bidirectional association and Sahidic vocalic fallback.
   - `IdiomAndValencyTable`: Multiword Sahidic phrasal patterns and MorphGNT verb/preposition case frames.
3. **Clause Transfer, LM & Beam Search (`reconstruct/`):**
   - Clause chunker + Coptic→Koine morphosyntactic transfer grammar (VBD speech formulas, `ⲛϭⲓ` postposed subjects, `οὐ μή` + subjunctive, attributive/articular participles, conditionals, modal infinitives, copular agreement).
   - Smoothed 3-gram Koine LM (`train` split + Swete LXX) + deterministic log-linear beam search (`beam_size=8`, weights tuned strictly on 51-verse NT `dev`).

`model-v2` and `model-v3` are separate from-scratch neural decoders. `model-v3` can condition on a leak-free `model-v1` draft. Their weights are the Git LFS files named above. The commands below still build and run `model-v1`.

## Verified Benchmark Results (`model-v1`)

See [`docs/benchmark-report.md`](docs/benchmark-report.md) and [`data/generated/eval-report-v1.json`](data/generated/eval-report-v1.json):

- **41 Intact NT Gospel Holdouts:** Coverage `1.0000`, Content Lemma F1 `0.5586` (`P=0.5700, R=0.5477`), All-Token Lemma F1 `0.5789`, Surface F1 `0.3335`, Morph Accuracy `0.5605`, chrF++ `0.2423`, BLEU-4 `0.0312`.
- **Span-Aligned, Lacuna-Aware P.Oxy. 1 / 654 / 655 (20 overlapping units):**
  - **Convergent Clauses (23 spans / 19 units / 312 surviving words):** Content Lemma Recall **`0.6432`**, Top-3 Recall **`0.6973`**, Content Lemma F1 **`0.5324`**, Surviving Char F1 (exact / Koine itacism) **`0.7116` / `0.7260`**, Loanwords **`1.0000` (14/14)**.
  - **Overall (29 spans / 20 units / 368 surviving words):** Content Lemma Recall `0.5849`, Content Lemma F1 `0.4901`, Surviving Char F1 (itacism) `0.6934`.

## Published neural tracks

`model-v1` stays the default. Three neural tracks are published and tracked:

- **`model-v2`** — from-scratch neural Coptic→Koine. Weights under `data/mappings/model-v2/` are Git LFS. Report: [`docs/benchmark-report-v2.md`](docs/benchmark-report-v2.md).
- **`model-v3`** — from-scratch Greekified + Patristic neural model. Training uses the full canonical NT train split (7,782 verses), Sahidic OT / Swete, Strong's Greek (CC0), SCRIPTORIUM Apophthegmata (CC BY 4.0), and the CC BY-SA 4.0 Apostolic Fathers and First1K Koine prose corpora. Those two CC BY-SA texts are already inside the weights under `data/mappings/model-v3/` (Git LFS). On the 41-verse NT holdout the published report records ordered content lemma F1 `0.6334` and chrF++ `0.4747`. Report: [`docs/benchmark-report-v3.md`](docs/benchmark-report-v3.md).
- **`model-v4`** — faithfulness-first 3-seed neural ensemble trained with morpheme-level cross-attention coverage + guided alignment supervision, VSTAT-separated eflomal lexicon + Sahidic orthographic gloss-hint fallbacks (exact `lemma|POS` and exact bare-lemma keys take precedence over `~orth`/`~ao` fallbacks), canonicalized prenominal prepositions, multi-verse 36–72 morpheme windows, faithful-length-filtered Swete/Patristic back-translation (`filter_faithful=True`, token ratio `0.42–1.30`, alongside NT holdout/dev 5-gram exclusion on patristic Greek), banked lexically-constrained beam search (word-boundary-verified hard-loan integrity plus nominal `COP` → present `εἰμί`, object-resumptive `CREL` → accusative `ὅς`, and exclamatory `PTC` → `οὐαί`; `243/243` per-instance inflected-paradigm / `237/243` strict MorphGNT exact-lemma loanword retention across all 115 sayings), and unbounded positional Pass-2 coverage repair with unparseable-surface and over-length guards. Weights under `data/mappings/model-v4/` are Git LFS. On the 41-verse NT holdout the published report records ordered content lemma F1 **`0.6528`** (`Bag=0.7166`), ordered surface F1 **`0.5335`**, morph accuracy **`0.7530`**, chrF++ **`0.4732`**, and P.Oxy. convergent clear-ink char F1 **`0.7370`** (surviving-only ordered lemma F1 **`0.4553`**, non-synoptic clear-ink **`0.7741`**). Report: [`docs/benchmark-report-v4.md`](docs/benchmark-report-v4.md).

## CLI Workflow

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+:

```bash
uv sync --locked

# One-time setup (downloads corpora/lexicons to data/downloads/ and builds local eflomal)
uv run thomas setup

# Build & validate normalized sources and zero-leakage NT splits
uv run thomas sources build
uv run thomas sources validate

# Train/freeze offline model-v1 (<1024 KiB shards under data/mappings/model-v1/)
uv run thomas model build

# Reconstruct Thomas (100% offline, deterministic)
uv run thomas reconstruct run --all
uv run thomas reconstruct run --saying thomas.2

# Run offline holdout & lacuna-aware P.Oxy. evaluation suite
uv run thomas eval run
```

`thomas reconstruct run`, `thomas reconstruct inspect`, and `thomas eval run` default to `--model v1`. `--model v2`, `--model v3`, and `--model v4` select the neural tracks (`thomas reconstruct inspect --saying thomas.7 --model v4` displays per-morpheme alignments). `thomas model build-v2`, `thomas model build-v3`, and `thomas model build-v4` train them. The committed `model-v2`, `model-v3`, and `model-v4` weights remain Git LFS.

Quality & zero-leakage checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run python scripts/check_spdx_headers.py python/zetodos_thomas/*.py
```
