# Thomas

Offline, deterministic, auditable **Koine Greek reconstruction (`Generated Greek`, Category C)** and witness-separated evidence benchmark for the *Gospel of Thomas*.

**Repository:** <https://github.com/Zetodos/thomas>
**Architecture:** Deterministic two-stage bilingual Coptic→Koine clause-transfer & beam-search pipeline (`model-v1`), completely free of LLMs and network calls after setup.

## Objective

Reconstruct grammatically coherent Koine Greek for all 115 units (`thomas.0`–`thomas.114`) of the Sahidic Coptic *Gospel of Thomas* while enforcing strict provenance boundaries:

1. **Attested Greek Fragments (Category G)** — P.Oxy. 1 / 654 / 655 DCLP EpiDoc TEI witnesses preserving surviving letters, unclear ink, lacunae, abbreviations, and editorial restorations as separate layers (`evaluation-witness` only — never leaked into training or decoding).
2. **Generated Greek (Category C)** — 100% deterministic, offline reconstructions from Coptic (`model-v1`) with morpheme-level provenance (`L` loanword, `N` NT bitext, `P` OT/LXX bitext, `C` grammatical transfer).
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
| **Greek Thomas fragments** | **DCLP P.Oxy. 1 / 654 / 655** (`greek-thomas-fragments.json`, `evaluation-witness`) | CC BY 4.0 / CC BY-SA 3.0 |

Gated corpora (Wells **Sahidica**, **NTAEG**, **Bohairica**, `UD_Coptic`) are **never** fetched, read, or bundled. Every tracked/generated file under the repo stays strictly `< 1024 KiB` (`--maxkb=1024`).

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

## Verified Benchmark Results (`model-v1`)

See [`docs/benchmark-report.md`](docs/benchmark-report.md) and [`data/generated/eval-report-v1.json`](data/generated/eval-report-v1.json):

- **41 Intact NT Gospel Holdouts:** Coverage `1.0000`, Content Lemma F1 `0.5586` (`P=0.5700, R=0.5477`), All-Token Lemma F1 `0.5789`, Surface F1 `0.3335`, Morph Accuracy `0.5605`, chrF++ `0.2423`, BLEU-4 `0.0312`.
- **Span-Aligned, Lacuna-Aware P.Oxy. 1 / 654 / 655 (20 overlapping units):**
  - **Convergent Clauses (23 spans / 19 units / 312 surviving words):** Content Lemma Recall **`0.6432`**, Top-3 Recall **`0.6973`**, Content Lemma F1 **`0.5324`**, Surviving Char F1 (exact / Koine itacism) **`0.7116` / `0.7260`**, Loanwords **`1.0000` (14/14)**.
  - **Overall (29 spans / 20 units / 368 surviving words):** Content Lemma Recall `0.5849`, Content Lemma F1 `0.4901`, Surviving Char F1 (itacism) `0.6934`.

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

Quality & zero-leakage checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run python scripts/check_spdx_headers.py python/zetodos_thomas/*.py
```
