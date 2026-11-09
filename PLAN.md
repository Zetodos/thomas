# Zetodos Thomas — Implementation Plan

**Repository:** <https://github.com/Zetodos/thomas>
**Status:** Proposed implementation plan; validate reconstruction feasibility before committing to a full engine
**Target:** Version 0.1, an open-source, offline-capable, evidence-based reconstruction and reader for the *Gospel of Thomas*

## 1. Objective

Develop a reproducible **Koine Greek reconstruction of the Gospel of Thomas**, primarily reconstructing the Greek text that could underlie the extant Coptic manuscript. Publish both:

1. **Generated Greek (C):** A deterministic, Coptic-derived reconstruction using an approved, general Coptic-to-Greek translation model and rules.
2. **Critical Greek:** A manually curated reconstruction that evaluates the generated Greek against surviving Greek fragments, Greek loanwords preserved in Coptic, Greek New Testament/Septuagint parallels, and other early Greek attestations.

Provide a static web reader that exposes the Coptic text, the generated and critical Greek, their rough English translations, and the ancient evidence for each logion. The reader must support **bidirectional phrase alignment** and **search**.

The project is not claiming to recover a uniquely certain lost Greek original. It must distinguish what is attested from what is reconstructed, and must document disagreements among ancient witnesses.

## 2. Non-negotiable decisions

- **Python owns all work upstream of the website:** data acquisition/ingestion, linguistic analysis, dictionary construction, alignment, reconstruction, editorial validation, export, and website-data generation.
- **TypeScript is used only in `web/`** for the static website. Do not mix Python and TypeScript source files in the same subdirectories.
- Python must be **strictly, statically typed**: complete annotations for public interfaces, no untyped application boundaries, `ty` type checking, `ruff` formatting/linting, `uv` dependency/environment management, `pytest` tests, and **Typer** for the CLI. Isolate untyped libraries behind typed adapters.
- **Offline after clone and setup:** dependency packages, optional Ollama models, corpora, and other permitted resources may be downloaded during explicit setup. Subsequent CLI commands, site builds, and local website use must not need network access, cloud APIs, hosted databases, remote fonts, or CDNs.
- **No LLM in the deterministic reconstruction path.** Optional local models may help *propose* lexical/construction mappings or editorial annotations, but accepted results must be stored and versioned before deterministic use.
- All original text, normalized analysis, source editions, reconstruction inputs, generated outputs, critical decisions, and alignments have explicit provenance and stable identifiers.
- **Human-edited critical Greek is never overwritten by regeneration.** Generated Greek and critical Greek are separate artifacts.
- The generated C reconstruction must not simply copy Greek from Thomas's Greek fragments or selected Thomas-specific canonical/patristic parallels. Those are separately assessed for the critical edition. General Greek–Coptic parallel corpora may train the language mapping model, with safeguards against evaluation leakage.
- Store possible alternative Greek **spans** (including one word, multiple words, or a clause), but defer an elaborate alternatives UI until experience with actual reconstructions justifies it.
- No evidence-type filtering UI is required. Show the evidence clearly on each logion's page.
- Downloadable **public text editions** are the **critical Greek reconstruction** and its **rough English translation**. Other source/research data may remain inspectable in the open-source repository but need not be exposed as additional website downloads.
- Build for all 114 logia in the data model. Do not equate having provisional machine output with completed scholarly review.

## 3. Evidence policy and reconstruction target

### 3.1 What is being reconstructed

The primary edition follows the sequence and division of the extant Coptic Thomas. Divergent Greek forms (for example, the differing relationship of P.Oxy. 1 to Coptic Thomas 30/77) are preserved as separate witnesses and evaluated in the critical apparatus, not silently merged.

### 3.2 Evidence categories

| Code | Kind | Policy |
| --- | --- | --- |
| **G** | Direct Greek manuscripts of Thomas | Preserve actually surviving letters, lacunae, and editorial restorations separately. |
| **L** | Greek-derived words in Coptic | Strong lexical clues; not automatic proof of Greek case, tense, number, or syntax. |
| **N** | Greek NT and Septuagint parallels | Record exact spans and distinguish lexical/verbal agreement from semantic similarity. |
| **P** | Greek attestations in other early literature | Distinguish a quotation of *Thomas* from a shared or independently transmitted saying. |
| **C** | Coptic-derived computational reconstruction | Hypothesis produced by the approved mapping model; not an independent historical witness. |
| **R** | Modern scholarly retrotranslation (optional) | Comparative research note only, **not independent ancient evidence**; incomplete and not required. |

Do **not** treat this order as a universal numeric confidence ranking. A precise early quotation may support a phrase more strongly than a merely thematic canonical parallel. Evidence types and confidence are separate fields. For example, **1 Corinthians 2:9 is a Greek verbal parallel for Thomas 17**, not proof that Paul quoted the completed *Gospel of Thomas*.

### 3.3 Editorial safeguards

- Preserve original witness text separately from normalized tokenization and scholarly restoration.
- Identify editions and variants for source passages; do not silently treat one modern critical edition as identical to a manuscript.
- Permit multiple evidence records per reconstructed span and multiple possible Greek spans per Coptic span.
- Require written rationale for manual changes from generated Greek to critical Greek.
- Mark provisional, reviewed, uncertain, and unresolved readings explicitly.

## 4. Research resources and licensing gate

**Before bundling or downloading each resource, verify current availability, coverage, editorial quality, and redistribution terms.** A webpage being accessible is not permission to package its contents.

| Resource or source family | Intended use | Due diligence |
| --- | --- | --- |
| Coptic SCRIPTORIUM Thomas (Dilley edition) | Core Coptic text and existing annotations | Obtain original/manuscript-oriented display text, normalized text, tokenization; confirm version, attribution, CC license. |
| P.Oxy. 1, 654, 655 | Direct Greek fragments | Use a legally reusable transcription; record manuscript, fragment, lacunae, restorations, and edition. |
| Other Greek sayings witnesses (e.g., P.Oxy. 5575) | Independent parallels | Do not mislabel as a manuscript of Thomas. |
| Sahidic Greek–Coptic NT parallels | Primary mapping/training corpus | Confirm exact aligned coverage, dialect, Greek edition, and redistribution rights. In particular, review Sahidica's custom license. |
| Sahidic OT and LXX parallels | Broader lexical/construction evidence | Verify access and rights for each text and reconstructed section. |
| Bohairic Greek–Coptic parallels | Secondary supplementary corpus | Keep dialect metadata; do not mix with Sahidic without explicit control. |
| Coptic lexicons, Greek-loanword indexes, DIOSKOROS | Lexical candidate generation | Check machine access and licenses; preserve authoritative source citations. |
| Greek NT, LXX, early Christian Greek corpora | Source parallels, linguistic checking | Prefer redistributable editions; verify exact version and license of each. |
| Public-domain English translations (e.g., WEB where appropriate) | English for cited Greek passages | Check passage coverage and maintain translation-to-edition references. |
| Existing modern Greek retrotranslations | Optional comparison only | No purchase or dependence assumed; honor copyright. |

Keep a machine-readable **source manifest** with canonical ID, title, editor/edition, textual witness, dialect/language, source URL, version/date, rights/attribution, checksum, local location, and availability mode (`bundled`, `setup-download`, or `manual-import`). If a source cannot be redistributed, use legally permitted excerpts or an explicit optional import rather than assuming it is part of the clone.

**Setup invariant:** the reproducible working dataset must be present locally *before* an offline run. No hidden data fetching during build or reconstruction.

## 5. Reconstruction architecture

### 5.1 Pipeline A — Build the approved mapping model (research mode)

**Inputs:** Aligned ancient Greek–Coptic texts (Sahidic preferred), Coptic morphology/lemmas, Greek morphology/lemmas, verified loanword correspondences, and lexicon evidence.

1. **Ingest and segment:** preserve source spellings and stable token IDs; extract normalized Coptic grammatical units and Greek tokens/lemmas.
2. **Align parallel passages:** pair Greek/Coptic verses or clauses from compatible editions. Track verse-level agreement and textual differences.
3. **Learn lexical correspondences:** evaluate a local bidirectional statistical aligner (e.g., `eflomal`) to propose Coptic lemma ↔ Greek lemma mappings, with observed frequencies and confidence estimates.
4. **Extract phrase/construction correspondences:** learn recurring multiword expressions and Coptic constructions corresponding to Greek inflectional, verbal, negative, relative, and pronominal structures.
5. **Incorporate independently verified lexicon/loanword entries:** use them as constraints or stronger evidence, never as unexplained substitutions.
6. **Validate and review candidates:** automatically flag contradictions, unsupported correspondences, dialect mismatches, and rare/unseen constructions. A local Ollama model may suggest candidates, but does not approve them.
7. **Freeze an approved model:** version the accepted lexical mappings, phrase mappings, grammatical templates, weights, and underlying training-corpus identifiers. The deterministic engine reads this fixed model only.

**Required mapping-model components:**

- `lexical`: Coptic lemma/span → one or more Greek lemmas, conditioned on POS/context where supported.
- `constructions`: Coptic grammatical pattern → Greek morphosyntactic patterns, with restrictions and alternatives.
- `phrases`: Coptic multiword span → candidate Greek multiword expressions, with source occurrences.
- `observations`: observed examples and alignments; never fabricate attestation counts.
- `version`: exact rules/weights/source snapshot/checksums and model schema version.

### 5.2 Pipeline B — Deterministic Coptic → Greek (production mode)

**Inputs:** One normalized/annotated Coptic Thomas logion plus the approved mapping model.

1. Enumerate lexical and phrase candidates for Coptic spans.
2. Match compatible grammatical construction templates.
3. Compose Greek phrase candidates, preferring historically attested forms where appropriate.
4. Validate morphology, agreement, source coverage, and unsupported additions/deletions.
5. Score candidates using **published, fixed** weights for lexical evidence, construction plausibility, grammatical validity, and source fidelity; penalize unexplained additions/omissions. Do not optimize merely for similarity to familiar canonical Gospel wording.
6. Use a deterministic search/decoding strategy (beam search is a candidate; justify choice after feasibility tests). Apply stable tie-breaks and sorting.
7. Emit the selected provisional Greek, ranked alternative **spans**, confidence/uncertainty, and the Coptic↔Greek alignment. Permit unresolved structures rather than inventing certainty.

**Outputs:** machine-readable generated Greek and span alignments, plus reports explaining the mapping decisions. For identical data, config, lockfiles, and approved model version, outputs must match byte-for-byte on supported environments.

### 5.3 Pipeline C — Critical edition (editorial mode)

A human maintainer evaluates generated Greek against **G, L, N, and P** evidence. For every divergence from the baseline, store the chosen Greek span, its aligned Coptic span, citations, rejected/plausible alternatives where relevant, editorial rationale, and review status. Critical Greek is the source of truth for public downloads.

### 5.4 English translations

- Maintain a **rough English translation of generated Greek** and a **rough English translation of critical Greek**, versioned and aligned to their actual Greek texts.
- v0.1 does **not** require a third automatic Greek→English translation engine. English can be drafted with optional local tools and maintained/reviewed as version-controlled data.
- Greek text edits should fail validation if the corresponding English translation is stale or unacknowledged.
- External Greek passage translations should be recorded by source and license, not silently reused from unrelated English editions.

## 6. Data contracts (design now; elaborate only when needed)

Use **Pydantic v2** models for validation, JSON Schema for the website boundary, and serialized JSON/JSONL for machine products. Human-maintained files may use a readable format (JSON or YAML), but choose one per data family and enforce schema validation. Avoid ad hoc free-text references for relationships.

### 6.1 Required entities

| Entity | Essential fields |
| --- | --- |
| `Source` | Stable ID, language/dialect, work, witness/edition, citation, license, checksum, content location |
| `TextUnit` | Source ID, unit ID (logion/verse/section), sequence, source text, normalized text (when applicable) |
| `Token` | Stable ID, surface, normalized form, lemma/POS when known, source span, sequence, parent grammatical group if any |
| `Span` | Stable ID, text unit, ordered token IDs or interval; support discontinuous and empty spans where justified |
| `Parallel` | Thomas span(s), external source span(s), relationship type, degree of agreement, provenance, notes |
| `Evidence` | Category G/L/N/P/R, source citation, source span, target reconstruction span, certainty, restrictions |
| `MappingObservation` | Coptic and Greek spans, alignment, source training passage, dialect, candidate score |
| `MappingRule` | Lexical/phrase/construction pattern, possible outputs, conditions, support/contradictions, approval/version |
| `GeneratedReading` | Logion, Greek tokens/spans, source alignments, alternatives, confidence/status, model version |
| `CriticalReading` | Logion, chosen tokens/spans, Coptic alignments, evidence links, editorial rationale, status |
| `EnglishReading` | Linked generated/critical/parallel Greek edition, English spans, alignment, revision status |

### 6.2 Design principles

- **Span-first, not string matching:** relationships may be many-to-many, one-to-many, zero-to-one, or discontinuous; Greek word order may differ from Coptic.
- **Stable IDs:** survive presentation changes and allow links in the browser; explicitly version retokenization migrations.
- **Original vs normalized:** keep the original Coptic display form and normalized/segmented form for linguistic processing; these are *representations of one text*, not an additional competing Coptic edition/UI feature.
- **Alternatives:** variable-length Greek spans, not a schema that assumes one candidate per Coptic word. A full variants UI is out of scope for v0.1.
- **Provenance and uncertainty:** distinguish observed witnesses, algorithmic suggestions, and editorial selection. Never store an AI inference as an ancient citation.
- **Validation:** reject dangling IDs, invalid/overlapping spans where not permitted, missing source licenses, unacknowledged stale translations, and undocumented editorial overrides.

## 7. Proposed repository layout

```text
thomas/
├── Plan.md
├── README.md
├── LICENSE                    # Code license; data licenses documented individually
├── pyproject.toml              # uv, Typer, ruff, ty, pytest, etc.
├── uv.lock
├── python/
│   └── zetodos_thomas/        # Typed Python packages only
│       ├── cli/
│       ├── schemas/
│       ├── ingest/
│       ├── align/
│       ├── lexicon/
│       ├── reconstruct/
│       ├── editorial/
│       ├── export/
│       └── integrations/      # Typed third-party adapters; optional Ollama
├── data/
│   ├── manifests/             # Rights, versions, URLs, checksums
│   ├── sources/               # Redistributable source texts / imported licensed excerpts
│   ├── training/              # Greek–Coptic pairs and observed alignments
│   ├── mappings/              # Approved dictionary, phrases, constructions, rules
│   ├── evidence/              # Thomas-specific fragments, parallels and cross-refs
│   ├── generated/             # Reproducible Coptic→Greek baseline
│   ├── critical/              # Human-edited Greek and editorial apparatus
│   └── english/               # Rough aligned translations
├── tests/                     # Python tests, fixtures, evaluation sets
├── web/                       # TypeScript-only web application
│   ├── package.json
│   ├── src/
│   └── public/               # Generated, validated static site data
├── exports/                   # Generated public Greek / English text editions
├── scripts/                   # Shell/bootstrap entrypoints only, if necessary
└── docs/                      # Methodology, evidence policy, licenses, evaluations
```

`web/public/` is generated by the Python site-data exporter; the TypeScript web build only consumes it. Avoid checked-in duplicate mutable sources of truth. Decide for each generated output whether it is committed for easy browsing or reproducibly built and ignored, and document that choice.

## 8. CLI contract (proposed, refine after feasibility)

Expose one Typer CLI (e.g. `thomas`) with cohesive subcommands:

```bash
# Explicit one-time setup; may access network and download optional models
uv sync --locked
uv run thomas setup [--with-models]

# Local/offline research and model building
uv run thomas sources validate
uv run thomas corpus build
uv run thomas mappings propose
uv run thomas mappings validate
uv run thomas mappings freeze

# Fully deterministic, offline production commands
uv run thomas reconstruct [--logion 17 | --all]
uv run thomas critical validate
uv run thomas export texts
uv run thomas export web
uv run thomas verify-offline

# Tests and toolchain
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

These are **planned command names**, not assertions that implementations already exist. The `--logion` / `--all` interface should be implemented using a clear, Typer-compatible option design. CI must validate that running `reconstruct`, `export`, and `verify-offline` never invokes network access or an Ollama/cloud service.

## 9. Implementation sequence for Cursor

Each phase should end with a runnable demonstration, tests, documentation, and an explicit **done** gate. **Do not build the entire website or train a neural model before Phase 2 demonstrates feasible reconstruction.**

### Phase 0 — Repository foundation and source audit

**Tasks**

- [ ] Create/inspect `Zetodos/thomas` repository and establish code license, contributing guidance, and documented source-license policy.
- [ ] Set up pinned Python with `uv`, `ruff`, `ty`, `pytest`, Typer, and Pydantic; enforce strict annotations and CI checks.
- [ ] Create the Python-only and TypeScript-only directory boundaries.
- [ ] Inventory Thomas Coptic and Greek fragments, candidate Sahidic NT/OT paired corpora, lexicons, source translations, and licensed English parallels.
- [ ] Record licenses, provenance, coverage, dialect, download/setup method, and checksums in source manifests.
- [ ] Implement minimum typed source-import models and a source-validation command.

**Done when:** a clean checkout can complete documented one-time setup, validate a small legally sourced corpus, pass type/lint/unit checks, and continue offline.

### Phase 1 — Small aligned corpus and benchmark fixtures

**Tasks**

- [ ] Import a small but representative legally usable Greek–Sahidic NT sample and the corresponding Greek edition.
- [ ] Ingest the annotated Coptic Thomas and preserve original and normalized forms.
- [ ] Ingest short, verifiable Greek fragments of Thomas, preserving lacunae/restorations.
- [ ] Define stable unit/token/span IDs and build **verse/phrase** alignments in a consistent schema.
- [ ] Choose benchmark holdouts by whole passages/books, not random isolated words; document any Greek fragment divergence from the Coptic.
- [ ] Prepare a small pilot set spanning distinct evidence types: **Thomas 7, 17, 30, and 77**.

**Done when:** paired source passages and target Thomas sayings can be inspected as structured, validated data, with training/test exclusions known in advance.

### Phase 2 — Reconstruction feasibility experiment **(critical decision gate)**

**Tasks**

- [ ] Run a local bidirectional statistical word aligner (evaluate `eflomal` before choosing).
- [ ] Generate an initial Coptic→Greek lexical candidate table with observed examples and confidence estimates.
- [ ] Add minimal phrase/construction mappings and manually verified loanword correspondences.
- [ ] Implement a minimal deterministic decoder using attested Greek forms, simple grammatical constraints, and stable ranking.
- [ ] Reconstruct held-out Greek–Sahidic passages and a selected Thomas pilot logion without consulting its hidden Greek witness.
- [ ] Report lemma recovery, morphology, word order, phrase fidelity, additions/omissions, uncertainty, and common failure modes. Compare with an elementary word-for-word baseline.
- [ ] Evaluate (but do not adopt by default) a local Ollama candidate-suggestion workflow for cases the aligner cannot handle.

**Decision gate:** Does the non-LLM method produce useful, auditable reconstructions that justify a more complete rules/candidate-search engine? If not, determine whether missing corpora, improved phrase extraction, morphology, or offline model assistance is the limiting factor. Document the evidence before selecting the next architecture. **Do not hard-code an accuracy target without measuring the baseline and establishing a fair evaluation set.**

### Phase 3 — Approved model and full deterministic reconstruction

**Tasks**

- [ ] Define approved lexical, phrase, and grammatical mapping schemas with source provenance and review workflow.
- [ ] Ingest additional *permitted* Sahidic NT/OT pairs; keep Bohairic and other dialect data explicitly labeled.
- [ ] Extend construction support for verb forms, articles, relative clauses, pronouns, negation, and multiword expressions.
- [ ] Evaluate Greek morphology analysis/generation libraries; use attested wordforms where possible.
- [ ] Implement deterministic candidate composition, scoring, tie-breaking, alternatives, and unresolved-span handling.
- [ ] Freeze approved mapping model version and source/checksum metadata.
- [ ] Generate a provisional reconstruction with phrase alignments for all 114 logia; mark gaps honestly.
- [ ] Add byte-for-byte regression tests and offline-network-denial tests.

**Done when:** all logia are processed without network or LLM access, every result is schema-valid and reproducible, and unsupported reconstructions are clearly distinguishable from attested text.

### Phase 4 — Historical evidence and human critical edition

**Tasks**

- [ ] Index all relevant Greek Thomas fragments with manuscript-level distinctions.
- [ ] Index Greek loanwords in Coptic Thomas, preserving lexical and morphological uncertainty.
- [ ] Record exact Greek NT/Septuagint parallels and Greek patristic attestations at phrase level; distinguish direct quotation, shared wording, semantic parallel, and thematic association.
- [ ] Create the critical-reading format and commands for editorial comparison/validation.
- [ ] Manually review the pilot sayings against G/L/N/P, documenting decisions and uncertain readings.
- [ ] Expand critical review across the 114 logia incrementally; provisional sections remain labeled.
- [ ] Add rough English aligned to both generated and critical Greek; validate translation freshness.

**Done when:** the pilot logia have traceable critical readings and English translations, no editorial data is overwritten by regeneration, and the remaining logia have explicit review states.

### Phase 5 — Static website and publication exports

**Tasks**

- [ ] Implement a Python exporter generating validated, versioned JSON for the website.
- [ ] Create a static TypeScript web app (Vite is an acceptable default; decide on framework based on simplest implementation).
- [ ] Show, per logion: original Coptic; generated Greek and rough English; critical Greek and rough English; relevant external Greek passages and their attributed English translations; manuscript/loanword/parallel evidence and editorial notes.
- [ ] Add **interactive bidirectional phrase highlighting** across the Coptic, each Greek text, and applicable English.
- [ ] Add local, client-side full-text search over the 114 sayings; no search server.
- [ ] Display provenance and provisional/unresolved status without burying material uncertainty.
- [ ] Generate downloadable `.txt` editions for the **critical Greek** and **rough critical English** only; include edition metadata and provisional-reading convention.
- [ ] Ensure all site assets are local and local serving/builds work offline.

**Done when:** a user can open any pilot logion, navigate aligned phrases in both directions, see supporting evidence, search, and download synchronized text editions with no runtime backend.

### Phase 6 — Version 0.1 completion and release checks

**Tasks**

- [ ] Confirm all 114 sayings have source data and a visible reconstruction or explicitly unresolved status.
- [ ] Validate cross-reference targets, translation alignments, source citations, and edition provenance.
- [ ] Verify reproducible CLI outputs and committed lockfiles; test in a network-disabled environment after setup.
- [ ] Run benchmark and regression suite; publish a concise accuracy/limitations report.
- [ ] Publish source and data licenses, contribution/editing procedure, setup instructions, and release changelog.
- [ ] Build the static website and text downloads from a clean checkout.

**Done when:** v0.1 is reproducible and usable, uncertainties are accurately represented, and no feature relies on a hidden cloud service.

## 10. Website scope

**Required for v0.1**

- Browse all 114 logia; each has a stable URL.
- Display original Coptic and generated/critical Greek, with rough English for both.
- Show Greek textual evidence and ancient Greek parallels with attributed English translations where legally available.
- Show cross-references as aligned phrases and provide bidirectional click/hover highlighting.
- Search logion text locally.
- Show editorial/uncertainty status and meaningful reasoning for divergences.
- Download the critical Greek and rough critical English as text.

**Design for, but defer**

- Dedicated alternative-reading comparison UI (schema supports variable-length alternate spans).
- Advanced multi-witness comparison and chronology displays.
- Evidence-type filters, accounts, backend, collaborative editor, commentary system, OCR, or a general-purpose Nag Hammadi translator.
- Dedicated model training or a Greek→English machine translation engine.

## 11. Test strategy

1. **Unit:** source normalization, token/span IDs, alignment composition, morphology adapters, deterministic rankings, Unicode handling.
2. **Schema/integrity:** all foreign keys, source manifests/licenses, spans, citation references, approved rules, English alignment freshness.
3. **Corpus benchmark:** reconstruct held-out Greek from Coptic using no direct access to the test Greek; report lemma recovery, morphology, phrase/order metrics, additions/omissions, and uncertainty. Avoid relying only on a single text-similarity score.
4. **Thomas benchmark:** compare reconstruction against actually preserved Greek fragment letters, not editors' restorations as if they were preserved; separately classify textual divergences.
5. **Leakage controls:** exclude held-out test material and any direct copied phrases from training mappings/phrase tables when evaluating recovery; record exact exclusions.
6. **Regression:** saved expected Greek, English alignment, and editorial data; changed rules must produce an explicit diff.
7. **Reproducibility:** same approved model/input yields the same hashes; stable sort order and deterministic tie-breaking.
8. **Offline:** run representative commands and web build after setup with network access blocked; optional Ollama unavailable must not break standard operation.
9. **Web:** bidirectional selection, search, correct evidence display, linkable logia, source/translation syncing, downloads.

## 12. Risks and decisions to revisit

| Risk | Mitigation / decision point |
| --- | --- |
| Insufficient legally reusable Greek–Sahidic paired data | Phase 0/1 inventory; obtain permissible excerpts or explicit import instructions before training. |
| Word aligner produces plausible but false equivalences | Preserve corpus observations and uncertainty; require review, validate on held-out passages. |
| Greek morphology not uniquely recoverable from Coptic | Keep candidate spans and uncertainty; do not claim impossible precision. |
| Statistical system overcopies familiar NT wording | Keep Thomas-specific parallels outside the baseline reconstruction and apply leakage controls. |
| Fragment Greek and Coptic edition diverge | Preserve witness identities and conflicting readings; primary target remains extant Coptic sequence. |
| LLM output changes across versions/hardware | LLM is only an optional research assistant; approved mappings are committed and reconstruction itself is deterministic. |
| Third-party licensing limits text distribution | Per-source manifests, appropriate excerpts, attribution, optional local imports, and no assumed blanket rights. |
| Critical text and English become inconsistent | Alignment checks and explicit stale-translation validation. |
| Full coverage mistaken for reviewed certainty | Clearly separate provisional machine output, human review, and unresolved spans. |
| Excess architecture before empirical learning | Keep Phase 2 a strict feasibility gate; implement only enough schema and CLI to measure the method. |

## 13. Instructions for Cursor / implementation discipline

- Work **one phase at a time**; do not jump ahead to web styling or model training before Phase 2 results are reviewed.
- Start each phase with a concise task list and identify the exact files/data contracts it changes.
- Never fabricate manuscript text, lexical attestations, scholarly references, corpus permissions, confidence values, or benchmark scores. Represent missing evidence as missing.
- Prefer reusing open linguistic tools and corpora to rewriting sophisticated analyzers. Isolate third-party integrations behind typed Python adapters.
- Separate **source observations**, **learned hypotheses**, **approved mapping rules**, **generated reconstructions**, and **editorial decisions** in code and in stored data.
- Keep tests, strict typing, licensing checks, documentation, and offline guarantees current with every feature.
- Avoid unnecessary frameworks/services; favor simple, inspectable code and stable text-based files.
- If research invalidates an assumption, document the findings and amend this plan rather than building around an unverified premise.

## 14. Immediate next action

**Implement Phase 0 and the smallest possible Phase 1 corpus import needed to start Phase 2.** The first meaningful deliverable is not a polished website: it is an inspectable Greek–Sahidic dataset, a reproducible lexical/phrase-alignment experiment, and a benchmark report demonstrating whether useful Greek can be recovered from withheld Coptic passages.

Only after that feasibility result should Cursor plan the production-grade reconstruction decoder in detail.
