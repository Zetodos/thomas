# Stable unit / token / span IDs (JAK-32)

Span-first linking for Coptic Thomas, Greek fragments, and Greek–Sahidic pairs
must not rely on string matching. This document defines the **v1** ID grammar
and the retokenization migration rule.

**Greek NT spine:** Vaticanus B\* (INTF/NTVMR GA 03 firsthand) only — JAK-25.
**Sahidic NT baseline:** CopSahHorner — JAK-24.
**Verse pairs:** existing `PilotCorpus` / `ParallelUnit` (unchanged).
**Phrase rows:** `AlignmentDocument` + `Parallel` in `zetodos_thomas.schemas`.

## Scheme version

| Constant | Value | Meaning |
| --- | --- | --- |
| `ID_SCHEME_VERSION` | `1` | Current minting grammar in `zetodos_thomas.schemas.ids` |

Every minted token, span, and parallel id is prefixed with `v1:`.
`AlignmentDocument.id_scheme_version` must equal the code constant or
validation fails (forces an explicit migration).

## ID grammar (v1)

| Kind | Pattern | Example |
| --- | --- | --- |
| Token | `v1:{source_id}/{unit_id}/t/{sequence}` | `v1:ntvmr-vaticanus-b/john.1.1/t/4` |
| Span | `v1:{source_id}/{unit_id}/s/{local}` | `v1:copsahhorner/john.1.1/s/pshaje` |
| Parallel | `v1:parallel/{local}` | `v1:parallel/john.1.1-logos` |

- `{sequence}` is the **0-based** order within the unit’s token layer.
- `{local}` is a short slug (no spaces, no `/`).
- `TextUnit.unit_id` remains the unit handle (`john.1.1`, `thomas.7`, …).
- Presentation changes (CSS, Unicode compatibility glyphs, HTML) **must not**
  change these ids.

## What is stable vs what is not

| Stable across | Not stable across |
| --- | --- |
| Display orthography / fonts | Retokenization that splits or merges tokens |
| Website highlighting markup | Changing `source_id` or `unit_id` assignment |
| Adding lemma/POS annotations onto an existing token | Reordering the token sequence |

## Empty and discontinuous spans

- **Empty** (`SpanKind.EMPTY`): `token_ids == []`. Used for shared absences
  (e.g. John 8:1–11 PA empty on both Horner and Vaticanus B\*).
- **Discontinuous** (`SpanKind.DISCONTINUOUS`): ordered `token_ids` whose
  sequences contain gaps (Coptic↔Greek word-order divergence).
- **Contiguous** (`SpanKind.CONTIGUOUS`): sequences form a dense range.

Many-to-many mappings are expressed by listing multiple span ids on each side
of a `Parallel`, or by multiple `Parallel` rows sharing spans.

## Illegal overlaps

Spans that set the same `exclusive_group` **and** the same
`(source_id, unit_id)` must not share any token ids. Validation rejects such
overlaps. Spans outside an exclusive group may overlap freely (needed for
overlapping phrase hypotheses before JAK-28 adjudication).

Dangling token/span ids, cross-unit token membership, duplicate ids, and
kind/shape mismatches are also rejected by `AlignmentDocument` validation.

## Retokenization migration rule (v1)

**Bump the scheme version and rewrite.**

When token boundaries change:

1. Increment `ID_SCHEME_VERSION` in `python/zetodos_thomas/schemas/ids.py`.
2. Remint all token / span / parallel ids under the new prefix.
3. Rewrite every `AlignmentDocument` JSON (and any dependents) so
   `id_scheme_version` matches.
4. Do **not** attempt fuzzy salvage of old `v1:` ids onto new tokens.

In-place migration tables are out of scope for v1; a future scheme may add an
optional mapping file if editorial cost justifies it.

## Fixtures

| Path | Role |
| --- | --- |
| `data/sources/pilot/horner-vaticanus-nt-pilot.json` | Verse-level Horner × Vaticanus B\* pairs (JAK-24) |
| `tests/fixtures/alignment/horner-vaticanus-john1.1-phrase.json` | Phrase-level stub from that pilot |
| `tests/fixtures/alignment/thomas-7-unit-stub.json` | Coptic Thomas unit + Greek fragment stub |

Offline tests load these fixtures without network access.
