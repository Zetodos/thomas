# Stable unit / token / span IDs (JAK-32)

Span-first linking for Coptic Thomas, Greek fragments, and Greek–Sahidic pairs
must not rely on string matching. This document defines the **v1** ID grammar
and the retokenization migration rule.

**Greek NT spine:** Vaticanus B\* (INTF/NTVMR GA 03 firsthand) only — JAK-25.
**Sahidic NT baseline:** CopSahHorner — JAK-24.
**Verse pairs:** existing `PilotCorpus` / `ParallelUnit` (unchanged).
**Phrase rows:** `AlignmentDocument` + `Parallel` in `zetodos_thomas.schemas`.
**Thomas ingest:** `ThomasCorpus` uses the **same** `Token` / `Span` models
(no second stub shape).

## Scheme version

| Constant | Value | Meaning |
| --- | --- | --- |
| `ID_SCHEME_VERSION` | `1` | Current minting grammar in `zetodos_thomas.schemas.ids` |

Every minted token, span, and parallel id is prefixed with `v1:`.
`AlignmentDocument.id_scheme_version` / `ThomasCorpus.id_scheme_version` must
equal the code constant or validation fails (forces an explicit migration).

Validation also parses each token/span/parallel id and rejects:

- wrong scheme (`v0:` inside a v1 document),
- source/unit encoded in the id that disagrees with model fields,
- token ids whose encoded sequence ≠ `Token.sequence`.

## ID grammar (v1)

| Kind | Pattern | Example |
| --- | --- | --- |
| Token | `v1:{source_id}/{unit_id}/t/{sequence}` | `v1:ntvmr-vaticanus-b/john.1.1/t/4` |
| Span | `v1:{source_id}/{unit_id}/s/{local}` | `v1:copsahhorner/john.1.1/s/pshaje` |
| Parallel | `v1:parallel/{local}` | `v1:parallel/john.1.1-logos` |

- `{sequence}` is the **0-based** order within the unit’s token layer.
- `{local}` is a short slug (no spaces, no `/`).
- `TextUnit.unit_id` remains the unit handle (`john.1.1`, `thomas.0`, `thomas.7`, …).
- Presentation changes (CSS, Unicode compatibility glyphs, HTML) **must not**
  change these ids.

## Thomas prologue and logia

| Unit | `unit_id` | `chapter` | `kind` |
| --- | --- | --- | --- |
| Prologue | `thomas.0` | `0` | `prologue` |
| Logion N (1–114) | `thomas.N` | `N` | `logion` |

`TextUnit.chapter` allows `0` specifically so the SCRIPTORIUM prologue chapter
label can be preserved without inventing a fake chapter 1. NT verse units keep
`chapter >= 1` in practice via their OSIS-style ids. Later token ids must use
`v1:dilley-thomas/thomas.0/t/0` (etc.); do not mint `thomas.N.stub0`.

## Shared Token / Span (ingest PRs must rebase)

Canonical models: `zetodos_thomas.schemas.alignment.Token` and `.Span`
(field name `id`, plus `source_id`, span `kind`, etc.).

Do **not** add a second Token/Span with `token_id` / `span_id` fields in
`schemas.source`, and do **not** mint `thomas.N.stub0`-style ids. JAK-30 Dilley
ingest should:

1. Keep `thomas.0` / `chapter=0` / `kind=prologue` (logia `kind=logion`).
2. Emit shared `ThomasCorpus` with exact units `thomas.0`…`thomas.114`.
3. **Unit-level package:** leave `tokens[]` / `spans[]` empty and label notes
   as unit-level / not alignment-ready (current `data/sources/thomas/` import).
4. When tokens are ingested later, mint with `token_id(source, unit, sequence)`
   from **per-token** SCRIPTORIUM orig↔norm pairs — never from
   `whitespace_tokens()` positional zip (concatenated orig ≠ segmented norm).

## Whitespace helper limits

`zetodos_thomas.align.tokenize.whitespace_tokens` is a fixture helper for
Horner/Vaticanus-style text where whitespace token counts match between
`original_text` and `normalized_text`. It raises `TokenCountMismatchError`
on Dilley-like mismatch and always records `char_*` offsets into
`original_text`.

## Empty and discontinuous spans

- **Empty** (`SpanKind.EMPTY`): `token_ids == []`. Used for shared absences
  (e.g. John 8:1–11 PA empty on both Horner and Vaticanus B\*).
- **Discontinuous** (`SpanKind.DISCONTINUOUS`): ordered `token_ids` whose
  sequences contain gaps (Coptic↔Greek word-order divergence).
- **Contiguous** (`SpanKind.CONTIGUOUS`): sequences form a dense range.

`token_ids` must be **strictly increasing** by token sequence (rejects
`[t/2, t/0, t/1]` and reversed discontinuous spans).

Many-to-many mappings are expressed by listing multiple span ids on each side
of a `Parallel`, or by multiple `Parallel` rows sharing spans.

## Illegal overlaps

Spans that set the same `exclusive_group` **and** the same
`(source_id, unit_id)` must not share any token ids. Validation rejects such
overlaps. Spans outside an exclusive group may overlap freely (needed for
overlapping phrase hypotheses before JAK-28 adjudication).

Dangling token/span ids, cross-unit token membership, duplicate ids, and
kind/shape mismatches are also rejected by `AlignmentDocument` /
`ThomasCorpus` validation.

## Retokenization migration rule (v1)

**Bump the scheme version and rewrite.**

When token boundaries change:

1. Increment `ID_SCHEME_VERSION` in `python/zetodos_thomas/schemas/ids.py`.
2. Remint all token / span / parallel ids under the new prefix.
3. Rewrite every `AlignmentDocument` / `ThomasCorpus` JSON (and any dependents)
   so `id_scheme_version` matches.
4. Do **not** attempt fuzzy salvage of old `v1:` ids onto new tokens.

In-place migration tables are out of scope for v1; a future scheme may add an
optional mapping file if editorial cost justifies it.

## Fixtures

| Path | Role |
| --- | --- |
| `data/sources/pilot/horner-vaticanus-nt-pilot.json` | Verse-level Horner × Vaticanus B\* pairs (JAK-24) |
| `data/sources/thomas/dilley-thomas-v6.2.0.json` | **Unit-level** Dilley SCRIPTORIUM Coptic Thomas (JAK-30; not token-complete / not alignment-ready) |
| `tests/fixtures/alignment/horner-vaticanus-john1.1-phrase.json` | Phrase-level stub from that pilot |
| `tests/fixtures/alignment/thomas-7-unit-stub.json` | Phrase-alignment stub using `dilley-thomas` source id |

Offline tests load these fixtures without network access.
