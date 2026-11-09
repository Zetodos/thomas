# Greek Thomas fragments (P.Oxy.) — JAK-31

**Evidence type:** G (direct Greek manuscripts of Thomas)
**Corpus role:** `evaluation-witness` (hold out from JAK-28 Greek–Sahidic training unless waived)
**Mode:** `bundled` (offline after clone)

## Imported transcriptions

| Manifest | Manuscript | TM | License (verified) | TEI pin |
| --- | --- | --- | --- | --- |
| [`dclp-poxy-1`](../data/manifests/dclp-poxy-1.json) | P.Oxy. 1 | 62838 | **CC BY 3.0** (DCLP TEI `availability`) | `idp.data@d9387868bbe3` `DCLP/62/62838.xml` |
| [`dclp-poxy-654`](../data/manifests/dclp-poxy-654.json) | P.Oxy. 654 | 62840 | **CC BY 3.0** (same) | `DCLP/62/62840.xml` |
| [`dclp-poxy-655`](../data/manifests/dclp-poxy-655.json) | P.Oxy. 655 | 62839 | **CC BY 3.0** (same) | `DCLP/62/62839.xml` |

Each TEI file was checked individually for a Creative Commons Attribution 3.0 license ref. Do **not** treat DCLP materials as public domain by default. Manifest checksums are of the **committed** TEI after repo trailing-whitespace / EOF normalization (substantive text unchanged from the idp.data pin).

Derived layered corpus: [`data/sources/fragments/greek-thomas-fragments.json`](../data/sources/fragments/greek-thomas-fragments.json).

## Layers

| Layer | TEI source | Meaning |
| --- | --- | --- |
| `surviving` | plain text, `unclear`, diplomatic letters | Letters present on the papyrus |
| `lacuna` | `gap` | Lost text without a restoration in this encoding |
| `restoration` | `supplied` (+ `ex` abbreviation expansions) | Scholarly restorations / expansions — **not** surviving letters |

Editorial `app` uses **`lem` only**; `rdg` (Attridge, Fitzmyer, Hofius, …) are not promoted into layers. This is **not** redistribution of Layton/Attridge continuous critical Greek.

## Explicitly excluded

- **P.Oxy. 5575** (TM 989105) — early sayings parallel overlapping Synoptics and Thomas; **not** a manuscript of the *Gospel of Thomas*.
- Grenfell/Hunt print PD and Evelyn-White 1920 PD remain allowable alternatives (see audit §2) but were not re-keyed in this ingest while DCLP CC BY 3.0 TEI was available and verified.
- NA28/UBS, manuscript images, Brill/NHS continuous critical Thomas Greek — never-ship.

## Validation

```bash
uv run thomas sources validate
```

Rejects missing licenses/checksums on fragment manifests, dangling `source_id`s in the fragment corpus, mislabeled P.Oxy. 5575, and corpus_role / evidence_type drift.

Regenerate the derived JSON after TEI edits:

```bash
uv run thomas sources build-fragments
```

## Attribution

Digital Corpus of Literary Papyri (DCLP) / papyri.info; LDAB (Trismegistos). [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
