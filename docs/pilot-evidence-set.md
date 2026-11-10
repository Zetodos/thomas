# Phase 1 pilot evidence set — Thomas 7, 17, 30, 77 (JAK-34)

Inspectable structured data spanning PLAN.md evidence types **G / L / N / P**.
This is **not** a polished critical edition.

**Package:** [`data/evidence/pilot/thomas-7-17-30-77.json`](../data/evidence/pilot/thomas-7-17-30-77.json)

**JSONL:** [`data/evidence/pilot/thomas-7-17-30-77.jsonl`](../data/evidence/pilot/thomas-7-17-30-77.jsonl)

**Schema:** `PilotEvidenceSet` in `zetodos_thomas.schemas.evidence`

**IDs:** [`docs/id-scheme.md`](id-scheme.md) (JAK-32)

## Baselines (unchanged)

| Role | Source id | Notes |
| --- | --- | --- |
| Coptic Thomas | `dilley-thomas` | Dilley / SCRIPTORIUM v6.2.0 units `thomas.7` / `.17` / `.30` / `.77` (JAK-30) |
| Greek fragments (G) | `dclp-poxy-*` | P.Oxy. 1 / 654 / 655; `evaluation-witness` (JAK-31) |
| Greek NT spine | `ntvmr-vaticanus-b` | Vaticanus B\* only — **do not retarget** |

## Per-saying attested vs missing

| Saying | Attested / recorded | Missing / placeholder |
| --- | --- | --- |
| **7** | Dilley Coptic unit; **G** P.Oxy. 654 **partial coverage** (`poxy-654-dclp`, ~lines 40–42; word-level deferred); **L** ⲙⲁⲕⲁⲣⲓⲟⲥ ← μακάριος | **N**, **P**; other P.Oxy. mss; G word-level line/span IDs |
| **17** | Dilley Coptic unit; **N** 1 Cor 2:9 as **Greek verbal parallel** (not proof Paul quoted Thomas — PLAN.md §3.2) | **G**, **L**, **P**; 1 Cor verse `TextUnit` not yet in NT pilot ranges |
| **30** | Dilley Coptic unit; **G** P.Oxy. 1 **partial coverage**; divergent-sequence witness alignment with 77 | **L**, **P**; **N** semantic placeholder only; G word-level spans |
| **77** | Dilley Coptic unit; **G** P.Oxy. 1 **77b partial coverage**; same divergent-sequence witness alignment with 30 | **L**, **N**, **P**; G word-level spans |

## G fidelity (partial coverage ≠ word attestation)

Pilot **G** rows that only link a whole DCLP transcription + Coptic `thomas.N`
use `status=partial-coverage`, **not** `attested`. Example: P.Oxy. 654 lines
40–42 for Thomas 7 mix surviving letters with extensive scholarly supplies.
Word-level verification (manuscript-local line/span IDs + surviving/restored
flags) is deferred so later reconstruction-confidence assignment is not
overstated. Fragment token layers already distinguish surviving / lacuna /
restoration in the corpus itself (**JAK-31**).

## P.Oxy. 1 vs Coptic 30 / 77

Recorded as `witness_alignments[]` entry `v1:witness/poxy-1-coptic-30-77` with
`kind=divergent-sequence`. The Greek papyrus juxtaposes material for Coptic 30
and 77b; the pilot **does not** silently merge that into one Coptic reading.

Longer holdout / divergence documentation: **JAK-33** (link, do not duplicate).

## Validate offline

```bash
uv run thomas sources validate
```

Cross-checks Dilley unit ids, fragment transcription ids, required G/L/N/P
slots, G partial-coverage fidelity for whole-transcription links, the Thomas 17
N policy note, and the 30/77 divergent witness alignment.
