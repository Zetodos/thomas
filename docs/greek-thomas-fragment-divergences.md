# Greek Thomas fragment divergences from Coptic (JAK-33)

**Policy:** The primary Thomas edition follows the **Coptic** (NHC II) sequence and division. Greek fragment forms that diverge are **separate witnesses** — never silently merged into Coptic units. Machine-readable catalog: [`data/evidence/divergences/greek-thomas-fragment-divergences-v1.json`](../data/evidence/divergences/greek-thomas-fragment-divergences-v1.json).

**Coptic source id (JAK-30):** `dilley-thomas`  
**Fragment corpus id (JAK-31):** `greek-thomas-fragments-v1`  
**Corpus role:** `evaluation-witness` (excluded from Greek–Sahidic training mappings — see [`benchmark-holdouts.md`](benchmark-holdouts.md))

## Documented cases

### P.Oxy. 1 vs Coptic Thomas 30 / 77 (division + sequence)

On P.Oxy. 1, material corresponding to Coptic logion **30** continues into the stone/wood clause that Coptic places in logion **77**, in a contiguous Greek sequence. Coptic keeps **30** and **77** as separate sayings, with **77** much later in the NHC II order.

| Side | Handling |
| --- | --- |
| Coptic | Keep `thomas.30` and `thomas.77` as distinct Dilley units |
| Greek (P.Oxy. 1) | Keep manuscript-local line/span ids; map explicitly to both Coptic units when comparing |
| Scoring | Do **not** rewrite Coptic boundaries to match the papyrus, or collapse the papyrus into one Coptic saying |

Papyrus coverage (JAK-31): roughly logia **26–30**, then **77b**, then **31–33** — not Coptic order.

### P.Oxy. 654 (coverage)

Incipit + logia **1–7**. Partial witness only. Evaluate overlapping Coptic units; do not impute missing Greek from Coptic or from modern continuous critical editions. Surviving letters ≠ restorations (JAK-31 layers).

### P.Oxy. 655 (coverage / uncertain catalog mapping)

Covers roughly logia **36–39**; some catalogs discuss possible **24** material. Record uncertainty; do not force a silent unit-id merge.

### P.Oxy. 5575 (not a Thomas manuscript)

Independent early sayings parallel overlapping Synoptics and Thomas traditions. **Excluded** from Thomas fragment ingest (JAK-31). Do not label as evidence type G for the *Gospel of Thomas*.

## What this document is not

- Not a letter-by-letter critical apparatus (JAK-28 / editorial phases).
- Not permission to train on fragment Greek when evaluating reconstruction of those same sayings.
- Not multi-edition Greek NT policy (Vaticanus B\* only for NT spine).
