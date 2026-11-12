# Redistributable source texts

Committed, legally redistributable excerpts and imports live here.

| Path | Contents |
| --- | --- |
| [`pilot/horner-vaticanus-nt-pilot.json`](pilot/horner-vaticanus-nt-pilot.json) | Horner × Vaticanus B* NT corpus **v3** (407 units / 396 nonempty; Gospel-first; JAK-43) |
| [`pilot/horner-vaticanus-nt-pilot-v2.json`](pilot/horner-vaticanus-nt-pilot-v2.json) | Archival Phase 1 pilot v2 (46 units) |
| [`thomas/dilley-thomas-v6.2.0.json`](thomas/dilley-thomas-v6.2.0.json) | **Unit-level** Dilley / SCRIPTORIUM Coptic Thomas (prologue + 114 logia; orig+norm; JAK-30). Not token-complete / not alignment-ready. |
| [`fragments/`](fragments/) | Greek Thomas P.Oxy. fragments (DCLP CC BY 3.0; evidence type G / evaluation-witness) |

Phrase-level alignment stubs (JAK-32) used by offline tests live under
[`tests/fixtures/alignment/`](../../tests/fixtures/alignment/). ID grammar and
migration rules: [`docs/id-scheme.md`](../../docs/id-scheme.md).

Train/dev/holdout roles live under [`data/training/splits/`](../training/splits/) (**v3** = JAK-43; archival v2 = JAK-33), not here. Omission, eval, and divergence catalogs: [`data/evidence/`](../evidence/).

Full module / TEI / TT downloads stay under gitignored [`data/downloads/`](../downloads/) after explicit setup. See [`docs/horner-vaticanus-alignment-notes.md`](../../docs/horner-vaticanus-alignment-notes.md), [`docs/greek-thomas-fragments.md`](../../docs/greek-thomas-fragments.md), and [`docs/source-licenses.md`](../../docs/source-licenses.md).
