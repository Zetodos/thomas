# Alignment pilot passage splits (JAK-28)

**Corpus:** [`data/sources/pilot/horner-vaticanus-nt-pilot.json`](../data/sources/pilot/horner-vaticanus-nt-pilot.json) (`horner-vaticanus-nt-pilot-v2`)  
**Machine splits:** [`data/sources/pilot/splits.json`](../data/sources/pilot/splits.json)  
**Greek:** NTVMR Vaticanus B\* only (GA 03 / `docID` 20003) — never WH/SBLGNT  
**Sahidic:** CopSahHorner v1.5 (Public Domain)

Splits are **whole passages**, never random words. Omission fixtures are kept in the corpus but excluded from positive lexical/phrase denominators.

## Split table

| Role | Passage | Units | Why |
| --- | --- | --- | --- |
| **train** | John 1:1–18 | 18 | Prologue; nomina sacra; sayings-like density |
| **train** | Mark 1:1–8 | 8 | Gospel opening; loanwords; Mark 1:1 heading **paratext** |
| **train** | Matthew 5:14–16 | 3 | Light/city sayings (Thomas-adjacent imagery) |
| **train** | Matthew 7:7–8 | 2 | Imperatives ask/seek/knock (Thomas 2/92/94 zone) |
| **dev** | Luke 6:20–21 | 2 | Beatitudes; pronouns; clause variation (Thomas 54 zone) |
| **holdout** | Matthew 13:31–32 | 2 | Mustard-seed parable whole-passage holdout (Thomas 20 zone) |
| **omission** | John 8:1–11 | 11 | Dual-empty PA; negative/omission fixture only |

**Totals:** 46 units · 35 nonempty positive pairs · 11 omission slots.

## Textual / versification notes

| Issue | Handling |
| --- | --- |
| Mark 1:1 Sahidic `ⲙⲁⲣⲕⲟⲥ` book heading | Paratext; not a Greek verse token |
| Mark 1:1 Greek `υιου θυ` vs shorter Horner | Edition mismatch / null on Sahidic — not an aligner bug alone |
| Matt 7:7 Horner `<ⲟⲩ>` restoration | Kept on `original_text`; stripped in `normalized_text` |
| John 1:15 Greek `[η]` unclear | Editorial mark on original; normalized drops wrappers |
| John 8:1–11 empty both sides | Omission fixture; never positive training/eval denominator |
| Luke 6:20 heavens vs `θυ` | Documented tradition difference |

## Leakage controls

- Greek Thomas fragment witnesses (notably logia **7 / 17 / 30 / 77**) are **not** in this NT stack (see JAK-31 / JAK-33 / JAK-34).
- Coptic Thomas (Dilley) is **not** absorbed here (JAK-30).
- Holdout Matthew 13:31–32 must not be used when fitting the lexical baseline.

## Rebuild

```bash
uv run thomas sources download
uv run thomas sources build-pilot
uv run thomas sources validate
uv run thomas align validate
```
