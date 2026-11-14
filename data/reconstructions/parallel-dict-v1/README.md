# parallel-dict-v1

Coptic→Koine dictionary induced from the train split of Horner Sahidic ×
Vaticanus B\* (`eflomal` and `fast_align` together). Thomas Greek is not an
input. The verified loanword table is not an input. No Coptic→Greek pair is
hand-entered.

| File | Role |
| --- | --- |
| `dictionary.json` | Emissions that clear the posterior bar, plus attested Greek types |
| `thomas-reconstruction.json` | Dilley logia. `g: null` is an abstention |
| `metrics.json` | Fragment score. The loanword audit is evaluation-only |

Rebuild:

```bash
uv run thomas dictionary build
```

A statistical emission requires both aligners. Inflections of one Greek stem
share a posterior; the string emitted is the most frequent attested surface in
that cluster. Unrelated candidates still abstain. Orthographic snap (transliterate
an unseen loan-shaped form onto a Greek type attested in train) is computed, then
turned on only if NT dev precision clears a pre-set bar. This run leaves it off.

Fragment score uses surviving letters of P.Oxy. 1, 654, and 655. Restorations
are a diagnostic field only. See `metrics.json` for the numbers and the next
iteration. JAK-40 artifacts are not modified.
