# parallel-dict-v3

Whole Horner Sahidic NT × Vaticanus B* (6,991 training sentences), both
aligners. The v3 dev, holdout, and John 8 omission verses stay out of
training. No handwritten glosses, no verified-loanword table, no Thomas Greek
in the fit.

Same decoder and fragment score as v1. Context inflection stays on because
held-out NT token precision rose (0.610 → 0.636). Orthographic snap stays off.

| Metric | Result |
| --- | --- |
| Loanword stem match | 10/11 = 0.909 |
| Loanword exact match | 5/11 = 0.455 |
| Surviving content-type recall | 18/149 = 0.121 |
| Content emission precision | 0.124 |
| Abstain rate | 0.574 |

Not sufficient against the pre-set bars (content recall 0.20, precision 0.25).
The whole licensed Horner × Vaticanus NT is in this fit.
