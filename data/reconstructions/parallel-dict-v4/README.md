# parallel-dict-v4

Horner Sahidic NT × Vaticanus B* (6,991 train sentences) plus Sahidic OT ×
Swete LXX (18,435 verse pairs). Both aligners. The NT dev, holdout, and
John 8 omission verses stay out of training.

OT text: Coptic SCRIPTORIUM `sahidic.ot` v6.2.0, CC BY-SA 4.0 (Göttingen /
CoptOT). Greek: Swete via `nathans/lxx-swete` `26bad3eb`, CC BY-SA 4.0
(First1KGreek). This dictionary is CC BY-SA 4.0 because it is adapted from
those texts. No handwritten Coptic→Greek pairs. No verified-loanword table.
No Thomas Greek in the fit.

Verses are joined by book, chapter, and verse. A pair is dropped when one
side is more than five times as long as the other.

| Metric | Result |
| --- | --- |
| Loanword stem match | 10/11 = 0.909 |
| Loanword exact match | 6/11 = 0.545 |
| Surviving content-type recall | 28/149 = 0.188 |
| Content emission precision | 0.156 |
| Abstain rate | 0.436 |

Not sufficient. Exact loanword match is about half, and under a fifth of
surviving fragment content types are recovered.
