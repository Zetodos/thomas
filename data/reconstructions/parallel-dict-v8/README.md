# parallel-dict-v8

Same Horner × Vaticanus and Sahidic OT × Swete bitext as parallel-dict-v4
(25,426 train sentences). No new verse pairs. Both aligners. An eflomal link
is kept only when a second sample agrees. The dictionary is CC BY-SA 4.0.

No handwritten Coptic→Greek pairs. No verified-loanword table. No Thomas
Greek in the fit or in the choice of switches. Switches stay on only when
held-out NT content recall or token precision rises and precision does not
drop by more than 0.015.

What the NT dev gate kept: local context, pooled suffix transfer, phrases,
and loan spelling. A loan-shaped token uses the attested Greek string its
letters already spell when context did not choose a case. A phrase may add
the other Greek tokens from a 2–4 word window. It does not replace a direct
emission. Paradigm extras stayed off.

| Metric | Result |
| --- | --- |
| Loanword stem match | 10/11 = 0.909 |
| Loanword exact match | 6/11 = 0.545 |
| Truncated-prefix misses, still counted as exact misses | 1 |
| Surviving content-type recall | 31/149 = 0.208 |
| Content emission precision | 0.137 |
| Abstain rate | 0.257 |

`dictionary.json` is a lossy summary of selected emissions. A later fit writes
`model.json` with the full inference state. The overlap numbers above are an
exploratory, non-positional diagnostic. They are not a reconstruction-fidelity
gate, and they are not a blind. JAK-40 and JAK-44 stay the official freeze.
