# Mapping artifacts (JAK-38)

Minimal phrase/construction mappings and verified loanwords for Phase 2.

| File | Evidence class | Notes |
| --- | --- | --- |
| `statistical-phrase-constructions-v1.json` | **statistical** | Train-only; dual-aligner (`eflomal` + `fast_align`) |
| `verified-loanwords-ccl-ddglc-v1.json` | **independently_verified** | CCL/DDGLC CC BY-SA 4.0; spot-checked |
| `ablation-nt-aligner-only-v1.json` | — | Arm A: NT aligner stack without verified loanwords |
| `ablation-nt-aligner-plus-loanwords-v1.json` | — | Arm B: arm A + verified loanwords |

Do not collapse statistical rows into verified loanword facts.

```bash
uv run thomas mappings validate
```

See [`docs/phrase-loanword-mappings.md`](../../../docs/phrase-loanword-mappings.md).
