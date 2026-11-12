# SPDX-License-Identifier: Apache-2.0
"""Inferred-lemma heuristics for JAK-37 (explicitly non-morphological).

These are **proxies**, not MorphGNT / SCRIPTORIUM lemmas. Consumers must read
``inference_method`` and treat this layer separately from observed wordforms.
"""

from __future__ import annotations

import unicodedata

COPTIC_METHOD = "normalized_surface_as_lemma_proxy_v1"
GREEK_METHOD = "nfd_strip_combining_marks_v1"


def infer_coptic_lemma(wordform: str) -> str:
    """Proxy lemma = already-normalized Coptic surface (identity).

    Bound-group segmentation already peels clitics into separate tokens in
    bitext; we do not invent further morphology here.
    """
    return wordform


def infer_greek_lemma(wordform: str) -> str:
    """Strip combining diacritics (accents/breathings) after NFD; keep letters.

    Does **not** claim to recover dictionary lemmas (no stemmer / MorphGNT).
    """
    nfd = unicodedata.normalize("NFD", wordform)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def inference_method_payload() -> dict[str, str]:
    """Stable method labels for the inferred-lemma layer."""
    return {
        "coptic": COPTIC_METHOD,
        "greek": GREEK_METHOD,
        "disclaimer": (
            "Inferred proxies only — not attested morphological lemmas; "
            "do not collapse with observed_wordform layer."
        ),
    }
