# SPDX-License-Identifier: Apache-2.0
"""Script and spelling helpers for the parallel dictionary.

Coptic→Greek letter mapping here is the shared alphabet (orthography), not a
gloss list. No Coptic word is paired with a Greek word by hand.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from zetodos_thomas.align.segment_coptic import GRAMMATICAL_PREFIXES

# Extra closed-class clitics the Horner segmenter leaves fused on purpose
# (so ``ⲁⲩⲱ`` stays whole). Used only to decide where a bound form may be cut.
_BRIDGE_CLITICS: tuple[str, ...] = (
    "ⲉϥ",
    "ⲁϥ",
    "ⲛϥ",
    "ⲙϥ",
    "ⲉⲥ",
    "ⲁⲥ",
    "ⲛⲥ",
    "ⲉⲩ",
    "ⲁⲩ",
    "ⲛⲩ",
    "ⲉⲕ",
    "ⲁⲕ",
    "ⲛⲕ",
    "ⲉⲓ",
    "ⲁⲓ",
    "ⲛⲓ",
    "ⲙⲙⲟ",
    "ⲉⲣⲟ",
    "ϥ",
    "ⲥ",
    "ⲕ",
)

_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        {*GRAMMATICAL_PREFIXES, *_BRIDGE_CLITICS},
        key=len,
        reverse=True,
    )
)

# Coptic block letters that continue a Greek letter. Coptic-only letters are
# omitted on purpose: a form that needs them is not a Greek-script loan.
_COPTIC_TO_GREEK: dict[str, str] = {
    "ⲁ": "α",
    "ⲃ": "β",
    "ⲅ": "γ",
    "ⲇ": "δ",
    "ⲉ": "ε",
    "ⲍ": "ζ",
    "ⲏ": "η",
    "ⲑ": "θ",
    "ⲓ": "ι",
    "ⲕ": "κ",
    "ⲗ": "λ",
    "ⲙ": "μ",
    "ⲛ": "ν",
    "ⲝ": "ξ",
    "ⲟ": "ο",
    "ⲡ": "π",
    "ⲣ": "ρ",
    "ⲥ": "σ",
    "ⲧ": "τ",
    "ⲩ": "υ",
    "ⲫ": "φ",
    "ⲭ": "χ",
    "ⲯ": "ψ",
    "ⲱ": "ω",
}

COPTIC_ONLY: frozenset[str] = frozenset("ϣϥϩϫϭϯ")


def _is_coptic_or_greek_letter(ch: str) -> bool:
    code = ord(ch)
    return (
        0x2C80 <= code <= 0x2CFF
        or 0x03E2 <= code <= 0x03EF
        or 0x0370 <= code <= 0x03FF
        or 0x1F00 <= code <= 0x1FFF
    )


def _strip_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def fold_coptic(text: str) -> str:
    """Letters only: NFC, no combining marks, no punctuation."""
    folded = _strip_marks(unicodedata.normalize("NFC", text.casefold()))
    return "".join(ch for ch in folded if _is_coptic_or_greek_letter(ch))


def fold_greek(text: str) -> str:
    """Accent-free Greek letters; final sigma folded to medial sigma."""
    folded = _strip_marks(unicodedata.normalize("NFC", text.casefold()))
    letters = "".join(ch for ch in folded if _is_coptic_or_greek_letter(ch))
    return letters.replace("ς", "σ")


def transliterate_coptic(coptic: str) -> str:
    """Map Greek-shared Coptic letters onto Greek. Coptic-only letters drop."""
    letters = fold_coptic(coptic)
    return "".join(_COPTIC_TO_GREEK.get(ch, "") for ch in letters)


def common_prefix_len(left: str, right: str) -> int:
    n = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        n += 1
    return n


def levenshtein(left: str, right: str) -> int:
    """Edit distance. Inputs are short wordforms."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left
    prev = list(range(len(right) + 1))
    for i, ch_l in enumerate(left, start=1):
        cur = [i]
        for j, ch_r in enumerate(right, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ch_l != ch_r)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def same_paradigm(left: str, right: str) -> bool:
    """True when two attested Greek forms are inflections of one stem.

    Lengths must be close, so a short word is not absorbed into a longer one
    that merely begins with the same letters.
    """
    if not greek_forms_similar(left, right):
        return False
    a = fold_greek(left)
    b = fold_greek(right)
    longer = max(len(a), len(b))
    shorter = min(len(a), len(b))
    return longer > 0 and shorter / longer >= 0.66


def greek_forms_similar(left: str, right: str) -> bool:
    """True when two Greek wordforms share a stem, not only a short prefix."""
    a = fold_greek(left)
    b = fold_greek(right)
    if len(a) < 4 or len(b) < 4:
        return False
    if a == b:
        return True
    pref = common_prefix_len(a, b)
    shorter = min(len(a), len(b))
    if pref >= 4 and pref / shorter >= 0.6:
        return True
    return shorter >= 5 and pref >= 3 and levenshtein(a, b) <= 2


def transliteration_similar(coptic: str, greek: str) -> bool:
    """True when the Coptic form is a Greek loan shape of ``greek``."""
    return greek_forms_similar(transliterate_coptic(coptic), greek)


def loan_shape(coptic: str) -> bool:
    """Greek-alphabet Coptic form long enough to be a loan candidate.

    This is an orthographic filter. It does not assign a Greek gloss.
    """
    letters = fold_coptic(coptic)
    if len(letters) < 5:
        return False
    return not any(ch in COPTIC_ONLY for ch in letters)


def spelling_variants(coptic: str) -> tuple[str, ...]:
    """Systematic Sahidic spelling alternations. Identity is first.

    Only the ei/i diphthong collapse and one global o/omega swap. Callers
    must abstain when the variants disagree.
    """
    base = fold_coptic(coptic)
    if not base:
        return ()
    found: list[str] = [base]
    if "ⲉⲓ" in base:
        found.append(base.replace("ⲉⲓ", "ⲓ"))
    elif "ⲓ" in base:
        found.append(base.replace("ⲓ", "ⲉⲓ"))
    if "ⲱ" in base:
        found.append(base.replace("ⲱ", "ⲟ"))
    elif "ⲟ" in base:
        found.append(base.replace("ⲟ", "ⲱ"))
    out: list[str] = []
    for item in found:
        if item and item not in out:
            out.append(item)
    return tuple(out)


@lru_cache(maxsize=8192)
def is_prefix_chain(text: str) -> bool:
    """True when ``text`` is only closed-class prefixes/clitics."""
    if text == "":
        return True
    for prefix in _PREFIXES:
        if text.startswith(prefix) and is_prefix_chain(text[len(prefix) :]):
            return True
    return False


def loan_stem(coptic: str, greek: str) -> str | None:
    """Stem of a fused Coptic form when that stem transliterates to ``greek``.

    Returns None when the link is not a loan-shaped cut. The Greek string is
    one the aligner already proposed; this function does not invent it.
    """
    folded = fold_coptic(coptic)
    if len(folded) < 5 or not fold_greek(greek):
        return None
    if transliteration_similar(folded, greek):
        return None
    for cut in range(1, len(folded) - 3):
        left, right = folded[:cut], folded[cut:]
        if len(right) < 4:
            break
        if is_prefix_chain(left) and transliteration_similar(right, greek):
            return right
    return None
