# SPDX-License-Identifier: Apache-2.0
"""Tests for CCL v1.2 + DDGLC Greek loanword ingest and Thomas coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from zetodos_thomas.ingest.dilley_thomas import load_thomas_corpus
from zetodos_thomas.ingest.holdouts import load_manifest_sha256
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon

REPO_ROOT = Path(__file__).resolve().parents[1]
CCL_XML = (
    REPO_ROOT
    / "data"
    / "downloads"
    / "ccl-ddglc"
    / "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
)
CCL_MANIFEST = REPO_ROOT / "data" / "manifests" / "ccl-ddglc-tei.json"
THOMAS_JSON = REPO_ROOT / "data" / "sources" / "thomas" / "dilley-thomas-v6.2.0.json"


def test_ccl_lexicon_nfc_and_thomas_greek_loanword_coverage(tmp_path: Path) -> None:
    if not CCL_XML.is_file():
        pytest.skip("CCL XML not downloaded in this test environment")
    expected_sha = load_manifest_sha256(CCL_MANIFEST)
    lex = CclLexicon.from_tei_xml(CCL_XML, expected_sha256=expected_sha)
    assert lex.source_sha256 == expected_sha

    with pytest.raises(ValueError, match="sha256 mismatch"):
        CclLexicon.from_tei_xml(CCL_XML, expected_sha256="0" * 64)

    # Canonical Thomas loanwords in NFC polytonic Greek
    assert any(
        e.greek_lemma == "ἑρμηνεία"
        for e in lex.resolve(norm="ϩⲉⲣⲙⲏⲛⲉⲓⲁ", lemma="ϩⲉⲣⲙⲏⲛⲉⲓⲁ", pos="N", lang="Greek")
    )
    assert any(
        e.greek_lemma == "μακάριος"
        for e in lex.resolve(norm="ⲙⲁⲕⲁⲣⲓⲟⲥ", lemma="ⲙⲁⲕⲁⲣⲓⲟⲥ", pos="N", lang="Greek")
    )
    assert any(
        e.greek_lemma == "τόπος"
        for e in lex.resolve(norm="ⲧⲟⲡⲟⲥ", lemma="ⲧⲟⲡⲟⲥ", pos="N", lang="Greek")
    )
    assert any(
        e.greek_lemma == "νηστεύω"
        for e in lex.resolve(norm="ⲣⲛⲏⲥⲧⲉⲩⲉ", lemma="ⲣⲛⲏⲥⲧⲉⲩⲉ", pos="V", lang="Greek")
    )

    # Verify > 90% type & token resolution across all Dilley Thomas <lang="Greek">
    corpus = load_thomas_corpus(THOMAS_JSON)
    gk_tokens = [
        m
        for u in corpus.units
        for m in u.morphemes
        if m.lang == "Greek" and m.pos != "FM"
    ]
    resolved = sum(
        1
        for m in gk_tokens
        if lex.resolve(norm=m.norm, lemma=m.lemma, pos=m.pos, lang=m.lang)
    )
    rate = resolved / max(len(gk_tokens), 1)
    print(
        f"\n[CCL Thomas <lang=Greek> coverage] {resolved}/{len(gk_tokens)} = {rate:.4f}"
    )
    assert rate >= 0.90, f"Loanword coverage {rate:.4f} < 0.90"

    # JSON round-trip stays strictly < 1024 KiB
    out = tmp_path / "ccl_loanwords.json"
    lex.save_json(out)
    assert out.stat().st_size < 1024 * 1024
    reloaded = CclLexicon.load_json(out)
    assert reloaded.source_sha256 == expected_sha
    assert any(
        e.greek_lemma == "μακάριος"
        for e in reloaded.resolve(
            norm="ⲙⲁⲕⲁⲣⲓⲟⲥ", lemma="ⲙⲁⲕⲁⲣⲓⲟⲥ", pos="N", lang="Greek"
        )
    )
