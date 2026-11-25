# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CCL native Egyptian x Strong's Greek bridge & patristic loaders."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from zetodos_thomas.ingest.apophthegmata import parse_apophthegmata_tt
from zetodos_thomas.ingest.koine_patristics import (
    load_all_patristic_koine_sentences,
    load_first1k_koine_sentences,
)
from zetodos_thomas.lexicon.gloss_bridge import (
    _extract_en_content_words,
    _is_meta_gloss,
    _pos_compatible,
    build_ccl_strongs_bridge,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = REPO_ROOT / "data" / "downloads"


def test_gloss_bridge_helpers_and_pos_filter() -> None:
    ws = _extract_en_content_words("to open, unfastening (lit. doors)")
    assert "open" in ws
    assert "unfasten" in ws
    assert _pos_compatible("V", "V-") is True
    assert _pos_compatible("V", "N-") is False
    assert _pos_compatible("N", "N-") is True
    # Lexicographic uncertainty & placeholder meta-glosses must be rejected
    assert _is_meta_gloss("unknown") is True
    assert _is_meta_gloss("meaning uncertain") is True
    assert _is_meta_gloss("a certain plant") is True
    assert _is_meta_gloss("fourth month of the year") is True
    assert _is_meta_gloss("tree, wood") is False


def test_gloss_bridge_rejects_meta_glosses_and_reconstructed_lemmas(
    tmp_path: Path,
) -> None:
    ccl_xml = tmp_path / "ccl.xml"
    ccl_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
  <entry>
    <form type="lemma"><orth>ϣϫⲓⲗ</orth><usg type="geo">S</usg></form>
    <gramGrp><pos>Subst.</pos></gramGrp>
    <sense><quote xml:lang="en">meaning unknown</quote></sense>
  </entry>
  <entry>
    <form type="lemma"><orth>*ⲛⲉⲙⲟⲩⲣⲉ</orth><usg type="geo">S</usg></form>
    <gramGrp><pos>Subst.</pos></gramGrp>
    <sense><quote xml:lang="en">tree</quote></sense>
  </entry>
  <entry>
    <form type="lemma"><orth>ⲃⲱ</orth><usg type="geo">S</usg></form>
    <gramGrp><pos>Subst.</pos></gramGrp>
    <sense><quote xml:lang="en">tree</quote></sense>
  </entry>
</body></text></TEI>
""",
        encoding="utf-8",
    )
    strongs_xml = tmp_path / "strongs.xml"
    strongs_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<strongsdictionary><entries>
  <entry><greek unicode="ἄγνωστος"/><strongs_def>unknown</strongs_def></entry>
  <entry><greek unicode="δένδρον"/><strongs_def>a tree</strongs_def></entry>
</entries></strongsdictionary>
""",
        encoding="utf-8",
    )
    res = build_ccl_strongs_bridge(ccl_xml=ccl_xml, strongs_xml=strongs_xml)
    pairs = {(m.coptic_lemma, m.greek_lemma) for m in res.native_pairs}
    assert ("ⲃⲱ", "δένδρον") in pairs
    assert all(m.greek_lemma != "ἄγνωστος" for m in res.native_pairs)
    assert all(not m.coptic_lemma.startswith("*") for m in res.native_pairs)


def test_committed_v3_native_gloss_hints_free_of_meta_collisions() -> None:
    """Runs on CI too: verifies committed model-v3 hints have no meta-gloss piles."""
    hints_path = (
        REPO_ROOT / "data" / "mappings" / "model-v3" / "native-gloss-hints.json"
    )
    if not hints_path.is_file():
        pytest.skip("model-v3 native-gloss-hints.json not generated yet")
    hints = json.loads(hints_path.read_text(encoding="utf-8"))
    assert len(hints) >= 200
    assert all(not k.startswith("*") for k in hints)
    counts = Counter(hints.values())
    for meta_greek in ("ἄγνωστος", "δεῖνα", "ἀδηλότης", "ἄρκτος"):
        assert counts[meta_greek] == 0, f"Meta-gloss collision on {meta_greek}"
    assert max(counts.values()) <= 4


def test_real_ccl_strongs_bridge_produces_clean_native_pairs() -> None:
    ccl_xml = DOWNLOADS / "ccl-ddglc" / "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
    strongs_xml = DOWNLOADS / "strongs-greek" / "strongsgreek.xml"
    if not (ccl_xml.is_file() and strongs_xml.is_file()):
        pytest.skip("CCL or Strong's XML not downloaded in this test environment")
    res = build_ccl_strongs_bridge(ccl_xml=ccl_xml, strongs_xml=strongs_xml)
    assert res.native_coptic_count >= 4000
    assert res.strongs_entry_count >= 5000
    assert len(res.native_pairs) >= 2000
    assert len(res.grc_eng_pairs) >= 4000
    pair_map = {(m.coptic_lemma, m.greek_lemma) for m in res.native_pairs}
    assert ("ⲃⲱ", "δένδρον") in pair_map
    assert ("ⲃⲉⲛⲓⲡⲉ", "σίδηρος") in pair_map
    counts = Counter(m.greek_lemma for m in res.native_pairs)
    assert counts["ἄγνωστος"] == 0
    assert counts["δεῖνα"] == 0
    assert counts["ἀδηλότης"] == 0
    assert max(counts.values()) <= 4


def test_first1k_rejects_non_allowlisted_xml_and_parses_patristics(
    tmp_path: Path,
) -> None:
    bad_xml = tmp_path / "tlg9999.thomas.1st1K-grc1.xml"
    bad_xml.write_text("<TEI/>", encoding="utf-8")
    with pytest.raises(ValueError, match="non-allowlisted"):
        load_first1k_koine_sentences(tmp_path)

    if (DOWNLOADS / "apostolic-fathers" / "texts").is_dir():
        sents = load_all_patristic_koine_sentences(DOWNLOADS)
        assert len(sents) >= 8000
        assert all("thomas" not in uid.lower() for uid, _ in sents)


def test_apophthegmata_tt_parser_extracts_morphemes_and_english() -> None:
    sample = """
<chapter_n chapter_n="1">
<verse_n verse_n="2">
<translation translation="He said, I desire to save my soul.">
<orig_group orig_group="ⲡⲉϫⲁϥ">
<orig orig="ⲡⲉϫⲁ">
<norm xml:id="u1" func="root" pos="VBD" lemma="ⲡⲉϫⲉ" norm="ⲡⲉϫⲁ">ⲡⲉϫⲁ</norm>
</orig>
<orig orig="ϥ">
<norm xml:id="u2" func="nsubj" head="#u1" pos="PPERS" lemma="ⲛⲧⲟϥ" norm="ϥ">ϥ</norm>
</orig>
</orig_group>
</verse_n>
"""
    units = parse_apophthegmata_tt(sample, doc_id="ap_test")
    assert len(units) == 1
    assert units[0].coptic.unit_id == "ap.ap_test.1.2"
    assert [m.norm for m in units[0].coptic.morphemes] == ["ⲡⲉϫⲁ", "ϥ"]
    assert units[0].coptic.morphemes[1].head_seq == 0
    assert units[0].english == "He said, I desire to save my soul."
