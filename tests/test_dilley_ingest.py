# SPDX-License-Identifier: Apache-2.0
"""Tests for token-complete Dilley/SCRIPTORIUM Thomas .tt SGML ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from zetodos_thomas.ingest.dilley_thomas import (
    INGEST_LEVEL,
    expected_unit_ids,
    load_thomas_corpus,
    parse_tt,
    validate_thomas_unit_sequence,
    write_thomas_corpus,
)
from zetodos_thomas.schemas.linguistic import CopticCorpus
from zetodos_thomas.schemas.source import UnitKind

REPO_ROOT = Path(__file__).resolve().parents[1]
DILLEY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "dilley" / "thomas_gospel_snippet.tt"
)
THOMAS_PATH = REPO_ROOT / "data" / "sources" / "thomas" / "dilley-thomas-v6.2.0.json"


def test_parse_dilley_fixture_preserves_full_layers() -> None:
    corpus = parse_tt(DILLEY_FIXTURE, expect_complete=False)
    assert corpus.annotation_level == INGEST_LEVEL
    assert [u.chapter for u in corpus.units] == [0, 1, 2]

    prologue = corpus.units[0]
    assert prologue.unit_id == "thomas.0"
    assert prologue.kind is UnitKind.PROLOGUE
    assert prologue.original_text.startswith("ⲛⲁⲉⲓ")
    assert prologue.normalized_text.startswith("ⲛⲁⲓ")
    assert len(prologue.morphemes) >= 15

    first = prologue.morphemes[0]
    assert first.unit_id == "thomas.0"
    assert first.seq == 0
    assert first.id == "v1:dilley-thomas/thomas.0/t/0"
    assert first.orig == "ⲛⲁⲉⲓ"
    assert first.norm == "ⲛⲁⲓ"
    assert first.lemma == "ⲡⲁⲓ"
    assert first.pos == "PDEM"
    assert first.dep_rel == "root"

    cop = prologue.morphemes[1]
    assert cop.pos == "COP"
    assert cop.head_seq == 0
    assert cop.dep_rel == "cop"

    # Multiword & Greek loanwords appear in Thomas snippet.
    assert any(m.lang == "Greek" for u in corpus.units for m in u.morphemes)
    assert any(m.multiword for u in corpus.units for m in u.morphemes)


def test_committed_dilley_corpus_roundtrip_under_1024kib(tmp_path: Path) -> None:
    assert THOMAS_PATH.stat().st_size < 1024 * 1024
    corpus = load_thomas_corpus(THOMAS_PATH)
    assert len(corpus.units) == 115
    assert [u.unit_id for u in corpus.units] == expected_unit_ids()

    total_morphemes = sum(len(u.morphemes) for u in corpus.units)
    assert total_morphemes > 6000

    # Check hydration of id, unit_id, seq, and orig on compact reload.
    for unit in corpus.units[:5]:
        for idx, morph in enumerate(unit.morphemes):
            assert morph.seq == idx
            assert morph.unit_id == unit.unit_id
            assert morph.id == f"v1:dilley-thomas/{unit.unit_id}/t/{idx}"
            assert morph.orig

    out = tmp_path / "thomas-roundtrip.json"
    write_thomas_corpus(corpus, out)
    assert out.stat().st_size < 1024 * 1024
    reloaded = load_thomas_corpus(out)
    assert len(reloaded.units) == 115
    assert reloaded.units[7].morphemes == corpus.units[7].morphemes


def test_thomas_validate_rejects_wrong_sequence(tmp_path: Path) -> None:
    good = load_thomas_corpus(THOMAS_PATH)
    broken_units = list(good.units[1:])
    with pytest.raises(ValueError, match=r"thomas\.0"):
        validate_thomas_unit_sequence(broken_units)

    dup = list(good.units)
    dup[50] = good.units[114]
    with pytest.raises(ValueError, match=r"thomas\.0 through thomas\.114"):
        validate_thomas_unit_sequence(dup)

    tmp = tmp_path / "bad_sequence.json"
    bad = CopticCorpus(
        id="bad",
        title="bad",
        source_id=good.source_id,
        units=broken_units,
    )
    write_thomas_corpus(bad, tmp)
    with pytest.raises(ValueError, match=r"thomas\.0"):
        load_thomas_corpus(tmp)
