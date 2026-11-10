# SPDX-License-Identifier: Apache-2.0
"""JAK-43: Sahidic bound-group segmentation for Horner NT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zetodos_thomas.align.lexical_baseline import fit_lexical_baseline
from zetodos_thomas.align.segment_coptic import (
    SEGMENTATION_VERSION,
    peel_bound_group,
    segment_normalized_coptic,
    segmentation_stats,
    segmented_coptic_tokens,
)
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.schemas.source import TextUnit

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "horner-vaticanus-nt-pilot.json"
BLIND_PATH = REPO_ROOT / "data" / "evidence" / "eval" / "thomas-7-blinded-v1.json"
INVENTORY_PATH = (
    REPO_ROOT / "data" / "training" / "horner-vaticanus-nt-v3-inventory.json"
)


def test_peel_common_horner_bound_groups() -> None:
    assert peel_bound_group("ⲛϭⲓⲡϣⲁϫⲉ") == ["ⲛϭⲓ", "ⲡ", "ϣⲁϫⲉ"]
    assert peel_bound_group("ϩⲛⲧⲉϩⲟⲩⲉⲓⲧⲉ")[0] in {"ϩⲛⲧⲉ", "ϩⲛ"}
    assert peel_bound_group("ⲁⲩⲱ") == ["ⲁⲩⲱ"]  # no empty remainder peel
    assert SEGMENTATION_VERSION == "bound-group-v1"


def test_segment_normalized_splits_john_1_1() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    unit = next(u for u in corpus.units if u.unit_id == "john.1.1")
    ws = unit.sahidic.normalized_text.split()
    seg = segment_normalized_coptic(unit.sahidic.normalized_text)
    assert len(seg) > len(ws)
    assert "ⲛϭⲓ" in seg
    assert "ϣⲁϫⲉ" in seg
    stats = segmentation_stats(unit.sahidic)
    assert stats["extra_segments"] > 0


def test_segmented_tokens_mint_contiguous_offsets() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    unit = next(u for u in corpus.units if u.unit_id == "john.1.1")
    tokens = segmented_coptic_tokens(unit.sahidic)
    assert tokens
    assert tokens[0].id.startswith("v1:") and "/t/" in tokens[0].id
    text = unit.sahidic.normalized_text
    for tok in tokens:
        assert text[tok.char_start : tok.char_end] == tok.normalized == tok.surface


def test_segmented_tokens_reject_dilley() -> None:
    unit = TextUnit(
        source_id="dilley-thomas",
        unit_id="thomas.7",
        book="thomas",
        chapter=7,
        verse=0,
        original_text="ⲡⲉϫⲉ",
        normalized_text="ⲡⲉϫⲉ",
    )
    with pytest.raises(ValueError, match="Dilley"):
        segmented_coptic_tokens(unit)


def test_lexical_baseline_accepts_bound_group_mode() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    unit = next(u for u in corpus.units if u.unit_id == "john.1.1")
    model = fit_lexical_baseline([unit], coptic_segmentation="bound-group-v1")
    assert model.train_unit_ids == ["john.1.1"]
    # Bound-group peel should observe ⲛϭⲓ as its own type.
    assert model.left_counts["ⲛϭⲓ"] == 1


def test_blinded_thomas7_package_seals_greek() -> None:
    payload = json.loads(BLIND_PATH.read_text(encoding="utf-8"))
    assert payload["id"] == "thomas-7-blinded-eval-v1"
    assert payload["pilot_logion"] == 7
    assert payload["review_status"] == "ready-for-independent-review"
    assert payload["seal"]["greek_witness_id"] == "poxy-654-dclp"
    assert payload["seal"]["corpus_role"] == "evaluation-witness"
    blob = BLIND_PATH.read_text(encoding="utf-8")
    # No inline P.Oxy. Greek diplomatic dump (seal).
    assert "μακάριος" in blob  # loanword clue lemma hint OK
    assert "poxy-654" in blob.lower()
    assert "surviving-letter-agreement" in payload["scoring_rubric"]["levels"]
    assert "thomas.7" == payload["input"]["coptic_unit"]["unit_id"]
    assert "ⲙⲁⲕⲁⲣⲓⲟⲥ" in payload["input"]["coptic_unit"]["normalized_text"]


def test_inventory_records_expansion_counts() -> None:
    inv = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inv["nonempty_positive_pairs"] == 396
    assert inv["unit_slots"] == 407
    assert inv["split_nonempty_counts"]["train"] == 323
    assert inv["split_nonempty_counts"]["holdout"] == 22
    assert inv["phase1_comparison"]["expansion_factor_nonempty"] >= 10
