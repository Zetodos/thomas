# SPDX-License-Identifier: Apache-2.0
"""JAK-43: Sahidic bound-group segmentation for Horner NT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zetodos_thomas.align.lexical_baseline import fit_lexical_baseline
from zetodos_thomas.align.segment_coptic import (
    ALIGNMENT_READY,
    SEGMENTATION_VERSION,
    SEGMENTED_ID_SCHEME_VERSION,
    peel_bound_group,
    segment_normalized_coptic,
    segmentation_stats,
    segmented_coptic_tokens,
    segmented_token_id,
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
    assert peel_bound_group("ⲁⲩⲱ") == ["ⲁⲩⲱ"]
    # Exact closed forms stay atomic (no fall-through to shorter prefixes).
    assert peel_bound_group("ⲡⲉ") == ["ⲡⲉ"]
    assert peel_bound_group("ⲛϭⲓ") == ["ⲛϭⲓ"]
    assert SEGMENTATION_VERSION == "bound-group-v1"
    assert SEGMENTED_ID_SCHEME_VERSION == "2"
    assert ALIGNMENT_READY is True


def test_peel_does_not_break_common_stems() -> None:
    """Jake review 4544895967: bare ⲛ/ⲟⲩ/ⲧⲉ/ⲡⲉ must not slice lexemes.

    Exact Unicode from the review body (ϫ in ⲡⲉϫⲉ is U+03EB as in Horner).
    """
    # Without a stem guard these become [ⲛ, ⲟⲩ, ⲧⲉ] / [ⲡ, ⲛ, ⲟⲩ, ⲧⲉ] / [ⲡⲉ, ϫⲉ].
    assert peel_bound_group("ⲛⲟⲩⲧⲉ") == ["ⲛⲟⲩⲧⲉ"]
    assert peel_bound_group("ⲡⲛⲟⲩⲧⲉ") == ["ⲡ", "ⲛⲟⲩⲧⲉ"]
    assert peel_bound_group("ⲡⲉϫⲉ") == ["ⲡⲉϫⲉ"]
    assert peel_bound_group("ⲙⲡⲛⲟⲩⲧⲉ") == ["ⲙⲡ", "ⲛⲟⲩⲧⲉ"]
    assert peel_bound_group("ⲡⲉϫⲁϥ") == ["ⲡⲉϫⲁϥ"]
    # Codepoint lock for the review examples (guards against UTF corruption).
    assert [hex(ord(c)) for c in "ⲛⲟⲩⲧⲉ"] == [
        "0x2c9b",
        "0x2c9f",
        "0x2ca9",
        "0x2ca7",
        "0x2c89",
    ]
    assert [hex(ord(c)) for c in "ⲡⲉϫⲉ"] == ["0x2ca1", "0x2c89", "0x3eb", "0x2c89"]


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


def test_segmented_tokens_use_v2_ids_and_original_offsets() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    unit = next(u for u in corpus.units if u.unit_id == "john.1.1")
    tokens = segmented_coptic_tokens(unit.sahidic)
    assert tokens
    assert tokens[0].id == segmented_token_id("copsahhorner", "john.1.1", 0)
    assert tokens[0].id.startswith("v2:")
    assert "/t/0" in tokens[0].id
    # Must not collide with whitespace v1 id for sequence 0.
    assert not tokens[0].id.startswith("v1:")
    original = unit.sahidic.original_text
    for tok in tokens:
        assert tok.char_start is not None and tok.char_end is not None
        assert original[tok.char_start : tok.char_end] == tok.surface
        assert tok.normalized
        # Surface may include combining marks stripped from normalized.
        assert len(tok.surface) >= len(tok.normalized) or tok.surface == tok.normalized


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
    assert model.left_counts["ⲛϭⲓ"] == 1


def test_blinded_thomas7_package_seals_greek() -> None:
    payload = json.loads(BLIND_PATH.read_text(encoding="utf-8"))
    assert payload["id"] == "thomas-7-blinded-eval-v1"
    assert payload["pilot_logion"] == 7
    assert payload["review_status"] == "ready-for-independent-review"
    assert payload["seal"]["greek_witness_id"] == "poxy-654-dclp"
    assert payload["seal"]["corpus_role"] == "evaluation-witness"
    blob = BLIND_PATH.read_text(encoding="utf-8")
    assert "μακάριος" in blob
    assert "poxy-654" in blob.lower()
    assert "surviving-letter-agreement" in payload["scoring_rubric"]["levels"]
    assert "thomas.7" == payload["input"]["coptic_unit"]["unit_id"]
    assert "ⲙⲁⲕⲁⲣⲓⲟⲥ" in payload["input"]["coptic_unit"]["normalized_text"]


def test_inventory_records_expansion_counts() -> None:
    inv = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inv["nonempty_positive_pairs"] == 396
    assert inv["unit_slots"] == 407
    assert inv["split_nonempty_counts"]["train"] == 304
    assert inv["split_nonempty_counts"]["holdout"] == 41
    assert inv["split_nonempty_counts"]["dev"] == 51
    assert inv["phase1_comparison"]["expansion_factor_nonempty"] >= 10


def test_eval_review_gate_blocks_unreviewed_thomas7() -> None:
    from zetodos_thomas.eval.review_gate import (
        EvalReviewGateError,
        assert_eval_package_human_checked,
    )

    with pytest.raises(EvalReviewGateError, match="human-checked"):
        assert_eval_package_human_checked(BLIND_PATH)
    payload = json.loads(BLIND_PATH.read_text(encoding="utf-8"))
    assert payload["review_status"] == "ready-for-independent-review"
    assert "blocking follow-up" in payload["notes"].lower() or (
        "follow-up" in payload.get("independent_review_follow_up", {}).get("status", "")
    )
