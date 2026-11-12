# SPDX-License-Identifier: Apache-2.0
"""JAK-37: lexical candidate layers, train-only rules, dual-aligner provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.integrations.aligners.types import (
    AlignerName,
    AlignmentLinks,
    BitextSentence,
    DirectionalAlignment,
)
from zetodos_thomas.lexicon.candidates import (
    accumulate_wordform_observations,
    build_confidence_estimates,
    build_inferred_lemmas,
    build_lexical_candidate_table,
    build_observed_wordforms,
    load_compact_alignment_artifact,
    write_lexical_candidate_table,
)
from zetodos_thomas.lexicon.lemmas import infer_greek_lemma, inference_method_payload
from zetodos_thomas.lexicon.validate import validate_lexical_candidate_table
from zetodos_thomas.schemas.lexicon import LexicalLayer, SplitMembership

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "lexicon"


def _sentence(
    unit_id: str,
    src: tuple[str, ...],
    trg: tuple[str, ...],
    *,
    role: str = "train",
) -> BitextSentence:
    return BitextSentence(
        unit_id=unit_id,
        source_tokens=src,
        target_tokens=trg,
        source_token_ids=tuple(f"v2:c/{unit_id}/t/{i}" for i in range(len(src))),
        target_token_ids=tuple(f"v1:g/{unit_id}/t/{i}" for i in range(len(trg))),
        split_role=role,
        source_parent_v1_ids=tuple(f"v1:c/{unit_id}/t/{i}" for i in range(len(src))),
    )


def _links(
    name: AlignerName,
    sentences: list[BitextSentence],
    sym_pairs: dict[str, tuple[tuple[int, int], ...]],
) -> AlignmentLinks:
    forward = [
        DirectionalAlignment(unit_id=s.unit_id, pairs=sym_pairs.get(s.unit_id, ()))
        for s in sentences
    ]
    return AlignmentLinks(
        aligner=name,
        sentences=sentences,
        forward=forward,
        reverse=list(forward),
        symmetrized=list(forward),
        notes=["fixture"],
    )


def test_layers_stay_separate_and_counts_are_evidence_backed() -> None:
    sents = [
        _sentence("john.1.1", ("ϩⲛ", "ⲧⲉϩⲟⲩⲉⲓⲧⲉ"), ("εν", "αρχη")),
        _sentence("john.1.2", ("ϩⲛ",), ("εν",)),
    ]
    efl = _links(
        AlignerName.EFLOMAL,
        sents,
        {
            "john.1.1": ((0, 0), (1, 1)),
            "john.1.2": ((0, 0),),
        },
    )
    fa = _links(
        AlignerName.FAST_ALIGN,
        sents,
        {
            "john.1.1": ((0, 0),),  # misses second pair
            "john.1.2": ((0, 0),),
        },
    )
    table = build_lexical_candidate_table(
        {"eflomal": efl, "fast_align": fa},
        provenance={"owned_by": "JAK-37"},
    )
    assert table.layers["observed_wordform"]["entry_count"] == len(
        table.observed_wordforms
    )
    assert table.layers["inferred_lemma"]["entry_count"] == len(table.inferred_lemmas)
    assert table.layers["confidence_estimate"]["entry_count"] == len(
        table.confidence_estimates
    )

    hn = next(
        r
        for r in table.observed_wordforms
        if r.coptic_wordform == "ϩⲛ" and r.greek_wordform == "εν"
    )
    assert hn.layer is LexicalLayer.OBSERVED_WORDFORM
    assert hn.split_membership is SplitMembership.TRAIN
    assert hn.per_aligner_observation_counts["eflomal"] == 2
    assert hn.per_aligner_observation_counts["fast_align"] == 2
    # Headline count is max(per_aligner), never a fabricated sum.
    assert hn.observation_count == 2
    assert set(hn.contributing_aligners) == {"eflomal", "fast_align"}
    assert hn.source_unit_ids == ["john.1.1", "john.1.2"]

    te = next(r for r in table.observed_wordforms if r.coptic_wordform == "ⲧⲉϩⲟⲩⲉⲓⲧⲉ")
    assert te.contributing_aligners == ["eflomal"]
    assert te.observation_count == 1

    lemma = next(
        r
        for r in table.inferred_lemmas
        if r.coptic_lemma == "ϩⲛ" and r.greek_lemma == "εν"
    )
    assert lemma.layer is LexicalLayer.INFERRED_LEMMA
    assert lemma.inference_method["greek"] == inference_method_payload()["greek"]
    assert lemma.observation_count >= 1

    conf = next(
        c
        for c in table.confidence_estimates
        if c.key.get("coptic_wordform") == "ϩⲛ" and c.key.get("greek_wordform") == "εν"
    )
    assert conf.layer is LexicalLayer.CONFIDENCE_ESTIMATE
    assert conf.applies_to_layer is LexicalLayer.OBSERVED_WORDFORM
    assert conf.confidence is not None
    assert conf.components.aligner_agreement == 1.0

    report = validate_lexical_candidate_table(table)
    assert report.ok, report.issues


def test_rejects_dev_or_thomas_observations() -> None:
    bad = [_sentence("john.3.1", ("ⲁ",), ("α",), role="dev")]
    links = _links(AlignerName.EFLOMAL, bad, {"john.3.1": ((0, 0),)})
    with pytest.raises(ValueError, match="train-only"):
        build_lexical_candidate_table({"eflomal": links})

    thomas = [_sentence("thomas.7", ("ⲁ",), ("α",))]
    links_t = _links(AlignerName.EFLOMAL, thomas, {"thomas.7": ((0, 0),)})
    with pytest.raises(ValueError, match="forbidden"):
        accumulate_wordform_observations({"eflomal": links_t})


def test_greek_lemma_strips_diacritics_without_claiming_morph() -> None:
    assert infer_greek_lemma("ἀρχῇ") == "αρχη"
    assert infer_greek_lemma("λόγος") == "λογος"
    # Accented vs plain roll into one inferred lemma.
    sents = [_sentence("john.1.1", ("ϣⲁϫⲉ", "ϣⲁϫⲉ"), ("λόγος", "λογος"))]
    # Two tokens same verse — use two units for clarity.
    sents = [
        _sentence("john.1.1", ("ϣⲁϫⲉ",), ("λόγος",)),
        _sentence("john.1.14", ("ϣⲁϫⲉ",), ("λογος",)),
    ]
    links = _links(
        AlignerName.EFLOMAL,
        sents,
        {"john.1.1": ((0, 0),), "john.1.14": ((0, 0),)},
    )
    wordforms = build_observed_wordforms(
        accumulate_wordform_observations({"eflomal": links})
    )
    lemmas = build_inferred_lemmas(wordforms)
    assert len(wordforms) == 2
    assert len(lemmas) == 1
    assert lemmas[0].greek_lemma == "λογος"
    assert len(lemmas[0].underlying_wordform_pairs) == 2


def test_confidence_null_when_no_components() -> None:
    # Manually craft empty-ish path via zero-count skip: confidence built only
    # from positive observed rows; empty list → empty confidence.
    assert build_confidence_estimates([]) == []


def test_compact_artifact_roundtrip(tmp_path: Path) -> None:
    sents = [_sentence("john.1.1", ("ϩⲛ", "ⲗⲟⲅⲟⲥ"), ("εν", "λογος"))]
    links = _links(
        AlignerName.FAST_ALIGN,
        sents,
        {"john.1.1": ((0, 0), (1, 1))},
    )
    from zetodos_thomas.align.statistical import export_alignment_artifact

    path = tmp_path / "fast_align-train-links-v1.json"
    export_alignment_artifact(
        links,
        path=path,
        provenance={"owned_by": "JAK-37", "aligner": "fast_align"},
        compact=True,
    )
    loaded = load_compact_alignment_artifact(path, sents)
    assert loaded.aligner is AlignerName.FAST_ALIGN
    assert loaded.symmetrized[0].pairs == ((0, 0), (1, 1))
    table = build_lexical_candidate_table({"fast_align": loaded})
    out = tmp_path / "table.json"
    write_lexical_candidate_table(table, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["owned_by"] == "JAK-37"
    assert payload["summary"]["adopted_aligner"] is None


def test_validate_cli_on_fixture(tmp_path: Path) -> None:
    sents = [
        _sentence("matt.5.3", ("ⲛⲁⲓⲁⲧⲟⲩ",), ("μακαριοι",)),
        _sentence("matt.5.4", ("ⲛⲁⲓⲁⲧⲟⲩ",), ("μακαριοι",)),
    ]
    efl = _links(
        AlignerName.EFLOMAL,
        sents,
        {"matt.5.3": ((0, 0),), "matt.5.4": ((0, 0),)},
    )
    fa = _links(
        AlignerName.FAST_ALIGN,
        sents,
        {"matt.5.3": ((0, 0),), "matt.5.4": ((0, 0),)},
    )
    table = build_lexical_candidate_table(
        {"eflomal": efl, "fast_align": fa},
        provenance={"test": True},
    )
    table.summary["phase2_active_aligners"] = ["eflomal", "fast_align"]
    table.summary["adopted_aligner"] = None
    path = tmp_path / "coptic-greek-lexical-candidates-v1.json"
    write_lexical_candidate_table(table, path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "lexicon",
            "validate-candidates",
            "--repo-root",
            str(REPO_ROOT),
            "--table",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_committed_artifact_roundtrip_and_validate() -> None:
    path = (
        REPO_ROOT
        / "data"
        / "sources"
        / "lexicon"
        / "coptic-greek-lexical-candidates-v1.json"
    )
    if not path.is_file():
        pytest.skip("committed lexical candidate table not present")
    from zetodos_thomas.lexicon.candidates import load_lexical_candidate_table

    table = load_lexical_candidate_table(path)
    assert table.owned_by == "JAK-37"
    assert table.observed_wordforms
    assert table.inferred_lemmas
    assert table.confidence_estimates
    assert table.summary.get("adopted_aligner") is None
    assert set(table.summary.get("phase2_active_aligners") or []) == {
        "eflomal",
        "fast_align",
    }
    # Layers remain distinguishable after columnar expand.
    assert all(
        r.layer is LexicalLayer.OBSERVED_WORDFORM for r in table.observed_wordforms
    )
    assert all(r.layer is LexicalLayer.INFERRED_LEMMA for r in table.inferred_lemmas)
    assert all(
        r.layer is LexicalLayer.CONFIDENCE_ESTIMATE for r in table.confidence_estimates
    )
    report = validate_lexical_candidate_table(table)
    assert report.ok, report.issues
    # No Thomas / P.Oxy leakage in source units.
    for row in table.observed_wordforms:
        assert all(
            not (uid.lower().startswith("thomas.") or "poxy" in uid.lower())
            for uid in row.source_unit_ids
        )


def test_validation_flags_leakage() -> None:
    sents = [_sentence("john.1.1", ("ⲁ",), ("α",))]
    links = _links(AlignerName.EFLOMAL, sents, {"john.1.1": ((0, 0),)})
    table = build_lexical_candidate_table({"eflomal": links})
    # Inject leakage after build.
    table.observed_wordforms[0].source_unit_ids.append("thomas.7")
    report = validate_lexical_candidate_table(table)
    assert not report.ok
    assert any(i.code == "holdout_leakage" for i in report.issues)
