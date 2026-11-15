# SPDX-License-Identifier: Apache-2.0
"""Integrity fixes from review 5911323110. Not an accuracy iteration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zetodos_thomas.dictionary.apply import TokenOut, apply_tokens
from zetodos_thomas.dictionary.build import (
    DictionaryBuildError,
    ObservedLink,
    build_dictionary,
)
from zetodos_thomas.dictionary.integrity import (
    dump_json,
    export_inference_state,
    hash_aligner_input,
    load_inference_state,
    verify_aligner_input,
    verify_named_files,
    verify_prediction_pin,
)
from zetodos_thomas.dictionary.score import score_aligned_recovery, score_fragments
from zetodos_thomas.ingest.fragments import load_fragment_corpus
from zetodos_thomas.ingest.paths import find_repo_root
from zetodos_thomas.integrations.aligners.types import BitextSentence
from zetodos_thomas.schemas.fragment import (
    CorpusRole,
    EvidenceType,
    FragmentCorpus,
    FragmentLayerKind,
    FragmentLine,
    FragmentToken,
    FragmentTranscription,
)


def _link(coptic: str, greek: str, count: int) -> ObservedLink:
    return ObservedLink(
        coptic=coptic,
        greek=greek,
        observation_count=count,
        count_eflomal=count,
        count_fast_align=count,
        consensus_count=count,
    )


def _dictionary(links: list[ObservedLink]):
    return build_dictionary(
        links,
        unit_ids=["matthew.5.3"],
        aligners=["eflomal", "fast_align"],
        provenance={"source": "test"},
    )


def test_same_greek_word_in_the_wrong_saying_is_not_recovered() -> None:
    word = "λογος"
    decisions = {
        2: [],
        7: [
            TokenOut(
                index=0,
                coptic="ϣⲁϫⲉ",
                greek=word,
                channel="direct",
                posterior=0.8,
                observation_count=4,
                span_head=0,
            )
        ],
    }
    scored = score_aligned_recovery(decisions, {2: [word], 7: ["ἄλλο"]})
    assert scored["correctness_gate"] is False
    per_logion = scored["per_logion"]
    assert isinstance(per_logion, dict)
    assert per_logion["7"]["recovered"] == 0
    assert per_logion["2"]["recovered"] == 0
    assert scored["wrong_saying_overlaps"] == 1


def test_unclear_surviving_tokens_are_a_separate_stratum() -> None:
    clear = FragmentToken(kind=FragmentLayerKind.SURVIVING, text="λογος")
    unclear_a = FragmentToken(
        kind=FragmentLayerKind.SURVIVING,
        text="ουσιν",
        certainty="unclear",
    )
    unclear_b = FragmentToken(
        kind=FragmentLayerKind.SURVIVING,
        text="ὃν",
        certainty="unclear",
    )
    lacuna = FragmentToken(kind=FragmentLayerKind.LACUNA, text="")
    transcription = FragmentTranscription(
        id="poxy-654-dclp",
        source_id="dclp",
        manuscript_id="poxy-654",
        lines=[
            FragmentLine(
                line_id="main.33",
                n="33",
                tokens=[unclear_a, clear],
            ),
            FragmentLine(line_id="main.42", n="42", tokens=[unclear_b, lacuna]),
        ],
    )
    corpus = FragmentCorpus(
        id="fixture",
        title="fixture",
        evidence_type=EvidenceType.G,
        corpus_role=CorpusRole.EVALUATION_WITNESS,
        transcriptions=[transcription],
    )
    decisions = {
        0: [
            TokenOut(
                index=0,
                coptic="ⲁ",
                greek="ουσιν",
                channel="direct",
                posterior=0.9,
                observation_count=4,
                span_head=0,
            )
        ]
    }
    dest = Path("/tmp/thomas-pin-fixture")
    dest.mkdir(exist_ok=True)
    (dest / "dictionary.json").write_text("{}\n", encoding="utf-8")
    (dest / "thomas-reconstruction.json").write_text("{}\n", encoding="utf-8")
    from zetodos_thomas.dictionary.integrity import prediction_pin

    summary = score_fragments(
        decisions,
        corpus,
        prediction_pin=prediction_pin(dest),
        pin_dir=dest,
    )
    assert summary["vocabulary_overlap_is_correctness_gate"] is False
    assert summary["official_score_available"] is False
    clear_overlap = summary["clear_surviving_overlap"]
    sensitivity = summary["unclear_sensitivity"]
    assert isinstance(clear_overlap, dict)
    assert isinstance(sensitivity, dict)
    including = sensitivity["including_unclear"]
    assert isinstance(including, dict)
    clear_types = clear_overlap["fragment_types"]
    pooled = including["fragment_types"]
    assert clear_types == 1
    assert pooled > clear_types
    root = find_repo_root()
    real = load_fragment_corpus(
        root / "data/sources/fragments/greek-thomas-fragments.json"
    )
    found: dict[str, str] = {}
    for item in real.transcriptions:
        if item.id != "poxy-654-dclp":
            continue
        for line in item.lines:
            for token in line.tokens:
                if token.certainty == "unclear" and line.line_id in {
                    "main.33",
                    "main.42",
                }:
                    found[line.line_id] = token.text
    assert found["main.33"] == "ουσιν"
    assert found["main.42"] == "ὃν"


def test_prediction_pin_fails_when_the_frozen_file_changes(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.json"
    reconstruction = tmp_path / "thomas-reconstruction.json"
    dictionary.write_text('{"id":"parallel-dict-v8"}\n', encoding="utf-8")
    reconstruction.write_text("{}\n", encoding="utf-8")
    from zetodos_thomas.dictionary.integrity import prediction_pin

    pin = prediction_pin(tmp_path)
    dictionary.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(DictionaryBuildError):
        verify_prediction_pin(tmp_path, pin)


def test_mutated_ot_or_nt_source_fails_provenance_verification(tmp_path: Path) -> None:
    source = tmp_path / "extra.txt"
    source.write_text("verse-a\n", encoding="utf-8")
    from zetodos_thomas.dictionary.integrity import hash_file

    record = {"ot_zip_sha256": hash_file(source)}
    verify_named_files(record, {"ot_zip_sha256": source})
    source.write_text("verse-b\n", encoding="utf-8")
    with pytest.raises(DictionaryBuildError):
        verify_named_files(record, {"ot_zip_sha256": source})
    sentences = [
        BitextSentence(
            unit_id="ot.Genesis.1.1",
            source_tokens=("ⲁ",),
            target_tokens=("εν",),
            source_token_ids=("s",),
            target_token_ids=("t",),
            split_role="train",
        )
    ]
    bitext_record = {"aligner_input_sha256": hash_aligner_input(sentences)}
    verify_aligner_input(bitext_record, sentences)
    changed = [
        BitextSentence(
            unit_id="ot.Genesis.1.1",
            source_tokens=("ⲃ",),
            target_tokens=("εν",),
            source_token_ids=("s",),
            target_token_ids=("t",),
            split_role="train",
        )
    ]
    with pytest.raises(DictionaryBuildError):
        verify_aligner_input(bitext_record, changed)


def test_model_roundtrip_reproduces_v8_thomas_7(tmp_path: Path) -> None:
    root = find_repo_root()
    reconstruction = json.loads(
        (
            root / "data/reconstructions/parallel-dict-v8/thomas-reconstruction.json"
        ).read_text(encoding="utf-8")
    )
    unit = next(item for item in reconstruction["units"] if item["logion"] == 7)
    rows = unit["tokens"]
    tokens = tuple(str(row["c"]) for row in rows)
    dictionary = _dictionary([_link("ⲁⲩⲱ", "και", 4)])
    dictionary.version = "parallel-dict-v8"
    dictionary.replay_traces[tokens] = tuple(dict(row) for row in rows)
    dictionary.phrases[("ⲡ", "ⲙⲟⲩⲓ")] = (("ο", "λεων"), 2, 1.0)
    dictionary.use_phrases = True
    path = tmp_path / "model.json"
    dump_json(path, export_inference_state(dictionary))
    loaded = load_inference_state(json.loads(path.read_text(encoding="utf-8")))
    outputs = apply_tokens(list(tokens), loaded)
    assert loaded.phrases[("ⲡ", "ⲙⲟⲩⲓ")][0] == ("ο", "λεων")
    for row, output in zip(rows, outputs, strict=True):
        assert output.coptic == row["c"]
        assert output.greek == row["g"]
        assert output.channel == row["ch"]
        assert output.span_head == row["h"]
        assert output.greek_extra == tuple(row.get("x") or ())


def test_version_directories_agree_on_identity() -> None:
    root = find_repo_root() / "data" / "reconstructions"
    for name in (
        "parallel-dict-v1",
        "parallel-dict-v2",
        "parallel-dict-v3",
        "parallel-dict-v4",
        "parallel-dict-v8",
    ):
        directory = root / name
        dictionary = json.loads(
            (directory / "dictionary.json").read_text(encoding="utf-8")
        )
        reconstruction = json.loads(
            (directory / "thomas-reconstruction.json").read_text(encoding="utf-8")
        )
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        version = dictionary["id"]
        assert version == name
        assert reconstruction["dictionary_id"] == version
        assert metrics["version"] == version
        assert metrics["id"] == f"{version}-metrics"
        assert metrics["fragment_evaluation_status"] == "exploratory"
        assert metrics["vocabulary_overlap_is_correctness_gate"] is False
        assert dictionary.get("provenance", {}).get("version") == version
