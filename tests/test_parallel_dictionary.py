# SPDX-License-Identifier: Apache-2.0
"""Parallel dictionary: train alignments only, fragment Greek is a score."""

from __future__ import annotations

import inspect
from pathlib import Path

from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.dictionary import apply as apply_mod
from zetodos_thomas.dictionary import build as build_mod
from zetodos_thomas.dictionary.apply import TokenOut
from zetodos_thomas.dictionary.build import (
    ObservedLink,
    build_dictionary,
)
from zetodos_thomas.dictionary.orthography import (
    fold_greek,
    transliteration_similar,
)
from zetodos_thomas.dictionary.run import (
    links_from_candidate_table,
    run_parallel_dictionary,
)
from zetodos_thomas.dictionary.score import score_logia
from zetodos_thomas.ingest.paths import find_repo_root


def _link(
    coptic: str,
    greek: str,
    count: int,
    *,
    eflomal: int | None = None,
    fast_align: int | None = None,
) -> ObservedLink:
    return ObservedLink(
        coptic=coptic,
        greek=greek,
        observation_count=count,
        count_eflomal=count if eflomal is None else eflomal,
        count_fast_align=count if fast_align is None else fast_align,
    )


def _dictionary(
    links: list[ObservedLink], *, snap: bool = False
) -> build_mod.ParallelDictionary:
    dictionary = build_dictionary(
        links,
        unit_ids=["matthew.5.3", "john.1.1"],
        aligners=["eflomal", "fast_align"],
        provenance={"source": "test"},
    )
    dictionary.snap_enabled = snap
    return dictionary


def test_transliteration_is_script_similarity() -> None:
    assert transliteration_similar("ⲙⲁⲕⲁⲣⲓⲟⲥ", "μακαριοι")
    assert not transliteration_similar("ⲣⲱⲙⲉ", "ανθρωπος")


def test_refuses_thomas_and_fragment_units() -> None:
    links = [_link("ⲁⲩⲱ", "και", 4)]
    for unit_id in ("thomas.7", "poxy-654-dclp"):
        try:
            build_dictionary(
                links,
                unit_ids=[unit_id],
                aligners=["eflomal", "fast_align"],
                provenance={},
            )
        except build_mod.DictionaryBuildError:
            continue
        raise AssertionError(unit_id)


def test_single_aligner_link_does_not_emit() -> None:
    dictionary = _dictionary([_link("ⲁⲩⲱ", "και", 5, eflomal=5, fast_align=0)])
    assert dictionary.statistical_emission("ⲁⲩⲱ") is None


def test_inflection_cluster_emits_the_attested_mode() -> None:
    dictionary = _dictionary(
        [
            _link("ⲣⲱⲙⲉ", "ανθρωπος", 6),
            _link("ⲣⲱⲙⲉ", "ανθρωποις", 5),
            _link("ⲣⲱⲙⲉ", "ανθρωπου", 4),
            _link("ⲣⲱⲙⲉ", "ανδρι", 3),
        ]
    )
    emission = dictionary.statistical_emission("ⲣⲱⲙⲉ")
    assert emission is not None
    assert emission.greek == fold_greek("ανθρωπος")
    assert emission.posterior > 0.5


def test_short_prefix_does_not_absorb_a_longer_greek_word() -> None:
    dictionary = _dictionary(
        [
            _link("ⲡⲁⲣⲁ", "παρα", 3),
            _link("ⲡⲁⲣⲁ", "παραβολη", 3),
        ]
    )
    assert dictionary.statistical_emission("ⲡⲁⲣⲁ") is None


def test_disagreement_abstains() -> None:
    dictionary = _dictionary(
        [
            _link("ⲉⲧⲉ", "ουκ", 4),
            _link("ⲉⲧⲉ", "εαν", 4),
        ]
    )
    assert dictionary.statistical_emission("ⲉⲧⲉ") is None


def test_stem_keeps_loan_and_drops_unrelated_greek() -> None:
    dictionary = _dictionary(
        [
            _link("ⲉϥⲙⲁⲑⲏⲧⲏⲥ", "μαθητας", 3),
            _link("ⲉϥⲙⲁⲑⲏⲧⲏⲥ", "αυτου", 3),
        ]
    )
    assert dictionary.statistical_emission("ⲉϥⲙⲁⲑⲏⲧⲏⲥ") is None
    emission = dictionary.statistical_emission("ⲙⲁⲑⲏⲧⲏⲥ")
    assert emission is not None
    assert emission.channel == "stem"
    assert emission.greek.startswith("μαθητ")


def test_snap_requires_an_attested_train_type() -> None:
    dictionary = _dictionary(
        [_link("ⲛⲁⲓⲁⲧⲟⲩ", "μακαριοι", 4)],
        snap=True,
    )
    emission = dictionary.snap_emission("ⲙⲁⲕⲁⲣⲓⲟⲥ")
    assert emission is not None
    assert emission.channel == "orthographic_snap"
    assert emission.greek.startswith("μακαρ")
    blocked = _dictionary(
        [_link("ⲣⲱⲙⲉ", "ανθρωπος", 6), _link("ⲛⲁⲓⲁⲧⲟⲩ", "μακαριοι", 4)],
        snap=True,
    )
    assert blocked.snap_emission("ⲣⲱⲙⲉ") is None


def test_abstention_does_not_count_as_recovery() -> None:
    token = TokenOut(
        index=0,
        coptic="ⲙⲁⲕⲁⲣⲓⲟⲥ",
        greek=None,
        channel="abstain_no_evidence",
        posterior=None,
        observation_count=None,
        span_head=0,
    )
    scored = score_logia({7: [token]}, ["μακαριοι"], {7})
    assert scored["loanword_tokens"] == 1
    assert scored["loanword_stem_matches"] == 0
    assert scored["content_types_recovered"] == 0


def test_stem_match_recovers_fragment_type() -> None:
    token = TokenOut(
        index=0,
        coptic="ⲙⲁⲕⲁⲣⲓⲟⲥ",
        greek="μακαριοι",
        channel="orthographic_snap",
        posterior=0.55,
        observation_count=4,
        span_head=0,
    )
    scored = score_logia({7: [token]}, ["μακαριοι", "λεγει"], {7})
    assert scored["loanword_stem_matches"] == 1
    assert scored["loanword_exact_matches"] == 1
    assert scored["content_types_recovered"] == 1
    assert scored["fragment_types_recovered"] == 1


def test_training_modules_do_not_import_the_scorer() -> None:
    for module in (build_mod, apply_mod):
        source = inspect.getsource(module)
        assert "dictionary.score" not in source
        assert "load_fragment_corpus" not in source
        assert "mappings.loanwords" not in source
        assert "verified_loanword_table(" not in source


def test_dictionary_help() -> None:
    result = CliRunner().invoke(app, ["dictionary", "--help"])
    assert result.exit_code == 0
    assert "fragment" in result.stdout.lower() or "Thomas" in result.stdout


def test_real_train_table_agrees_on_kai_and_abstains_on_conflict() -> None:
    root = find_repo_root()
    path = root / "data/sources/lexicon/coptic-greek-lexical-candidates-v1.json"
    links, unit_ids, aligners, provenance = links_from_candidate_table(path)
    dictionary = build_dictionary(
        links,
        unit_ids=unit_ids,
        aligners=aligners,
        provenance=provenance,
    )
    kai = dictionary.statistical_emission("ⲁⲩⲱ")
    assert kai is not None
    assert kai.greek == "και"
    assert set(aligners) == {"eflomal", "fast_align"}
    # Spread-out alignments must not become a confident gloss.
    assert dictionary.statistical_emission("ⲉⲧⲉ") is None


def test_loan_spelling_uses_an_attested_transliteration() -> None:
    dictionary = _dictionary(
        [
            _link("ⲡⲟⲗⲓⲥ", "πολιν", 10),
            _link("ⲡⲟⲗⲓⲥ", "πολισ", 4),
            _link("ⲟⲩ", "εισ", 4),
        ]
    )
    dictionary.use_context = True
    dictionary.use_loan_spelling = True
    outputs = apply_mod.apply_tokens(["ⲟⲩ", "ⲡⲟⲗⲓⲥ"], dictionary)
    assert outputs[1].greek == "πολισ"
    assert outputs[1].channel == "loan_spelling"
    dictionary.context_counts = {"ⲡⲟⲗⲓⲥ": {"πολιν": {"ⲟⲩ": 5}}}
    overridden = apply_mod.apply_tokens(["ⲟⲩ", "ⲡⲟⲗⲓⲥ"], dictionary)
    assert overridden[1].greek == "πολιν"
    assert overridden[1].channel == "direct"


def test_phrase_does_not_replace_direct_greek() -> None:
    dictionary = _dictionary(
        [
            _link("ⲡⲟⲗⲓⲥ", "πολισ", 6),
            _link("ⲟⲩⲡⲟⲗⲓⲥ", "πολισ", 4),
        ]
    )
    dictionary.use_phrases = True
    dictionary.phrases = {("ⲟⲩ", "ⲡⲟⲗⲓⲥ"): (("πολει", "μεγαλη"), 4, 0.95)}
    outputs = apply_mod.apply_tokens(["ⲟⲩ", "ⲡⲟⲗⲓⲥ"], dictionary)
    head = outputs[0]
    assert head.greek == "πολισ"
    assert head.channel != "phrase"
    assert "μεγαλη" in head.greek_extra


def test_full_run_scores_without_training_on_fragments(tmp_path: Path) -> None:
    root = find_repo_root()
    metrics = run_parallel_dictionary(root, output_dir=tmp_path, write=True)
    assert metrics["uses_thomas_greek_as_training"] is False
    assert metrics["uses_handwritten_glosses"] is False
    assert metrics["frozen_before_fragment_score"] is True
    assert metrics["loanword_tokens"] >= 1
    for key in (
        "loanword_stem_match_rate",
        "content_type_recall",
        "content_emission_precision",
    ):
        value = metrics[key]
        assert value is None or 0.0 <= float(value) <= 1.0
    assert (tmp_path / "dictionary.json").is_file()
    assert (tmp_path / "thomas-reconstruction.json").is_file()
    assert (tmp_path / "metrics.json").is_file()
    # Applying the dictionary still abstains on a token with no train support
    # when snap is off. The run records whichever NT-dev decision it made.
    assert isinstance(metrics["snap_enabled"], bool)
    assert isinstance(metrics["approach_sufficient"], bool)
