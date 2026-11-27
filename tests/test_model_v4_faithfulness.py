# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model-v4 faithfulness, constrained beam, coverage & leak filters."""

from __future__ import annotations

from pathlib import Path

import torch

from zetodos_thomas.ingest.holdouts import greek_token_ngrams, shares_greek_token_ngram
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon, LoanwordEntry
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.neural.constrained_beam import (
    ConstraintTrie,
    assign_bound_group_coptic_seqs,
    bipartite_lexicon_coverage,
    build_chunk_lexical_constraints,
    constrained_ensemble_beam_search,
    find_unmatched_content_indices,
    has_local_loop_repeat,
    is_bare_coptic_copy_word,
    is_grammaticalized_prep_noun,
    wu_attention_coverage_penalty,
)
from zetodos_thomas.neural.corpus import (
    SPECIAL_USER_SYMBOLS_V3,
    ParallelExample,
    build_multiverse_windows,
    select_patristic_training_sentences,
    serialize_coptic_morphemes,
    split_coptic_unit_for_neural,
)
from zetodos_thomas.neural.tokenizer import BilingualTokenizer
from zetodos_thomas.neural.trainer import _compute_cov_and_align_loss
from zetodos_thomas.neural.transformer import KoineSeq2SeqTransformer, Seq2SeqConfig
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit
from zetodos_thomas.schemas.source import UnitKind


def _m(
    seq: int,
    norm: str,
    lemma: str,
    pos: str,
    *,
    group_seq: int = 0,
    lang: str | None = None,
) -> CopticMorpheme:
    return CopticMorpheme(
        seq=seq,
        group_seq=group_seq,
        norm=norm,
        lemma=lemma,
        pos=pos,
        lang=lang,
    )


def test_all_inflected_forms_covers_noun_and_verb_paradigms() -> None:
    inf = GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={"μοναχός": "M", "γνῶσις": "F"},
        surface_analyses={},
    )
    mon_forms = inf.all_inflected_forms("μοναχός", pos="N-")
    assert "μοναχός" in mon_forms
    assert "μοναχοί" in mon_forms
    assert "μοναχόν" in mon_forms

    gn_forms = inf.all_inflected_forms("γνῶσις", pos="N-")
    assert "γνῶσις" in gn_forms
    assert "γνώσεως" in gn_forms or "γνῶσιν" in gn_forms

    ter_forms = inf.all_inflected_forms("τηρέω", pos="V-")
    assert "τηρέω" in ter_forms
    assert any(f.startswith("τηρ") for f in ter_forms)


def test_bipartite_lexicon_coverage_is_multiset_and_synonym_aware() -> None:
    # 3 occurrences of ⲣⲱⲙⲉ + 1 occurrence of ⲙⲟⲩⲓ
    cands = [
        (0, {"ανθρωποσ", "ανηρ"}),
        (1, {"λεων"}),
        (2, {"ανθρωποσ", "ανηρ"}),
        (3, {"ανθρωποσ", "ανηρ"}),
    ]
    # Dropping the 3rd ἄνθρωπος must score 3/4 = 0.75 (NOT 1.0 like a set!)
    assert bipartite_lexicon_coverage(cands, ["λεων", "ανθρωποσ", "ανθρωποσ"]) == 0.75
    # Synonym ἀνήρ + 2x ἄνθρωπος + λέων covers all 4 = 1.0
    assert (
        bipartite_lexicon_coverage(cands, ["λεων", "ανθρωποσ", "ανηρ", "ανθρωποσ"])
        == 1.0
    )


def test_wu_attention_coverage_penalty_penalizes_ignored_morpheme() -> None:
    tok_to_seq = [None, 0, 1, 2, None]
    # All 3 content morphemes well attended
    good_attn = [
        [0.05, 0.80, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.80, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.80, 0.05],
    ]
    # Morpheme 2 completely ignored (dropped)
    bad_attn = [
        [0.05, 0.85, 0.05, 0.0, 0.05],
        [0.05, 0.05, 0.85, 0.0, 0.05],
        [0.05, 0.45, 0.45, 0.0, 0.05],
    ]
    p_good = wu_attention_coverage_penalty(good_attn, tok_to_seq, [0, 1, 2], tau=0.5)
    p_bad = wu_attention_coverage_penalty(bad_attn, tok_to_seq, [0, 1, 2], tau=0.5)
    assert p_good == 0.0
    assert p_bad < -1.5


def test_assign_bound_group_coptic_seqs_attaches_function_morphs_only() -> None:
    ms = [
        _m(0, "ⲡ", "ⲡ", "ART", group_seq=0),
        _m(1, "ⲙⲟⲩⲓ", "ⲙⲟⲩⲓ", "N", group_seq=0),
        _m(2, "ⲛⲁ", "ⲛⲁ", "FUT", group_seq=1),
        _m(3, "ⲟⲩⲟⲙ", "ⲟⲩⲱⲙ", "V", group_seq=1),
        _m(4, "ϥ", "ⲛⲧⲟϥ", "PPERO", group_seq=1),
        _m(5, "ⲣⲱⲙⲉ", "ⲣⲱⲙⲉ", "N", group_seq=2),  # unaligned content word!
    ]
    spans = assign_bound_group_coptic_seqs([1, 3], ms)
    assert spans == [[0, 1], [2, 3, 4]]
    # Seq 5 (dropped content noun ⲣⲱⲙⲉ) remains visibly unaligned!
    assert all(5 not in sp for sp in spans)


def test_canonicalize_preps_and_vstat_gloss_lookup() -> None:
    ms = [
        _m(0, "ϣⲱⲡⲉ", "ϣⲱⲡⲉ", "V"),
        _m(1, "ⲣ", "ⲛ", "PREP"),
        _m(2, "ⲣⲱⲙⲉ", "ⲣⲱⲙⲉ", "N"),
        _m(3, "ⲃⲏⲧ", "ⲃⲱⲧⲉ", "VSTAT"),
    ]
    hints = {"ⲃⲱⲧⲉ|VS": "βδελυκτός", "ⲃⲱⲧⲉ|V": "ἐξολεθρεύω"}
    s_canon, _ = serialize_coptic_morphemes(
        ms,
        greekify=True,
        native_gloss_hints=hints,
        canonicalize_preps=True,
    )
    assert "ν ρωμε" in s_canon
    assert "ρ ρωμε" not in s_canon
    assert "<GLOSS> βδελυκτός" in s_canon
    assert "ἐξολεθρεύω" not in s_canon


def test_balanced_split_avoids_tiny_orphan_tail() -> None:
    ms = [_m(i, "ⲁⲩⲱ" if i == 25 else "ⲣⲱⲙⲉ", "ⲣⲱⲙⲉ", "N") for i in range(44)]
    unit = CopticUnit(
        unit_id="test.1",
        source_id="test",
        book="test",
        chapter=1,
        verse=1,
        kind=UnitKind.VERSE,
        original_text="",
        normalized_text="",
        morphemes=ms,
    )
    # With max_morphemes=72, 44 morphemes stays intact in 1 chunk
    assert len(split_coptic_unit_for_neural(unit, max_morphemes=72)) == 1
    # With max_morphemes=36 and min_tail_morphemes=12, 44 avoids a <12 tail
    chunks = split_coptic_unit_for_neural(unit, max_morphemes=36, min_tail_morphemes=12)
    assert len(chunks) == 2
    assert min(len(c) for c in chunks) >= 12


def test_constrained_beam_search_guarantees_mandatory_surface(tmp_path: Path) -> None:
    texts = [
        "<DOM_NT> π ρωμε <LOAN> μοναχός",
        "λέγει Ἰησοῦς μακάριοι οἱ μοναχοί καί οἱ ἐκλεκτοί",
        "εἶπεν ὁ Ἰησοῦς μακάριοι εἰσίν οἱ πτωχοί",
    ] * 12
    tok = BilingualTokenizer.train(
        texts=texts,
        out_prefix=tmp_path / "tok",
        vocab_size=384,
        user_defined_symbols=SPECIAL_USER_SYMBOLS_V3,
    )
    cfg = Seq2SeqConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_heads=2,
        d_ff=64,
        num_enc_layers=1,
        num_dec_layers=1,
        dropout=0.0,
        attn_dropout=0.0,
    )
    torch.manual_seed(0)
    model = KoineSeq2SeqTransformer(cfg).eval()
    trie = ConstraintTrie.from_surfaces(
        label="μοναχός",
        lemma="μοναχός",
        surfaces=["μοναχός", "μοναχοί"],
        tokenizer=tok,
        source_seq=1,
        rel_pos=0.4,
    )
    assert trie is not None
    src_ids = tok.encode("<DOM_NT> π ρωμε <LOAN> μοναχός")
    hyps = constrained_ensemble_beam_search(
        forward_models=[model],
        tokenizer=tok,
        src_ids=src_ids,
        constraints=[trie],
        beam_size=4,
        max_new_tokens=16,
        min_new_tokens=3,
        len_alpha=1.0,
        device=torch.device("cpu"),
    )
    assert hyps
    top_decoded = tok.decode(hyps[0][0])
    assert "μοναχός" in top_decoded or "μοναχοί" in top_decoded


def test_patristic_ngram_filter_and_multiverse_windows() -> None:
    holdout = ["μακάριοι οἱ πτωχοί τῷ πνεύματι ὅτι αὐτῶν ἐστίν ἡ βασιλεία"]
    blocked = greek_token_ngrams(holdout[0].lower())
    assert shares_greek_token_ngram("ὅτι αὐτῶν ἐστίν ἡ βασιλεία τοῦ θεοῦ", blocked)
    kept = select_patristic_training_sentences(
        [
            ("p1", "μακάριοι οἱ πτωχοί ὅτι αὐτῶν ἐστίν ἡ βασιλεία τοῦ θεοῦ"),
            ("p2", "ὁ λόγος τοῦ θεοῦ μένει εἰς τόν αἰῶνα ἀμήν"),
        ],
        blocked_nt_verses=holdout,
        seen_greek=set(),
    )
    assert len(kept) == 1
    assert "αἰῶνα" in kept[0]

    # Multi-verse windows only merge contiguous verses (never across a gap!)
    def _ex(uid: str, n: int) -> ParallelExample:
        return ParallelExample(
            unit_id=uid,
            domain_tag="<DOM_NT>",
            coptic_morphemes=tuple(_m(i, "ⲣⲱⲙⲉ", "ⲣⲱⲙⲉ", "N") for i in range(n)),
            greek_target="ἄνθρωπος",
        )

    wins = build_multiverse_windows(
        [_ex("john.1.1", 22), _ex("john.1.2", 22), _ex("john.1.5", 22)],
        min_morphemes=36,
        max_morphemes=72,
    )
    assert [w.unit_id for w in wins] == ["john.1.1+2"]


def test_coverage_and_alignment_loss_backpropagates() -> None:
    raw = torch.randn(2, 8, 12, requires_grad=True)
    attn = torch.softmax(raw, dim=-1)
    tgt_out = torch.randint(3, 50, (2, 8))
    src_groups = torch.tensor(
        [[0, 1, 1, 0, 2, 0, 3, 3, 0, 0, 0, 0], [0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0]]
    )
    align_tgts = torch.tensor([[1, 0, 2, 0, 3, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0]])
    gmask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    cov_l, aln_l = _compute_cov_and_align_loss(
        attn, tgt_out, src_groups, align_tgts, gmask, pad_id=0
    )
    (cov_l + aln_l).backward()
    assert cov_l.item() >= 0.0
    assert aln_l.item() > 0.0
    assert raw.grad is not None and raw.grad.norm().item() > 0.0


def test_local_loop_repeat_allows_parallel_clauses_but_blocks_stutters() -> None:
    # Legitimate repeated 4-token clause across non-adjacent positions is ALLOWED
    seq = [1, 10, 20, 30, 40, 50, 60, 10, 20, 30]
    assert not has_local_loop_repeat(seq, 40)
    # Contiguous stutter loops (1..6 BPE tokens) are BLOCKED
    assert has_local_loop_repeat([1, 7, 7], 7)
    assert has_local_loop_repeat([1, 7, 8, 7], 8)
    assert has_local_loop_repeat([1, 7, 8, 9, 7, 8], 9)
    assert has_local_loop_repeat([1, 7, 8, 9, 10, 11, 7, 8, 9, 10], 11)
    # Global 8-gram repeat is BLOCKED
    long_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 99, 2, 3, 4, 5, 6, 7, 8]
    assert has_local_loop_repeat(long_seq, 9)


def test_bare_coptic_copy_word_and_compound_prep_noun_guard() -> None:
    assert is_bare_coptic_copy_word("μακαριος")
    assert is_bare_coptic_copy_word("μοναχος")
    assert not is_bare_coptic_copy_word("μακάριος")
    assert not is_bare_coptic_copy_word("ο")
    assert not is_bare_coptic_copy_word("εισιν")

    prep_ms = [
        _m(0, "ⲛ", "ⲛ", "PREP"),
        _m(1, "ϩⲏⲧ", "ϩⲏⲧ", "N"),
        _m(2, "ⲥ", "ⲛⲧⲟⲥ", "PPERO"),
    ]
    assert is_grammaticalized_prep_noun(prep_ms, 1)

    det_ms = [
        _m(0, "ⲡⲉⲩ", "ⲡⲉⲩ", "PPOS"),
        _m(1, "ϩⲏⲧ", "ϩⲏⲧ", "N"),
    ]
    assert not is_grammaticalized_prep_noun(det_ms, 1)


def test_position_aware_unmatched_content_and_sahidic_orth_gloss_fallback() -> None:
    # 4 occurrences of ⲙⲟⲩⲓ interleaved with 4 ⲣⲱⲙⲉ; clause-1 ⲙⲟⲩⲓ (idx 0) dropped
    cands = [
        (6, {"λεων"}),
        (10, {"ανθρωποσ"}),
        (17, {"λεων"}),
        (20, {"ανθρωποσ"}),
        (27, {"ανθρωποσ"}),
        (31, {"λεων"}),
        (37, {"λεων"}),
        (41, {"ανθρωποσ"}),
    ]
    pred = [
        "ανθρωποσ",
        "λεων",
        "ανθρωποσ",
        "ανθρωποσ",
        "λεων",
        "λεων",
        "ανθρωποσ",
    ]
    assert find_unmatched_content_indices(cands, pred) == [0]

    ms = [
        _m(0, "ⲅⲁⲇ", "ⲅⲁⲇ", "N"),
        _m(1, "ϭⲣⲟⲙⲡⲉ", "ϭⲣⲟⲙⲡⲉ", "N"),
        _m(2, "ϣⲁϣⲧ", "ϣⲁϣⲧ", "N"),
    ]
    hints = {
        "ⲅⲁⲇ": "Γάδ",
        "~ⲅⲁⲇ|N": "γάλααδ",
        "~ϭⲣⲟⲙⲡⲉ|N": "περιστερά",
        "~aϣⲟϣⲧ|N": "κλείς",
    }
    ser, _ = serialize_coptic_morphemes(ms, greekify=True, native_gloss_hints=hints)
    # Exact bare lemma must beat fuzzy ~orth key when both exist
    assert "<GLOSS> Γάδ" in ser
    assert "γάλααδ" not in ser
    # Unattested orthographic variants without exact keys fall back to ~orth / ~ao
    assert "<GLOSS> περιστερά" in ser
    assert "<GLOSS> κλείς" in ser


def test_cop_resumptive_crel_and_unbounded_constraints(tmp_path: Path) -> None:
    texts = [
        "εἶπεν Ἰησοῦς μακάριος ἐστίν ὁ λέων ὅν ὁ ἄνθρωπος φάγεται",
        "καί ὁ μοναχός καί ἡ γνῶσις καί ὁ προφήτης καί ὁ κόσμος",
        "καί τό δεῖπνον καί ὁ τόπος καί ὁ καρπός καί ὁ ἀσκός",
    ] * 12
    tok = BilingualTokenizer.train(
        texts=texts,
        out_prefix=tmp_path / "tok_cop",
        vocab_size=384,
        user_defined_symbols=SPECIAL_USER_SYMBOLS_V3,
    )
    inf = GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={},
        surface_analyses={},
    )
    loans = {
        "ⲙⲁⲕⲁⲣⲓⲟⲥ": [LoanwordEntry("ⲙⲁⲕⲁⲣⲓⲟⲥ", "μακάριος", "A-")],
        "ⲙⲟⲛⲁⲭⲟⲥ": [LoanwordEntry("ⲙⲟⲛⲁⲭⲟⲥ", "μοναχός", "N-")],
        "ⲅⲛⲱⲥⲓⲥ": [LoanwordEntry("ⲅⲛⲱⲥⲓⲥ", "γνῶσις", "N-")],
        "ⲡⲣⲟⲫⲏⲧⲏⲥ": [LoanwordEntry("ⲡⲣⲟⲫⲏⲧⲏⲥ", "προφήτης", "N-")],
        "ⲕⲟⲥⲙⲟⲥ": [LoanwordEntry("ⲕⲟⲥⲙⲟⲥ", "κόσμος", "N-")],
        "ⲇⲓⲡⲛⲟⲛ": [LoanwordEntry("ⲇⲓⲡⲛⲟⲛ", "δεῖπνον", "N-")],
        "ⲧⲟⲡⲟⲥ": [LoanwordEntry("ⲧⲟⲡⲟⲥ", "τόπος", "N-")],
        "ⲕⲁⲣⲡⲟⲥ": [LoanwordEntry("ⲕⲁⲣⲡⲟⲥ", "καρπός", "N-")],
    }
    ccl = CclLexicon(loanwords=loans, itacism_index={}, greek_genders={})
    ms = [
        _m(0, "ⲙⲁⲕⲁⲣⲓⲟⲥ", "ⲙⲁⲕⲁⲣⲓⲟⲥ", "N", lang="Greek"),
        _m(1, "ⲡⲉ", "ⲡⲉ", "COP"),
        _m(2, "ⲡ", "ⲡ", "ART"),
        _m(3, "ⲙⲟⲩⲓ", "ⲙⲟⲩⲓ", "N"),
        _m(4, "ⲉⲧⲉ", "ⲉⲧⲉⲣⲉ", "CREL"),
        _m(5, "ⲡ", "ⲡ", "ART"),
        _m(6, "ⲣⲱⲙⲉ", "ⲣⲱⲙⲉ", "N"),
        _m(7, "ⲟⲩⲟⲙ", "ⲟⲩⲱⲙ", "V"),
        _m(8, "ϥ", "ⲛⲧⲟϥ", "PPERO"),
        # ART + CREL (headless relative) must NOT add accusative ὅς
        _m(9, "ⲡ", "ⲡ", "ART"),
        _m(10, "ⲉⲧ", "ⲉⲧⲉⲣⲉ", "CREL"),
        _m(11, "ⲥⲱⲧⲙ", "ⲥⲱⲧⲙ", "V"),
        _m(12, "ϥ", "ⲛⲧⲟϥ", "PPERO"),
        # 7 more hard loans (>6 total) must all get tries (no cap at 6)
        _m(13, "ⲙⲟⲛⲁⲭⲟⲥ", "ⲙⲟⲛⲁⲭⲟⲥ", "N", lang="Greek"),
        _m(14, "ⲅⲛⲱⲥⲓⲥ", "ⲅⲛⲱⲥⲓⲥ", "N", lang="Greek"),
        _m(15, "ⲡⲣⲟⲫⲏⲧⲏⲥ", "ⲡⲣⲟⲫⲏⲧⲏⲥ", "N", lang="Greek"),
        _m(16, "ⲕⲟⲥⲙⲟⲥ", "ⲕⲟⲥⲙⲟⲥ", "N", lang="Greek"),
        _m(17, "ⲇⲓⲡⲛⲟⲛ", "ⲇⲓⲡⲛⲟⲛ", "N", lang="Greek"),
        _m(18, "ⲧⲟⲡⲟⲥ", "ⲧⲟⲡⲟⲥ", "N", lang="Greek"),
        _m(19, "ⲕⲁⲣⲡⲟⲥ", "ⲕⲁⲣⲡⲟⲥ", "N", lang="Greek"),
    ]
    cs = build_chunk_lexical_constraints(
        ms, ccl_lexicon=ccl, inflector=inf, tokenizer=tok
    )
    labels = [c.label for c in cs]
    assert "εἰμί" in labels
    assert labels.count("ὅς") == 1  # seq 4 only; ART+CREL at seq 10 excluded
    assert len(cs) == 10  # 8 loans + 1 COP + 1 resumptive CREL (not capped at 6)

    # Guard edge cases: COP+CREL cleft, CREL+ADV/PREP/EXIST, 1sg/2sg PPERO,
    # clause-boundary crossing, and imperfect CPRET+COP must NOT fire.
    from zetodos_thomas.neural.constrained_beam import (
        _cop_is_nominal_predicate,
        _crel_has_resumptive_obj,
    )

    cleft = [
        _m(0, "ⲡⲉ", "ⲡⲉ", "COP"),
        _m(1, "ⲛⲧ", "ⲉⲧⲉⲣⲉ", "CREL"),
        _m(2, "ⲁ", "ⲁ", "APST"),
        _m(3, "ϥ", "ⲛⲧⲟϥ", "PPERS"),
        _m(4, "ⲁⲁ", "ⲉⲓⲣⲉ", "V"),
        _m(5, "ⲧ", "ⲁⲛⲟⲕ", "PPERO"),
    ]
    assert not _crel_has_resumptive_obj(cleft, 1)

    non_3p = [
        _m(0, "ⲉⲛⲧ", "ⲉⲧⲉⲣⲉ", "CREL"),
        _m(1, "ⲁ", "ⲁ", "APST"),
        _m(2, "ⲩ", "ⲛⲧⲟⲟⲩ", "PPERS"),
        _m(3, "ⲥⲁⲛⲟⲩϣ", "ⲥⲁⲛⲟⲩϣ", "V"),
        _m(4, "ⲕ", "ⲛⲧⲟⲕ", "PPERO"),
    ]
    assert not _crel_has_resumptive_obj(non_3p, 0)

    dem_then_next_clause = [
        _m(0, "ⲉⲧ", "ⲉⲧⲉⲣⲉ", "CREL"),
        _m(1, "ⲙⲙⲁⲩ", "ⲙⲙⲁⲩ", "ADV"),
        _m(2, "ⲁ", "ⲁ", "APST"),
        _m(3, "ϥ", "ⲛⲧⲟϥ", "PPERS"),
        _m(4, "ⲧⲁⲁ", "ϯ", "V"),
        _m(5, "ⲥ", "ⲛⲧⲟⲥ", "PPERO"),
    ]
    assert not _crel_has_resumptive_obj(dem_then_next_clause, 0)

    embedded_vstat_subject = [
        _m(0, "ⲉⲛⲧ", "ⲉⲧⲉⲣⲉ", "CREL"),
        _m(1, "ⲁ", "ⲁ", "APST"),
        _m(2, "ⲓⲏⲥⲟⲩⲥ", "ⲓⲏⲥⲟⲩⲥ", "NPROP"),
        _m(3, "ⲉⲧ", "ⲉⲧⲉⲣⲉ", "CREL"),
        _m(4, "ⲟⲛϩ", "ⲱⲛϩ", "VSTAT"),
        _m(5, "ϫⲟⲟ", "ϫⲱ", "V"),
        _m(6, "ⲩ", "ⲛⲧⲟⲟⲩ", "PPERO"),
    ]
    assert _crel_has_resumptive_obj(embedded_vstat_subject, 0)
    assert not _crel_has_resumptive_obj(embedded_vstat_subject, 3)

    imperfect_cop = [
        _m(0, "ⲛⲉ", "ⲛⲉⲣⲉ", "CPRET"),
        _m(1, "ϥ", "ⲛⲧⲟϥ", "PPERS"),
        _m(2, "ⲥⲱⲧⲙ", "ⲥⲱⲧⲙ", "V"),
        _m(3, "ⲡⲉ", "ⲡⲉ", "COP"),
    ]
    assert not _cop_is_nominal_predicate(imperfect_cop, 3)


def test_reconstruct_inspect_defaults_to_model_v1() -> None:
    import inspect

    from zetodos_thomas.cli.reconstruct_cmd import reconstruct_inspect

    sig = inspect.signature(reconstruct_inspect)
    assert sig.parameters["model"].default.default == "v1"
