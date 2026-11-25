# SPDX-License-Identifier: Apache-2.0
"""Unit tests for from-scratch neural NMT model-v2 components & determinism."""

from __future__ import annotations

from pathlib import Path

import torch

from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon, LoanwordEntry
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.neural.corpus import (
    normalize_greek_target,
    serialize_coptic_morphemes,
)
from zetodos_thomas.neural.decoder_v2 import (
    NeuralKoineDecoder,
    RerankWeights,
    load_analyzer_inflector,
    save_analyzer_shards,
)
from zetodos_thomas.neural.tokenizer import BilingualTokenizer
from zetodos_thomas.neural.transformer import (
    KoineCausalLM,
    KoineCausalLMConfig,
    KoineSeq2SeqTransformer,
    Seq2SeqConfig,
)
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit
from zetodos_thomas.schemas.source import UnitKind


def test_neural_corpus_normalization_and_serialization() -> None:
    assert (
        normalize_greek_target("Ἀβραὰμ ἐγέννησε(ν) τὸν Ἰσαάκ.")
        == "Ἀβραάμ ἐγέννησεν τόν Ἰσαάκ"
    )
    ccl = CclLexicon(
        loanwords={
            "ϩⲉⲣⲙⲏⲛⲉⲓⲁ": [LoanwordEntry("ϩⲉⲣⲙⲏⲛⲉⲓⲁ", "ἑρμηνεία", "N-", gender="F")]
        },
        itacism_index={},
        greek_genders={"ἑρμηνεία": "F"},
    )
    ms = [
        CopticMorpheme(seq=0, group_seq=0, norm="ⲡ", lemma="ⲡ", pos="ART"),
        CopticMorpheme(seq=1, group_seq=0, norm="ϩⲏⲡ⳿", lemma="ϩⲱⲡ⳿", pos="VSTAT"),
        CopticMorpheme(
            seq=2,
            group_seq=1,
            norm="ϩⲉⲣⲙⲏⲛⲉⲓⲁ",
            lemma="ϩⲉⲣⲙⲏⲛⲉⲓⲁ",
            pos="N",
            lang="Greek",
        ),
    ]
    txt, spans = serialize_coptic_morphemes(
        ms, ccl_lexicon=ccl, domain_tag="<DOM_NT>", v1_draft="ὁ λόγος ἑρμηνεία"
    )
    assert txt.startswith("<DOM_NT> ⲡ ")
    assert "<POS_ART>" not in txt
    assert "ϩⲏⲡ <LEM> ϩⲱⲡ <POS_VSTAT>" in txt
    assert "<LOAN> ἑρμηνεία" in txt
    assert "<V1> ὁ λόγος ἑρμηνεία" in txt
    assert [sq for sq, _, _ in spans] == [0, 1, 2]


def test_neural_v2_tokenizer_transformer_safetensors_and_determinism(
    tmp_path: Path,
) -> None:
    ccl = CclLexicon(
        loanwords={
            "ϩⲉⲣⲙⲏⲛⲉⲓⲁ": [LoanwordEntry("ϩⲉⲣⲙⲏⲛⲉⲓⲁ", "ἑρμηνεία", "N-", gender="F")]
        },
        itacism_index={},
        greek_genders={"ἑρμηνεία": "F"},
    )
    texts = [
        "<DOM_NT> ⲡ <POS_ART> ϣⲁϫⲉ <POS_N> ϩⲏⲡ <LEM> ϩⲱⲡ <POS_VSTAT>",
        "<DOM_NT> ⲧ <POS_ART> ϩⲉⲣⲙⲏⲛⲉⲓⲁ <POS_N> <LOAN> ἑρμηνεία",
        "ὁ λόγος κεκρυμμένος ἐστίν",
        "τήν ἑρμηνείαν τῶν λόγων τούτων",
    ] * 8
    tok = BilingualTokenizer.train(
        texts=texts, out_prefix=tmp_path / "tokenizer", vocab_size=360
    )
    cfg = Seq2SeqConfig(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_heads=4,
        d_ff=96,
        num_enc_layers=2,
        num_dec_layers=2,
        dropout=0.0,
        attn_dropout=0.0,
    )
    torch.manual_seed(7)
    model = KoineSeq2SeqTransformer(cfg)
    w_path = tmp_path / "forward-0.safetensors"
    model.save_safetensors(w_path, fp16=True)
    reloaded = KoineSeq2SeqTransformer.load_safetensors(w_path, cfg, device="cpu")

    lm_cfg = KoineCausalLMConfig(
        vocab_size=tok.vocab_size,
        d_model=48,
        n_heads=4,
        d_ff=96,
        num_layers=2,
        dropout=0.0,
        attn_dropout=0.0,
    )
    lm = KoineCausalLM(lm_cfg)
    lm_path = tmp_path / "koine-lm.safetensors"
    lm.save_safetensors(lm_path, fp16=True)
    lm_reloaded = KoineCausalLM.load_safetensors(lm_path, lm_cfg, device="cpu")
    scores = lm_reloaded.score_token_seqs([tok.encode("ὁ λόγος ἐστίν")])
    assert len(scores) == 1 and scores[0] < 0.0

    inf = GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={"λόγος": "M"},
        surface_analyses={"λόγος": [("λόγος", "N-:----NSM-", -0.5)]},
        known_lemmas={"λογος": ("λόγος", "N-")},
    )
    save_analyzer_shards(inf, tmp_path)
    loaded_inf = load_analyzer_inflector(tmp_path)
    assert loaded_inf.analyze_surface("λόγος")[0][0] == "λόγος"

    dec = NeuralKoineDecoder(
        tokenizer=tok,
        forward_models=[reloaded],
        ccl_lexicon=ccl,
        analyzer_inflector=loaded_inf,
        reverse_model=reloaded,
        koine_lm=lm_reloaded,
        weights=RerankWeights(w_fwd=1.0, w_rev=0.2, w_lm=0.1, beam_size=3),
        device="cpu",
    )
    unit = CopticUnit(
        unit_id="thomas.test.v2",
        source_id="dilley-thomas",
        book="thomas",
        chapter=1,
        verse=1,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(seq=0, group_seq=0, norm="ⲡ", lemma="ⲡ", pos="ART"),
            CopticMorpheme(seq=1, group_seq=0, norm="ϣⲁϫⲉ", lemma="ϣⲁϫⲉ", pos="N"),
            CopticMorpheme(seq=2, group_seq=1, norm="ϩⲏⲡ", lemma="ϩⲱⲡ", pos="VSTAT"),
        ],
    )
    out1 = dec.decode_unit(unit)
    out2 = dec.decode_unit(unit)
    assert out1.greek_text == out2.greek_text
    assert len(out1.tokens) > 0
    assert all(len(t.coptic_seqs) > 0 for t in out1.tokens)


def test_model_v3_greekification_unaccented_and_decoder_roundtrip(
    tmp_path: Path,
) -> None:
    from zetodos_thomas.neural.accent_restorer import KoineAccentRestorer
    from zetodos_thomas.neural.corpus import (
        SPECIAL_USER_SYMBOLS_V3,
        greekify_coptic,
    )

    # Script unification maps Greek-derived loanwords identically to bare Greek
    assert greekify_coptic("ⲕⲟⲥⲙⲟⲥ") == "κοσμος"
    assert greekify_coptic("ⲡⲟⲗⲓⲥ") == "πολις"
    # Demotic letters remain distinct & intact
    assert greekify_coptic("ϣⲁϫⲉ") == "ϣαϫε"
    assert greekify_coptic("ϩⲉⲣⲙⲏⲛⲉⲓⲁ") == "ϩερμηνεια"

    ccl = CclLexicon(
        loanwords={
            "ⲕⲟⲥⲙⲟⲥ": [
                LoanwordEntry(
                    coptic_lemma="ⲕⲟⲥⲙⲟⲥ",
                    greek_lemma="κόσμος",
                    greek_pos="N-",
                )
            ]
        },
        itacism_index={},
        greek_genders={},
    )
    ms = [
        CopticMorpheme(seq=0, group_seq=0, norm="ⲡ", lemma="ⲡ", pos="ART"),
        CopticMorpheme(
            seq=1, group_seq=0, norm="ⲕⲟⲥⲙⲟⲥ", lemma="ⲕⲟⲥⲙⲟⲥ", pos="N", lang="Greek"
        ),
        CopticMorpheme(seq=2, group_seq=1, norm="ⲃⲱ", lemma="ⲃⲱ", pos="N"),
    ]
    gloss_hints = {"ⲃⲱ": "δενδρον"}
    s_txt, spans = serialize_coptic_morphemes(
        ms,
        ccl_lexicon=ccl,
        domain_tag="<DOM_NT>",
        greekify=True,
        unaccented_hints=True,
        native_gloss_hints=gloss_hints,
        target_tag="<TGT_GRC>",
    )
    assert s_txt.startswith("<DOM_NT> <TGT_GRC> π κοσμος <LOAN> κοσμος")
    assert "<GLOSS> δενδρον" in s_txt
    assert len(spans) == 3

    # Unified BPE tokenizer shares pieces between Greekified loanword & bare Greek
    texts = [s_txt, "ο κοσμος δενδρον", "η πολη του κοσμου"] * 25
    tok = BilingualTokenizer.train(
        texts=texts,
        out_prefix=tmp_path / "tokenizer",
        vocab_size=360,
        user_defined_symbols=SPECIAL_USER_SYMBOLS_V3,
    )
    assert tok.has_piece("<TGT_GRC>")
    assert tok.has_piece("<GLOSS>")
    assert tok.encode("κοσμος", add_bos=False, add_eos=False) == tok.encode(
        greekify_coptic("ⲕⲟⲥⲙⲟⲥ"), add_bos=False, add_eos=False
    )

    cfg = Seq2SeqConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_heads=4,
        d_ff=64,
        num_enc_layers=2,
        num_dec_layers=2,
        greekify_source=True,
        unaccented_target=True,
        use_gloss_hints=True,
        use_target_lang_tag=True,
    )
    (tmp_path / "seq2seq-config.json").write_text(
        cfg.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    model = KoineSeq2SeqTransformer(cfg)
    model.save_safetensors(tmp_path / "forward-0.safetensors", fp16=True)
    ccl.save_json(tmp_path / "ccl-loanwords.json")
    (tmp_path / "native-gloss-hints.json").write_text(
        '{"ⲃⲱ":"δενδρον"}\n', encoding="utf-8"
    )
    ar = KoineAccentRestorer(
        unigrams={"ο": "ὁ", "κοσμος": "κόσμος", "δενδρον": "δένδρον"},
        prev_bigrams={},
    )
    ar.save_shards(tmp_path)
    inf = GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={},
        surface_analyses={
            "κόσμος": [("κόσμος", "N- ----NSM-", 0.0)],
            "δένδρον": [("δένδρον", "N- ----NSN-", 0.0)],
        },
        known_lemmas={},
    )
    save_analyzer_shards(inf, tmp_path)
    RerankWeights(beam_size=2).save_json(tmp_path / "rerank-weights.json")

    dec = NeuralKoineDecoder.from_model_dir(tmp_path, device="cpu")
    assert dec._greekify is True
    assert dec._unaccented is True
    assert dec.accent_restorer is not None
    assert dec.native_gloss_hints == {"ⲃⲱ": "δενδρον"}
    unit = CopticUnit(
        unit_id="test.v3",
        source_id="test",
        book="test",
        chapter=1,
        verse=1,
        original_text="ⲡⲕⲟⲥⲙⲟⲥ ⲃⲱ",
        normalized_text="ⲡ ⲕⲟⲥⲙⲟⲥ ⲃⲱ",
        morphemes=ms,
    )
    u1 = dec.decode_unit(unit)
    u2 = dec.decode_unit(unit)
    assert u1.greek_text == u2.greek_text
    assert len(u1.tokens) > 0
