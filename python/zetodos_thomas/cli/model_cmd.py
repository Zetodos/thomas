# SPDX-License-Identifier: Apache-2.0
"""``thomas model`` Typer subcommands: build & inspect offline model-v1."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.align.lemma_aligner import LexicalTranslationModel
from zetodos_thomas.ingest.download import CCL_TEI_FILENAME, SAHIDIC_OT_ZIP_FILENAME
from zetodos_thomas.ingest.holdouts import (
    SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS,
    all_excluded_nt_ids,
    clean_nt_train_ids,
    load_manifest_sha256,
    load_nt_split,
    write_training_provenance,
)
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt, pair_horner_morphgnt
from zetodos_thomas.ingest.ot_swete import load_ot_swete_pairs
from zetodos_thomas.ingest.paths import require_repo_root, resolve_nt_split_path
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.lexicon.idioms import IdiomAndValencyTable
from zetodos_thomas.morphology.coptic_analyzer import CopticAnalyzer
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.reconstruct.decoder import (
    KoineBeamDecoder,
    tune_weights_on_dev,
)
from zetodos_thomas.reconstruct.koine_lm import KoineNgramLM
from zetodos_thomas.schemas.holdout import PassageRole

MODEL_REL = Path("data/mappings/model-v1")
MODEL_V2_REL = Path("data/mappings/model-v2")
MODEL_V3_REL = Path("data/mappings/model-v3")
MODEL_V4_REL = Path("data/mappings/model-v4")
CCL_MANIFEST_REL = Path("data/manifests/ccl-ddglc-tei.json")

model_app = typer.Typer(
    name="model",
    help="Build and validate the frozen Coptic->Koine alignment & LM model.",
    no_args_is_help=True,
)


@model_app.command("build")
def build_model(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Output model directory (default: data/mappings/model-v1)",
    ),
    exclude_synoptic_thomas: bool = typer.Option(
        False,
        "--exclude-synoptic-thomas/--include-synoptic-thomas",
        help="Optional ablation: exclude 63 NT Synoptic parallels of P.Oxy. logia",
    ),
) -> None:
    """Train CopticAnalyzer, CCL loanwords, seeded eflomal, Koine LM & tune on dev."""
    root = require_repo_root(repo_root)
    downloads = root / "data" / "downloads"
    split_path = resolve_nt_split_path(root)
    target_dir = out_dir or (root / MODEL_REL)
    target_dir.mkdir(parents=True, exist_ok=True)

    split = load_nt_split(split_path)
    holdout_ids = {
        uid
        for p in split.passages
        if p.role is PassageRole.HOLDOUT
        for uid in p.unit_ids
    }
    dev_ids = {
        uid for p in split.passages if p.role is PassageRole.DEV for uid in p.unit_ids
    }
    clean_train_ids = clean_nt_train_ids(
        split, exclude_synoptic_thomas=exclude_synoptic_thomas
    )
    excluded_from_model = all_excluded_nt_ids(
        split, include_dev=True, exclude_synoptic_thomas=exclude_synoptic_thomas
    )
    excluded_synoptic = (
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS if exclude_synoptic_thomas else frozenset()
    )

    ot_zip = downloads / "sahidic-ot" / SAHIDIC_OT_ZIP_FILENAME
    ccl_xml = downloads / "ccl-ddglc" / CCL_TEI_FILENAME
    ccl_manifest = root / CCL_MANIFEST_REL
    ccl_expected_sha = (
        load_manifest_sha256(ccl_manifest) if ccl_manifest.is_file() else None
    )
    morphgnt_dir = downloads / "morphgnt-sblgnt"

    typer.echo("1/6 Training CopticAnalyzer (sahidic.ot + CCL v1.2)...")
    coptic_analyzer = CopticAnalyzer.train(ot_zip=ot_zip, ccl_xml=ccl_xml)
    c_shards = coptic_analyzer.save_shards(target_dir)

    typer.echo("2/6 Building CCL/DDGLC loanword & compound index (pure TEI)...")
    ccl_lexicon = CclLexicon.from_tei_xml(ccl_xml, expected_sha256=ccl_expected_sha)
    ccl_lexicon.save_json(target_dir / "ccl-loanwords.json")

    typer.echo(
        "3/6 Building Koine Greek inflector (+ CCL/Swete LXX, no holdout/Synoptic)..."
    )
    inflector = GreekInflector.build(
        morphgnt_dir=morphgnt_dir,
        exclude_unit_ids=excluded_from_model,
        ccl_lexicon=ccl_lexicon,
        swete_dir=downloads / "lxx-swete",
    )
    g_shards = inflector.save_shards(target_dir)

    typer.echo(
        "4/6 Running seeded bidirectional eflomal on clean NT train (3x) + OT x LXX..."
    )
    raw_nt = pair_horner_morphgnt(
        sword_root=downloads / "sword",
        morphgnt_dir=morphgnt_dir,
    )
    train_raw_c = [c for c, g in raw_nt if g.unit_id in clean_train_ids]
    greek_train_by_id = {
        g.unit_id: g for _c, g in raw_nt if g.unit_id in clean_train_ids
    }
    analyzed_nt_c = coptic_analyzer.analyze_units(train_raw_c)
    clean_nt_pairs = [
        (c_u, greek_train_by_id[c_u.unit_id])
        for c_u in analyzed_nt_c
        if c_u.unit_id in greek_train_by_id
    ]
    ot_pairs = load_ot_swete_pairs(
        tt_zip=ot_zip,
        swete_dir=downloads / "lxx-swete",
    )
    lexicon = LexicalTranslationModel.build_from_bitext(
        nt_pairs=clean_nt_pairs,
        ot_pairs=ot_pairs,
        inflector=inflector,
        ccl_lexicon=ccl_lexicon,
        nt_weight=3.0,
        repo_root=root,
    )
    l_shards = lexicon.save_shards(target_dir)

    typer.echo(
        "5/6 Extracting empirical MWE idioms/valency & training 3-gram Koine LM..."
    )
    morphgnt_units = load_morphgnt_nt(morphgnt_dir)
    idioms = IdiomAndValencyTable.build_from_bitext(
        nt_pairs=clean_nt_pairs,
        ot_pairs=ot_pairs,
        morphgnt_units=morphgnt_units.values(),
        exclude_unit_ids=excluded_from_model,
    )
    idioms.save_json(target_dir / "idiom-valency.json")

    lm = KoineNgramLM.train(
        downloads_dir=downloads,
        split_path=split_path,
        inflector=inflector,
        nt_weight=3.0,
        exclude_synoptic_thomas=exclude_synoptic_thomas,
    )
    lm_shards = lm.save_shards(target_dir)

    typer.echo(
        "6/6 Tuning decoder log-linear weights strictly on NT DEV split (51 verses)..."
    )
    decoder = KoineBeamDecoder(
        lexicon=lexicon,
        ccl_lexicon=ccl_lexicon,
        idioms=idioms,
        inflector=inflector,
        lm=lm,
    )
    dev_raw_c = [c for c, g in raw_nt if g.unit_id in dev_ids]
    analyzed_dev = coptic_analyzer.analyze_units(dev_raw_c)
    dev_pairs = [
        (cu, [t.lemma for t in morphgnt_units[cu.unit_id].tokens])
        for cu in analyzed_dev
        if cu.unit_id in morphgnt_units
    ]
    tuned = tune_weights_on_dev(decoder=decoder, dev_pairs=dev_pairs)
    tuned.save_json(target_dir / "decoder-weights.json")

    write_training_provenance(
        out_dir=target_dir,
        model_id=target_dir.name,
        split_path=split_path,
        ccl_xml_path=ccl_xml,
        nt_train_ids=[g.unit_id for _c, g in clean_nt_pairs],
        ot_train_ids=[g.unit_id for _c, g in ot_pairs],
        excluded_holdout_ids=holdout_ids,
        excluded_dev_ids=dev_ids,
        excluded_synoptic_ids=excluded_synoptic,
        extra={
            "empirical_idiom_count": len(idioms._idioms),
            "empirical_verb_case_count": len(idioms.verb_cases),
            "empirical_prep_case_count": len(idioms.prep_cases),
        },
    )

    all_files = sorted(target_dir.glob("*.json"))
    oversized = [p for p in all_files if p.stat().st_size >= 1024 * 1024]
    if oversized:
        msg = f"model shard(s) >= 1024 KiB: {[str(p) for p in oversized]}"
        typer.secho(msg, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(
        f"OK: froze model-v1 ({len(all_files)} files, all < 1024 KiB; "
        f"coptic={len(c_shards)}, greek={len(g_shards)}, lex={len(l_shards)}, "
        f"lm={len(lm_shards)}; tuned={tuned}) to {target_dir}"
    )


@model_app.command("build-v2")
def build_model_v2(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Output model directory (default: data/mappings/model-v2)",
    ),
    seeds: int = typer.Option(
        1, "--seeds", min=1, max=5, help="Forward ensemble seeds"
    ),
    epochs: int = typer.Option(
        26, "--epochs", min=1, help="Main joint training epochs"
    ),
    ft_epochs: int = typer.Option(
        8, "--ft-epochs", min=0, help="Gospel (<NT>) register fine-tune epochs"
    ),
    pretrain_epochs: int = typer.Option(
        0, "--pretrain-epochs", min=0, help="Monolingual span-denoising epochs"
    ),
    backtranslate: bool = typer.Option(
        False,
        "--backtranslate/--no-backtranslate",
        help="Train reverse Koine->Coptic model & back-translate unaligned Swete LXX",
    ),
    train_lm: bool = typer.Option(
        False,
        "--train-lm/--no-train-lm",
        help="Train standalone from-scratch Koine causal LM for reranking",
    ),
    use_v1_draft: bool = typer.Option(
        True,
        "--use-v1-draft/--no-use-v1-draft",
        help="Condition encoder on leak-free model-v1 Greek draft with dropout noise",
    ),
    include_v1: bool = typer.Option(
        False,
        "--include-v1/--no-include-v1",
        help="Pool model-v1 hypothesis as an extra candidate during reranking",
    ),
    reuse_aux: bool = typer.Option(
        False,
        "--reuse-aux/--no-reuse-aux",
        help="Reuse existing tokenizer, reverse model, BT cache & LM if present",
    ),
    reuse_seeds: bool = typer.Option(
        False,
        "--reuse-seeds/--no-reuse-seeds",
        help="Reuse already trained forward-*.safetensors seeds when scaling ensemble",
    ),
    vocab_size: int = typer.Option(8000, "--vocab-size", min=512),
    d_model: int = typer.Option(384, "--d-model", min=64),
    layers: int = typer.Option(5, "--layers", min=2),
    device: str = typer.Option(
        "auto", "--device", help="Training device: auto|mps|cpu"
    ),
    exclude_synoptic_thomas: bool = typer.Option(
        False,
        "--exclude-synoptic-thomas/--include-synoptic-thomas",
        help="Optional ablation: exclude 63 NT Synoptic parallels of P.Oxy. logia",
    ),
) -> None:
    """Train leak-free from-scratch neural Coptic->Koine model-v2 (Git LFS ready)."""
    import dataclasses
    import json
    import random

    from zetodos_thomas.neural.corpus import (
        build_parallel_corpus_bundle,
        make_sub_coptic_unit,
        normalize_greek_target,
        serialize_coptic_morphemes,
    )
    from zetodos_thomas.neural.decoder_v2 import (
        NeuralKoineDecoder,
        RerankWeights,
        load_analyzer_inflector,
        save_analyzer_shards,
        tune_rerank_weights_on_dev,
    )
    from zetodos_thomas.neural.tokenizer import BilingualTokenizer
    from zetodos_thomas.neural.trainer import (
        generate_backtranslations,
        make_forward_pairs_epoch,
        pretrain_denoising,
        resolve_device,
        train_causal_koine_lm,
        train_seq2seq,
    )
    from zetodos_thomas.neural.transformer import (
        KoineCausalLM,
        KoineCausalLMConfig,
        KoineSeq2SeqTransformer,
        Seq2SeqConfig,
    )

    root = require_repo_root(repo_root)
    downloads = root / "data" / "downloads"
    split_path = resolve_nt_split_path(root)
    v1_dir = root / MODEL_REL
    target_dir = out_dir or (root / MODEL_V2_REL)
    target_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    typer.echo(f"Training model-v2 on device={dev} -> {target_dir}")

    split = load_nt_split(split_path)
    holdout_ids = {
        uid
        for p in split.passages
        if p.role is PassageRole.HOLDOUT
        for uid in p.unit_ids
    }
    dev_ids = {
        uid for p in split.passages if p.role is PassageRole.DEV for uid in p.unit_ids
    }
    excluded_from_model = all_excluded_nt_ids(
        split, include_dev=True, exclude_synoptic_thomas=exclude_synoptic_thomas
    )
    excluded_synoptic = (
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS if exclude_synoptic_thomas else frozenset()
    )
    ccl_xml = downloads / "ccl-ddglc" / CCL_TEI_FILENAME

    # 1. Load leak-free CopticAnalyzer & CCL loanwords, build offline Greek analyzer
    coptic_analyzer = CopticAnalyzer.load_shards(v1_dir)
    ccl_lexicon = CclLexicon.load_json(v1_dir / "ccl-loanwords.json")
    ccl_lexicon.save_json(target_dir / "ccl-loanwords.json")

    if not reuse_aux or not (target_dir / "greek-analyzer-0.json").is_file():
        typer.echo(
            "1/6 Building leak-free Greek surface analyzer (no holdout/dev/Synoptic)..."
        )
        inflector = GreekInflector.build(
            morphgnt_dir=downloads / "morphgnt-sblgnt",
            exclude_unit_ids=excluded_from_model,
            ccl_lexicon=ccl_lexicon,
            swete_dir=downloads / "lxx-swete",
        )
        save_analyzer_shards(inflector, target_dir)
    else:
        typer.echo("1/6 Reusing existing Greek surface analyzer shards...")
    analyzer_inflector = load_analyzer_inflector(target_dir)

    # 2. Build strictly leak-free parallel & monolingual corpus bundle
    typer.echo("2/6 Building zero-leakage NT+OT+Swete+CCL neural dataset...")
    bundle = build_parallel_corpus_bundle(
        downloads_dir=downloads,
        split_path=split_path,
        coptic_analyzer=coptic_analyzer,
        ccl_lexicon=ccl_lexicon,
        exclude_synoptic_thomas=exclude_synoptic_thomas,
    )
    typer.echo(
        f"  Dataset: nt_train={len(bundle.nt_train)}, nt_dev={len(bundle.nt_dev)}, "
        f"ot_train={len(bundle.ot_train)}, clause_aug={len(bundle.clause_aug)}, "
        f"lex_aug={len(bundle.lex_aug)}, "
        f"swete_unaligned={len(bundle.swete_unaligned_greek)}"
    )

    # 3. Train or reuse shared bilingual BPE tokenizer (strictly no holdout/dev)
    tok_model = target_dir / "tokenizer.model"
    if reuse_aux and tok_model.is_file():
        typer.echo("3/6 Reusing existing SentencePiece tokenizer...")
        tokenizer = BilingualTokenizer.load(tok_model)
    else:
        typer.echo(
            f"3/6 Training shared SentencePiece BPE tokenizer (V={vocab_size})..."
        )
        tok_texts: list[str] = list(bundle.all_clean_greek)
        for ex in (*bundle.nt_train, *bundle.ot_train):
            s_txt, _ = serialize_coptic_morphemes(
                ex.coptic_morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag=ex.domain_tag,
            )
            tok_texts.append(s_txt)
        for s_l, g_l in bundle.lex_aug:
            tok_texts.append(s_l)
            tok_texts.append(g_l)
        tokenizer = BilingualTokenizer.train(
            texts=tok_texts,
            out_prefix=target_dir / "tokenizer",
            vocab_size=vocab_size,
        )

    cfg = Seq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=6 if d_model % 6 == 0 else 4,
        d_ff=d_model * 4,
        num_enc_layers=layers,
        num_dec_layers=layers,
    )
    (target_dir / "seq2seq-config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    # Optional leak-free model-v1 drafts (beam=2) for dual-source encoder conditioning
    v1_drafts: dict[str, str] = {}
    draft_vocab_pool: list[str] = []
    v1_draft_cache = root / "data" / "downloads" / ".cache-v1-drafts-v2.json"
    v1_fast: KoineBeamDecoder | None = None
    if (use_v1_draft or include_v1) and (v1_dir / "decoder-weights.json").is_file():
        v1_fast = KoineBeamDecoder.from_model_dir(v1_dir)
        v1_fast.weights = dataclasses.replace(v1_fast.weights, beam_size=2)
    if use_v1_draft and v1_fast is not None:
        all_exs = (
            *bundle.nt_train,
            *bundle.ot_train,
            *bundle.clause_aug,
            *bundle.nt_dev,
        )
        if reuse_aux and v1_draft_cache.is_file():
            v1_drafts = {
                str(k): str(v)
                for k, v in json.loads(
                    v1_draft_cache.read_text(encoding="utf-8")
                ).items()
            }
        missing_exs = [ex for ex in all_exs if ex.unit_id not in v1_drafts]
        if missing_exs:
            typer.echo(
                f"  Precomputing leak-free model-v1 drafts for "
                f"{len(missing_exs)} examples (beam=2)..."
            )
            for ex in missing_exs:
                tmp_u = make_sub_coptic_unit(
                    ex.coptic_morphemes, unit_id=ex.unit_id, source_id="train"
                )
                v1_drafts[ex.unit_id] = normalize_greek_target(
                    v1_fast.decode_unit(tmp_u).greek_text
                )
            v1_draft_cache.write_text(
                json.dumps(v1_drafts, ensure_ascii=False), encoding="utf-8"
            )
        vocab_seen: set[str] = set()
        for g_txt in bundle.all_clean_greek[:8000]:
            for w in g_txt.split():
                if len(w) >= 3 and w not in vocab_seen:
                    vocab_seen.add(w)
                    draft_vocab_pool.append(w)
                    if len(draft_vocab_pool) >= 4000:
                        break
            if len(draft_vocab_pool) >= 4000:
                break

    dev_pure_pairs: list[tuple[str, str]] = []
    dev_fwd_pairs: list[tuple[str, str]] = []
    for ex in bundle.nt_dev:
        s_pure, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes, ccl_lexicon=ccl_lexicon, domain_tag="<DOM_NT>"
        )
        dev_pure_pairs.append((s_pure, ex.greek_target))
        s_fwd, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes,
            ccl_lexicon=ccl_lexicon,
            domain_tag="<DOM_NT>",
            v1_draft=v1_drafts.get(ex.unit_id) if use_v1_draft else None,
        )
        dev_fwd_pairs.append((s_fwd, ex.greek_target))

    # 4. Optional reverse Koine->Coptic model & Swete LXX back-translation
    bt_cache = root / "data" / "downloads" / ".cache-bt-swete-v2.json"
    rev_path = target_dir / "reverse.safetensors"
    bt_pairs: list[tuple[str, str]] = []
    reverse_model: KoineSeq2SeqTransformer | None = None
    if backtranslate:
        if reuse_aux and rev_path.is_file() and bt_cache.is_file():
            typer.echo(
                "4/6 Reusing reverse Koine->Coptic model & Swete back-translations..."
            )
            reverse_model = KoineSeq2SeqTransformer.load_safetensors(
                rev_path, cfg, device="cpu"
            )
            bt_pairs = [
                (str(a), str(b))
                for a, b in json.loads(bt_cache.read_text(encoding="utf-8"))
            ]
        else:
            typer.echo(
                "4/6 Training reverse Koine->Coptic model for BT & noisy-channel..."
            )
            dev_rev_pairs = [(f"<REV_COPTIC> {g}", s) for s, g in dev_pure_pairs]

            def _rev_epoch(rng: random.Random) -> list[tuple[str, str]]:
                fwd = make_forward_pairs_epoch(
                    nt_train=bundle.nt_train,
                    ot_train=bundle.ot_train,
                    clause_aug=bundle.clause_aug,
                    lex_aug=[],
                    bt_pairs=[],
                    ccl_lexicon=ccl_lexicon,
                    nt_repeat=2,
                    loan_drop_p=0.0,
                    morpheme_drop_p=0.0,
                    rng=rng,
                )
                return [(f"<REV_COPTIC> {g}", s) for s, g in fwd]

            reverse_model, rev_hist = train_seq2seq(
                config=cfg,
                tokenizer=tokenizer,
                train_pairs_fn=_rev_epoch,
                dev_pairs=dev_rev_pairs,
                epochs=max(12, epochs // 2),
                lr=6e-4,
                seed=777,
                device=dev,
                select_by_chrf=False,
                patience=5,
                log_fn=typer.echo,
            )
            typer.echo(
                f"  Reverse best_ep={rev_hist.best_epoch}, "
                f"dev_loss={rev_hist.best_dev_loss:.4f}"
            )
            reverse_model.save_safetensors(rev_path, fp16=True)
            n_swete = len(bundle.swete_unaligned_greek)
            typer.echo(f"  Back-translating {n_swete} unaligned Swete LXX verses...")
            bt_pairs = generate_backtranslations(
                reverse_model=reverse_model,
                tokenizer=tokenizer,
                greek_texts=bundle.swete_unaligned_greek,
                device=dev,
            )
            bt_cache.write_text(
                json.dumps(bt_pairs, ensure_ascii=False), encoding="utf-8"
            )
            typer.echo(f"  Generated {len(bt_pairs)} clean synthetic BT pairs.")
    else:
        typer.echo("4/6 Skipping back-translation (--no-backtranslate).")

    # 5. Optional standalone Koine Greek Neural LM
    lm_path = target_dir / "koine-lm.safetensors"
    lm_cfg_path = target_dir / "koine-lm-config.json"
    koine_lm: KoineCausalLM | None = None
    if train_lm:
        lm_cfg = KoineCausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            n_heads=6 if d_model % 6 == 0 else 4,
            d_ff=d_model * 4,
            num_layers=4,
        )
        if reuse_aux and lm_path.is_file() and lm_cfg_path.is_file():
            typer.echo("5/6 Reusing standalone Koine Neural LM...")
            koine_lm = KoineCausalLM.load_safetensors(lm_path, lm_cfg, device="cpu")
        else:
            typer.echo("5/6 Training standalone from-scratch Koine Causal LM...")
            lm_cfg_path.write_text(
                json.dumps(lm_cfg.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            koine_lm, lm_loss = train_causal_koine_lm(
                config=lm_cfg,
                tokenizer=tokenizer,
                train_greek_texts=bundle.all_clean_greek,
                dev_greek_texts=[ex.greek_target for ex in bundle.nt_dev],
                epochs=12,
                lr=6e-4,
                seed=303,
                device=dev,
                log_fn=typer.echo,
            )
            koine_lm.save_safetensors(lm_path, fp16=True)
            typer.echo(f"  Koine LM best dev_loss={lm_loss:.4f}")
    else:
        typer.echo("5/6 Skipping standalone Koine LM (--no-train-lm).")

    # 6. Train 1..N forward ensemble seeds (+ optional pretraining & NT fine-tune)
    if not reuse_seeds:
        for old_f in target_dir.glob("forward-*.safetensors"):
            old_f.unlink()
    else:
        for old_f in target_dir.glob("forward-*.safetensors"):
            stem = old_f.stem.removeprefix("forward-")
            if stem.isdigit() and int(stem) >= seeds:
                old_f.unlink()

    mono_pool: list[str] = []
    if pretrain_epochs > 0:
        mono_pool.extend(bundle.all_clean_greek)
        for c_u in bundle.all_clean_coptic_units:
            s_txt, _ = serialize_coptic_morphemes(
                c_u.morphemes, ccl_lexicon=ccl_lexicon, domain_tag="<DOM_OT>"
            )
            mono_pool.append(s_txt)

    nt_clauses = [c for c in bundle.clause_aug if c.domain_tag == "<DOM_NT>"]
    forward_models: list[KoineSeq2SeqTransformer] = []
    for s_idx in range(seeds):
        seed_val = 42 + s_idx * 100
        out_w = target_dir / f"forward-{s_idx}.safetensors"
        if reuse_seeds and out_w.is_file():
            typer.echo(
                f"6/6 Reusing forward seed {s_idx + 1}/{seeds} ({out_w.name})..."
            )
            forward_models.append(
                KoineSeq2SeqTransformer.load_safetensors(out_w, cfg, device="cpu")
            )
            continue
        typer.echo(
            f"6/6 Training forward seed {s_idx + 1}/{seeds} (seed={seed_val})..."
        )
        init_m: KoineSeq2SeqTransformer | None = None
        if pretrain_epochs > 0:
            typer.echo(
                f"  Pretraining denoising ({pretrain_epochs} ep, "
                f"{len(mono_pool)} mono lines)..."
            )
            init_m = pretrain_denoising(
                model=KoineSeq2SeqTransformer(cfg),
                tokenizer=tokenizer,
                mono_texts=mono_pool,
                epochs=pretrain_epochs,
                lr=6e-4,
                seed=seed_val,
                device=dev,
                log_fn=typer.echo,
            )

        def _fwd_epoch(rng: random.Random) -> list[tuple[str, str]]:
            return make_forward_pairs_epoch(
                nt_train=bundle.nt_train,
                ot_train=bundle.ot_train,
                clause_aug=bundle.clause_aug,
                lex_aug=bundle.lex_aug,
                bt_pairs=bt_pairs,
                ccl_lexicon=ccl_lexicon,
                v1_drafts=v1_drafts if use_v1_draft else None,
                draft_vocab_pool=draft_vocab_pool,
                draft_drop_p=0.25,
                nt_repeat=2,
                loan_drop_p=0.05,
                morpheme_drop_p=0.03,
                rng=rng,
            )

        fwd_model, hist = train_seq2seq(
            config=cfg,
            tokenizer=tokenizer,
            train_pairs_fn=_fwd_epoch,
            dev_pairs=dev_fwd_pairs,
            epochs=epochs,
            lr=6e-4,
            seed=seed_val,
            device=dev,
            init_model=init_m,
            select_by_chrf=True,
            patience=8,
            log_fn=typer.echo,
        )
        typer.echo(
            f"  Joint phase best_ep={hist.best_epoch}, "
            f"dev_chrF={hist.best_dev_chrf:.4f}, dev_loss={hist.best_dev_loss:.4f}"
        )

        if ft_epochs > 0:
            typer.echo(f"  Fine-tuning on <NT> Gospel register ({ft_epochs} ep)...")

            def _nt_ft_epoch(rng: random.Random) -> list[tuple[str, str]]:
                return make_forward_pairs_epoch(
                    nt_train=bundle.nt_train,
                    ot_train=[],
                    clause_aug=nt_clauses,
                    lex_aug=bundle.lex_aug,
                    bt_pairs=[],
                    ccl_lexicon=ccl_lexicon,
                    v1_drafts=v1_drafts if use_v1_draft else None,
                    draft_vocab_pool=draft_vocab_pool,
                    draft_drop_p=0.25,
                    nt_repeat=1,
                    loan_drop_p=0.05,
                    morpheme_drop_p=0.02,
                    rng=rng,
                )

            ft_model, ft_hist = train_seq2seq(
                config=cfg,
                tokenizer=tokenizer,
                train_pairs_fn=_nt_ft_epoch,
                dev_pairs=dev_fwd_pairs,
                epochs=ft_epochs,
                lr=1.2e-4,
                seed=seed_val + 1,
                device=dev,
                init_model=fwd_model,
                select_by_chrf=True,
                patience=5,
                log_fn=typer.echo,
            )
            if ft_hist.best_dev_chrf >= hist.best_dev_chrf:
                fwd_model = ft_model
                typer.echo(
                    f"  Kept NT fine-tuned checkpoint "
                    f"(dev_chrF={ft_hist.best_dev_chrf:.4f})"
                )
            else:
                typer.echo(
                    f"  Kept joint checkpoint (joint {hist.best_dev_chrf:.4f} > "
                    f"ft {ft_hist.best_dev_chrf:.4f})"
                )

        out_w = target_dir / f"forward-{s_idx}.safetensors"
        fwd_model.save_safetensors(out_w, fp16=True)
        forward_models.append(
            KoineSeq2SeqTransformer.load_safetensors(out_w, cfg, device="cpu")
        )

    # Tune noisy-channel reranking weights strictly on 51-verse NT DEV split
    base_w = RerankWeights(
        w_fwd=1.0,
        w_rev=0.35 if reverse_model is not None else 0.0,
        w_lm=0.20 if koine_lm is not None else 0.0,
        w_loan=0.60,
        len_alpha=0.9,
        beam_size=8,
        use_v1_draft=use_v1_draft and v1_fast is not None,
        include_v1_candidate=include_v1,
    )
    dec = NeuralKoineDecoder(
        tokenizer=tokenizer,
        forward_models=forward_models,
        ccl_lexicon=ccl_lexicon,
        analyzer_inflector=analyzer_inflector,
        reverse_model=reverse_model.to("cpu") if reverse_model is not None else None,
        koine_lm=koine_lm.to("cpu") if koine_lm is not None else None,
        v1_decoder=v1_fast,
        weights=base_w,
        device="cpu",
    )
    typer.echo("Tuning noisy-channel rerank weights strictly on NT DEV (51 verses)...")
    tuned_w = tune_rerank_weights_on_dev(decoder=dec, dev_units=bundle.nt_dev_units)
    tuned_w.save_json(target_dir / "rerank-weights.json")

    write_training_provenance(
        out_dir=target_dir,
        model_id=target_dir.name,
        split_path=split_path,
        ccl_xml_path=ccl_xml,
        nt_train_ids=[ex.unit_id for ex in bundle.nt_train],
        ot_train_ids=[ex.unit_id for ex in bundle.ot_train],
        excluded_holdout_ids=holdout_ids,
        excluded_dev_ids=dev_ids,
        excluded_synoptic_ids=excluded_synoptic,
        extra={
            "seeds": len(forward_models),
            "epochs": epochs,
            "ft_epochs": ft_epochs,
            "pretrain_epochs": pretrain_epochs,
            "backtranslate": backtranslate,
            "bt_pair_count": len(bt_pairs),
            "train_lm": train_lm,
            "use_v1_draft": use_v1_draft,
            "include_v1": include_v1,
        },
    )

    sz_mb = sum(p.stat().st_size for p in target_dir.glob("*")) / (1024 * 1024)
    typer.echo(
        f"OK: froze model-v2 ({len(forward_models)} fwd seeds, "
        f"rev={reverse_model is not None}, lm={koine_lm is not None}, "
        f"total={sz_mb:.1f} MiB, weights={tuned_w}) to {target_dir}"
    )


@model_app.command("build-v3")
def build_model_v3(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Output model directory (default: data/mappings/model-v3)",
    ),
    seeds: int = typer.Option(
        3, "--seeds", min=1, max=5, help="Number of forward ensemble seeds"
    ),
    epochs: int = typer.Option(
        24, "--epochs", min=2, max=60, help="Joint training epochs per forward seed"
    ),
    ft_epochs: int = typer.Option(
        6, "--ft-epochs", min=0, max=30, help="NT Gospel fine-tuning epochs per seed"
    ),
    pretrain_epochs: int = typer.Option(
        6, "--pretrain-epochs", min=0, max=30, help="Span-denoising pretraining epochs"
    ),
    unaccented_target: bool = typer.Option(
        False,
        "--unaccented-target/--no-unaccented-target",
        help="Train on bare unaccented Greek + post-hoc KoineAccentRestorer (ablation)",
    ),
    use_ot_dp: bool = typer.Option(
        True,
        "--use-ot-dp/--no-use-ot-dp",
        help="Enable Sahidic OT x Swete versification normalization + chapter DP",
    ),
    use_gloss_bridge: bool = typer.Option(
        True,
        "--use-gloss-bridge/--no-use-gloss-bridge",
        help="Enable CCL native Egyptian x Strong's Greek gloss bridge & rare <GLOSS>",
    ),
    use_apophthegmata: bool = typer.Option(
        True,
        "--use-apophthegmata/--no-use-apophthegmata",
        help="Enable Apophthegmata Patrum denoising & cop/grc->eng multi-task pivot",
    ),
    use_patristic_bt: bool = typer.Option(
        True,
        "--use-patristic-bt/--no-use-patristic-bt",
        help="Enable Apostolic Fathers & First1KGreek back-translation + LM prose",
    ),
    backtranslate: bool = typer.Option(
        True,
        "--backtranslate/--no-backtranslate",
        help="Train reverse Koine->Greekified Coptic model and back-translate",
    ),
    train_lm: bool = typer.Option(
        True,
        "--train-lm/--no-train-lm",
        help="Train standalone unaccented Koine Greek causal LM for reranking",
    ),
    use_v1_draft: bool = typer.Option(
        True,
        "--use-v1-draft/--no-use-v1-draft",
        help="Condition forward encoder on leak-free bare model-v1 draft",
    ),
    include_v1: bool = typer.Option(
        False,
        "--include-v1/--no-include-v1",
        help="Include model-v1 output as an extra rerank candidate",
    ),
    reuse_aux: bool = typer.Option(
        False,
        "--reuse-aux/--no-reuse-aux",
        help="Reuse existing analyzer, accent restorer, tokenizer, rev, BT & LM",
    ),
    reuse_seeds: bool = typer.Option(
        False,
        "--reuse-seeds/--no-reuse-seeds",
        help="Reuse already trained forward-*.safetensors seeds when scaling ensemble",
    ),
    vocab_size: int = typer.Option(8000, "--vocab-size", min=512),
    d_model: int = typer.Option(384, "--d-model", min=64),
    layers: int = typer.Option(5, "--layers", min=2),
    device: str = typer.Option(
        "auto", "--device", help="Training device: auto|mps|cpu"
    ),
    exclude_synoptic_thomas: bool = typer.Option(
        False,
        "--exclude-synoptic-thomas/--include-synoptic-thomas",
        help="Optional ablation: exclude 63 NT Synoptic parallels",
    ),
) -> None:
    """Train strictly from-scratch Greekified + Patristic neural model-v3 (LFS)."""
    import dataclasses
    import json
    import random

    from zetodos_thomas.neural.accent_restorer import KoineAccentRestorer
    from zetodos_thomas.neural.corpus import (
        SPECIAL_USER_SYMBOLS_V3,
        build_parallel_corpus_bundle,
        make_sub_coptic_unit,
        normalize_greek_for_model,
        serialize_coptic_morphemes,
    )
    from zetodos_thomas.neural.decoder_v2 import (
        NeuralKoineDecoder,
        RerankWeights,
        load_analyzer_inflector,
        save_analyzer_shards,
        tune_rerank_weights_on_dev,
    )
    from zetodos_thomas.neural.tokenizer import BilingualTokenizer
    from zetodos_thomas.neural.trainer import (
        generate_backtranslations,
        make_forward_pairs_epoch,
        pretrain_denoising,
        resolve_device,
        train_causal_koine_lm,
        train_seq2seq,
    )
    from zetodos_thomas.neural.transformer import (
        KoineCausalLM,
        KoineCausalLMConfig,
        KoineSeq2SeqTransformer,
        Seq2SeqConfig,
    )

    root = require_repo_root(repo_root)
    downloads = root / "data" / "downloads"
    split_path = resolve_nt_split_path(root)
    v1_dir = root / MODEL_REL
    target_dir = out_dir or (root / MODEL_V3_REL)
    target_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    typer.echo(f"Training model-v3 on device={dev} -> {target_dir}")

    split = load_nt_split(split_path)
    holdout_ids = {
        uid
        for p in split.passages
        if p.role is PassageRole.HOLDOUT
        for uid in p.unit_ids
    }
    dev_ids = {
        uid for p in split.passages if p.role is PassageRole.DEV for uid in p.unit_ids
    }
    excluded_from_model = all_excluded_nt_ids(
        split, include_dev=True, exclude_synoptic_thomas=exclude_synoptic_thomas
    )
    excluded_synoptic = (
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS if exclude_synoptic_thomas else frozenset()
    )
    ccl_xml = downloads / "ccl-ddglc" / CCL_TEI_FILENAME
    ccl_manifest = root / CCL_MANIFEST_REL
    ccl_expected_sha = (
        load_manifest_sha256(ccl_manifest) if ccl_manifest.is_file() else None
    )

    # 1. Load leak-free CopticAnalyzer & pure CCL TEI, build Greek analyzer + restorer
    coptic_analyzer = CopticAnalyzer.load_shards(v1_dir)
    ccl_lexicon = CclLexicon.from_tei_xml(ccl_xml, expected_sha256=ccl_expected_sha)
    ccl_lexicon.save_json(target_dir / "ccl-loanwords.json")

    if not reuse_aux or not (target_dir / "greek-analyzer-0.json").is_file():
        typer.echo("1/7 Building leak-free Greek surface analyzer (full NT train)...")
        inflector = GreekInflector.build(
            morphgnt_dir=downloads / "morphgnt-sblgnt",
            exclude_unit_ids=excluded_from_model,
            ccl_lexicon=ccl_lexicon,
            swete_dir=downloads / "lxx-swete",
        )
        save_analyzer_shards(inflector, target_dir)
    else:
        typer.echo("1/7 Reusing existing Greek surface analyzer shards...")
    analyzer_inflector = load_analyzer_inflector(target_dir)

    accent_restorer: KoineAccentRestorer | None = None
    if unaccented_target:
        if not reuse_aux or not (target_dir / "accent-uni-0.json").is_file():
            typer.echo("2/7 Building deterministic polytonic KoineAccentRestorer...")
            accent_restorer = KoineAccentRestorer.build(
                morphgnt_dir=downloads / "morphgnt-sblgnt",
                exclude_unit_ids=excluded_from_model,
                swete_dir=downloads / "lxx-swete",
                ccl_lexicon=ccl_lexicon,
                downloads_dir=downloads,
                include_patristics=use_patristic_bt,
            )
            accent_restorer.save_shards(target_dir)
            typer.echo(
                f"  AccentRestorer: uni={accent_restorer.unigram_size}, "
                f"ctx={accent_restorer.bigram_size}"
            )
        else:
            typer.echo("2/7 Reusing existing KoineAccentRestorer shards...")
            accent_restorer = KoineAccentRestorer.load_shards(target_dir)
    else:
        typer.echo(
            "2/7 Using direct acute-only polytonic targets "
            "(skipping post-hoc KoineAccentRestorer)..."
        )
        for stale_acc in target_dir.glob("accent-*.json"):
            if stale_acc.is_file():
                stale_acc.unlink()

    # 3. Build unified Greekified parallel & non-biblical corpus bundle
    use_tgt_tag = use_apophthegmata
    tgt_grc_tag = "<TGT_GRC>" if use_tgt_tag else None
    mode_lbl = "unaccented" if unaccented_target else "polytonic"
    typer.echo(f"3/7 Building Greekified + {mode_lbl} model-v3 corpus bundle...")
    bundle = build_parallel_corpus_bundle(
        downloads_dir=downloads,
        split_path=split_path,
        coptic_analyzer=coptic_analyzer,
        ccl_lexicon=ccl_lexicon,
        exclude_synoptic_thomas=exclude_synoptic_thomas,
        recover_ot_versification=use_ot_dp,
        greekify=True,
        unaccented_target=unaccented_target,
        include_apophthegmata=use_apophthegmata,
        include_patristics=use_patristic_bt,
        include_gloss_bridge=use_gloss_bridge,
        target_tag=tgt_grc_tag,
    )
    native_gloss_hints: dict[str, str] | None = (
        bundle.native_gloss_hints if use_gloss_bridge else None
    )
    gloss_path = target_dir / "native-gloss-hints.json"
    if native_gloss_hints:
        gloss_path.write_text(
            json.dumps(
                dict(sorted(native_gloss_hints.items())),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    elif gloss_path.is_file():
        gloss_path.unlink()

    typer.echo(
        f"  Dataset: nt_train={len(bundle.nt_train)}, nt_dev={len(bundle.nt_dev)}, "
        f"ot_train={len(bundle.ot_train)}, clause_aug={len(bundle.clause_aug)}, "
        f"lex_aug={len(bundle.lex_aug)}, gloss_hints={len(bundle.native_gloss_hints)}, "
        f"swete_unaligned={len(bundle.swete_unaligned_greek)}, "
        f"patristic={len(bundle.patristic_greek)}, "
        f"apop={len(bundle.apop_coptic_units)} ({len(bundle.apop_eng_pairs)} eng)"
    )

    # 4. Train or reuse shared Greekified SentencePiece BPE tokenizer
    tok_model = target_dir / "tokenizer.model"
    if reuse_aux and tok_model.is_file():
        typer.echo("4/7 Reusing existing Greekified SentencePiece tokenizer...")
        tokenizer = BilingualTokenizer.load(tok_model)
    else:
        typer.echo(f"4/7 Training unified Greekified BPE tokenizer (V={vocab_size})...")
        tok_texts: list[str] = list(bundle.all_clean_greek)
        for ex in (*bundle.nt_train, *bundle.ot_train):
            s_txt, _ = serialize_coptic_morphemes(
                ex.coptic_morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag=ex.domain_tag,
                greekify=True,
                unaccented_hints=unaccented_target,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
            )
            tok_texts.append(s_txt)
        for ap_u in bundle.apop_coptic_units:
            s_txt, _ = serialize_coptic_morphemes(
                ap_u.morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag="<DOM_APOP>",
                greekify=True,
                unaccented_hints=unaccented_target,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
            )
            tok_texts.append(s_txt)
        for s_l, g_l in bundle.lex_aug:
            tok_texts.append(s_l)
            tok_texts.append(g_l)
        if use_apophthegmata:
            for _ap_u, eng in bundle.apop_eng_pairs:
                tok_texts.append(eng)
            for _gl, en_g in bundle.grc_eng_pairs[:1500]:
                tok_texts.append(en_g)
        tokenizer = BilingualTokenizer.train(
            texts=tok_texts,
            out_prefix=target_dir / "tokenizer",
            vocab_size=vocab_size,
            user_defined_symbols=SPECIAL_USER_SYMBOLS_V3,
        )

    cfg = Seq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=6 if d_model % 6 == 0 else 4,
        d_ff=d_model * 4,
        num_enc_layers=layers,
        num_dec_layers=layers,
        greekify_source=True,
        unaccented_target=unaccented_target,
        use_gloss_hints=bool(native_gloss_hints),
        use_target_lang_tag=use_tgt_tag,
    )
    (target_dir / "seq2seq-config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    # Leak-free model-v1 drafts for dual-source conditioning
    v1_drafts: dict[str, str] = {}
    draft_vocab_pool: list[str] = []
    v1_draft_cache = (
        downloads / ".cache-v1-drafts-v3.json"
        if unaccented_target
        else downloads / ".cache-v1-drafts-v3-poly.json"
    )
    v1_fast: KoineBeamDecoder | None = None
    if (use_v1_draft or include_v1) and (v1_dir / "decoder-weights.json").is_file():
        v1_fast = KoineBeamDecoder.from_model_dir(v1_dir)
        v1_fast.weights = dataclasses.replace(v1_fast.weights, beam_size=2)
    if use_v1_draft and v1_fast is not None:
        all_exs = (
            *bundle.nt_train,
            *bundle.ot_train,
            *bundle.clause_aug,
            *bundle.nt_dev,
        )
        if not unaccented_target and not v1_draft_cache.is_file():
            v2_cache = downloads / ".cache-v1-drafts.json"
            if v2_cache.is_file():
                v1_drafts = {
                    str(k): str(v)
                    for k, v in json.loads(v2_cache.read_text(encoding="utf-8")).items()
                }
        if v1_draft_cache.is_file():
            v1_drafts = {
                str(k): str(v)
                for k, v in json.loads(
                    v1_draft_cache.read_text(encoding="utf-8")
                ).items()
            }
        missing_exs = [ex for ex in all_exs if ex.unit_id not in v1_drafts]
        if missing_exs:
            typer.echo(
                f"  Precomputing leak-free {mode_lbl} model-v1 drafts for "
                f"{len(missing_exs)} examples (beam=2)..."
            )
            for ex in missing_exs:
                tmp_u = make_sub_coptic_unit(
                    ex.coptic_morphemes, unit_id=ex.unit_id, source_id="train"
                )
                v1_drafts[ex.unit_id] = normalize_greek_for_model(
                    v1_fast.decode_unit(tmp_u).greek_text,
                    unaccented=unaccented_target,
                )
            v1_draft_cache.write_text(
                json.dumps(v1_drafts, ensure_ascii=False), encoding="utf-8"
            )
        vocab_seen: set[str] = set()
        for g_txt in bundle.all_clean_greek[:12000]:
            for w in g_txt.split():
                if len(w) >= 3 and w not in vocab_seen:
                    vocab_seen.add(w)
                    draft_vocab_pool.append(w)
                    if len(draft_vocab_pool) >= 4000:
                        break
            if len(draft_vocab_pool) >= 4000:
                break

    dev_pure_pairs: list[tuple[str, str]] = []
    dev_fwd_pairs: list[tuple[str, str]] = []
    for ex in bundle.nt_dev:
        s_pure, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes,
            ccl_lexicon=ccl_lexicon,
            domain_tag="<DOM_NT>",
            greekify=True,
            unaccented_hints=unaccented_target,
            native_gloss_hints=native_gloss_hints,
        )
        dev_pure_pairs.append((s_pure, ex.greek_target))
        s_fwd, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes,
            ccl_lexicon=ccl_lexicon,
            domain_tag="<DOM_NT>",
            v1_draft=v1_drafts.get(ex.unit_id) if use_v1_draft else None,
            greekify=True,
            unaccented_hints=unaccented_target,
            native_gloss_hints=native_gloss_hints,
            target_tag=tgt_grc_tag,
        )
        dev_fwd_pairs.append((s_fwd, ex.greek_target))

    # 5. Reverse Koine -> Greekified Coptic model & Swete + Patristic BT
    bt_cache = downloads / (
        f".cache-bt-v3-u{int(unaccented_target)}-pat{int(use_patristic_bt)}"
        f"-gl{int(use_gloss_bridge)}-ap{int(use_apophthegmata)}.json"
    )
    rev_path = target_dir / "reverse.safetensors"
    swete_bt_pairs: list[tuple[str, str]] = []
    patristic_bt_pairs: list[tuple[str, str]] = []
    reverse_model: KoineSeq2SeqTransformer | None = None
    if backtranslate:
        if reuse_aux and rev_path.is_file() and bt_cache.is_file():
            typer.echo("5/7 Reusing reverse model & cached back-translations...")
            reverse_model = KoineSeq2SeqTransformer.load_safetensors(
                rev_path, cfg, device="cpu"
            )
            cached_bt = json.loads(bt_cache.read_text(encoding="utf-8"))
            swete_bt_pairs = [(str(a), str(b)) for a, b in cached_bt.get("swete", [])]
            patristic_bt_pairs = [
                (str(a), str(b)) for a, b in cached_bt.get("patristic", [])
            ]
        else:
            typer.echo(
                f"5/7 Training reverse {mode_lbl}-Koine -> Greekified-Coptic model..."
            )
            dev_rev_pairs = [(f"<REV_COPTIC> {g}", s) for s, g in dev_pure_pairs]

            def _rev_epoch(rng: random.Random) -> list[tuple[str, str]]:
                fwd = make_forward_pairs_epoch(
                    nt_train=bundle.nt_train,
                    ot_train=bundle.ot_train,
                    clause_aug=bundle.clause_aug,
                    lex_aug=[],
                    bt_pairs=[],
                    ccl_lexicon=ccl_lexicon,
                    nt_repeat=2,
                    loan_drop_p=0.0,
                    morpheme_drop_p=0.0,
                    rng=rng,
                    greekify=True,
                    unaccented_hints=unaccented_target,
                    native_gloss_hints=native_gloss_hints,
                    target_tag=None,
                )
                return [(f"<REV_COPTIC> {g}", s) for s, g in fwd]

            reverse_model, rev_hist = train_seq2seq(
                config=cfg,
                tokenizer=tokenizer,
                train_pairs_fn=_rev_epoch,
                dev_pairs=dev_rev_pairs,
                epochs=max(12, epochs // 2),
                lr=6e-4,
                seed=777,
                device=dev,
                select_by_chrf=False,
                patience=5,
                log_fn=typer.echo,
            )
            typer.echo(
                f"  Reverse best_ep={rev_hist.best_epoch}, "
                f"dev_loss={rev_hist.best_dev_loss:.4f}"
            )
            reverse_model.save_safetensors(rev_path, fp16=True)

            typer.echo(
                f"  Back-translating {len(bundle.swete_unaligned_greek)} Swete LXX..."
            )
            swete_bt_pairs = generate_backtranslations(
                reverse_model=reverse_model,
                tokenizer=tokenizer,
                greek_texts=bundle.swete_unaligned_greek,
                device=dev,
                domain_tag="<DOM_BT>",
                target_tag=tgt_grc_tag,
            )
            if use_patristic_bt and bundle.patristic_greek:
                typer.echo(
                    f"  Back-translating {len(bundle.patristic_greek)} "
                    f"Apostolic Fathers / Epictetus / Philo sentences..."
                )
                patristic_bt_pairs = generate_backtranslations(
                    reverse_model=reverse_model,
                    tokenizer=tokenizer,
                    greek_texts=bundle.patristic_greek,
                    device=dev,
                    domain_tag="<DOM_PAT>",
                    target_tag=tgt_grc_tag,
                )
            bt_cache.write_text(
                json.dumps(
                    {"swete": swete_bt_pairs, "patristic": patristic_bt_pairs},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            typer.echo(
                f"  Generated BT pairs: swete={len(swete_bt_pairs)}, "
                f"patristic={len(patristic_bt_pairs)}"
            )
    else:
        typer.echo("5/7 Skipping back-translation (--no-backtranslate).")

    all_bt_pairs = [*swete_bt_pairs, *patristic_bt_pairs]

    # 6. Standalone 4-layer Koine Greek Causal LM (NT + Swete + Patristics)
    lm_path = target_dir / "koine-lm.safetensors"
    lm_cfg_path = target_dir / "koine-lm-config.json"
    koine_lm: KoineCausalLM | None = None
    if train_lm:
        lm_cfg = KoineCausalLMConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            n_heads=6 if d_model % 6 == 0 else 4,
            d_ff=d_model * 4,
            num_layers=4,
            unaccented_target=unaccented_target,
        )
        if reuse_aux and lm_path.is_file() and lm_cfg_path.is_file():
            typer.echo(f"6/7 Reusing standalone {mode_lbl} Koine Causal LM...")
            koine_lm = KoineCausalLM.load_safetensors(lm_path, lm_cfg, device="cpu")
        else:
            typer.echo(
                f"6/7 Training standalone {mode_lbl} Koine Causal LM "
                f"({len(bundle.all_clean_greek)} clean Koine sentences)..."
            )
            lm_cfg_path.write_text(
                json.dumps(lm_cfg.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            koine_lm, lm_loss = train_causal_koine_lm(
                config=lm_cfg,
                tokenizer=tokenizer,
                train_greek_texts=bundle.all_clean_greek,
                dev_greek_texts=[ex.greek_target for ex in bundle.nt_dev],
                epochs=12,
                lr=6e-4,
                seed=303,
                device=dev,
                log_fn=typer.echo,
            )
            koine_lm.save_safetensors(lm_path, fp16=True)
            typer.echo(f"  Koine LM best dev_loss={lm_loss:.4f}")
    else:
        typer.echo("6/7 Skipping standalone Koine LM (--no-train-lm).")

    # 7. Denoising pretraining + 1..N forward ensemble seeds + Gospel fine-tuning
    if not reuse_seeds:
        for old_f in target_dir.glob("forward-*.safetensors"):
            old_f.unlink()
    else:
        for old_f in target_dir.glob("forward-*.safetensors"):
            stem = old_f.stem.removeprefix("forward-")
            if stem.isdigit() and int(stem) >= seeds:
                old_f.unlink()

    mono_pool: list[str] = []
    if pretrain_epochs > 0:
        mono_pool.extend(bundle.all_clean_greek)
        for c_u in bundle.all_clean_coptic_units:
            dom = (
                "<DOM_APOP>"
                if c_u.source_id == "scriptorium-apophthegmata"
                else "<DOM_OT>"
            )
            s_txt, _ = serialize_coptic_morphemes(
                c_u.morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag=dom,
                greekify=True,
                unaccented_hints=unaccented_target,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
            )
            mono_pool.append(s_txt)

    # Pre-serialize Apophthegmata cop -> eng pivot pairs once
    apop_eng_serialized: list[tuple[str, str]] = []
    if use_apophthegmata:
        for ap_u, eng in bundle.apop_eng_pairs:
            s_ap, _ = serialize_coptic_morphemes(
                ap_u.morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag="<DOM_APOP>",
                greekify=True,
                unaccented_hints=unaccented_target,
                native_gloss_hints=native_gloss_hints,
                target_tag="<TGT_ENG>",
            )
            apop_eng_serialized.append((s_ap, eng))
    grc_eng_serialized: list[tuple[str, str]] = (
        [(f"<DOM_LEX> <TGT_ENG> {g}", en) for g, en in bundle.grc_eng_pairs]
        if use_apophthegmata
        else []
    )

    nt_clauses = [c for c in bundle.clause_aug if c.domain_tag == "<DOM_NT>"]
    forward_models: list[KoineSeq2SeqTransformer] = []
    for s_idx in range(seeds):
        seed_val = 42 + s_idx * 100
        out_w = target_dir / f"forward-{s_idx}.safetensors"
        if reuse_seeds and out_w.is_file():
            typer.echo(
                f"7/7 Reusing forward seed {s_idx + 1}/{seeds} ({out_w.name})..."
            )
            forward_models.append(
                KoineSeq2SeqTransformer.load_safetensors(out_w, cfg, device="cpu")
            )
            continue
        typer.echo(
            f"7/7 Training forward seed {s_idx + 1}/{seeds} (seed={seed_val})..."
        )
        init_m: KoineSeq2SeqTransformer | None = None
        if pretrain_epochs > 0:
            typer.echo(
                f"  Pretraining denoising ({pretrain_epochs} ep, "
                f"{len(mono_pool)} Greekified+Koine mono lines)..."
            )
            init_m = pretrain_denoising(
                model=KoineSeq2SeqTransformer(cfg),
                tokenizer=tokenizer,
                mono_texts=mono_pool,
                epochs=pretrain_epochs,
                lr=6e-4,
                seed=seed_val,
                device=dev,
                log_fn=typer.echo,
            )

        def _fwd_epoch_v3(rng: random.Random) -> list[tuple[str, str]]:
            mt_pairs: list[tuple[str, str]] = []
            if apop_eng_serialized:
                mt_pairs.extend(apop_eng_serialized)
                mt_pairs.extend(apop_eng_serialized)
            if grc_eng_serialized:
                k_ge = min(len(grc_eng_serialized), 1500)
                mt_pairs.extend(rng.sample(grc_eng_serialized, k_ge))
            return make_forward_pairs_epoch(
                nt_train=bundle.nt_train,
                ot_train=bundle.ot_train,
                clause_aug=bundle.clause_aug,
                lex_aug=bundle.lex_aug,
                bt_pairs=all_bt_pairs,
                ccl_lexicon=ccl_lexicon,
                v1_drafts=v1_drafts if use_v1_draft else None,
                draft_vocab_pool=draft_vocab_pool,
                draft_drop_p=0.25,
                nt_repeat=2,
                loan_drop_p=0.05,
                morpheme_drop_p=0.03,
                rng=rng,
                greekify=True,
                unaccented_hints=unaccented_target,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
                multitask_pairs=mt_pairs,
            )

        fwd_model, hist = train_seq2seq(
            config=cfg,
            tokenizer=tokenizer,
            train_pairs_fn=_fwd_epoch_v3,
            dev_pairs=dev_fwd_pairs,
            epochs=epochs,
            lr=6e-4,
            seed=seed_val,
            device=dev,
            init_model=init_m,
            select_by_chrf=True,
            patience=8,
            log_fn=typer.echo,
        )
        typer.echo(
            f"  Joint phase best_ep={hist.best_epoch}, "
            f"dev_chrF={hist.best_dev_chrf:.4f}, dev_loss={hist.best_dev_loss:.4f}"
        )

        if ft_epochs > 0:
            typer.echo(
                f"  Fine-tuning on <NT> + patristic retention ({ft_epochs} ep)..."
            )

            def _nt_ft_epoch_v3(rng: random.Random) -> list[tuple[str, str]]:
                pat_sample = (
                    rng.sample(patristic_bt_pairs, min(len(patristic_bt_pairs), 1000))
                    if patristic_bt_pairs
                    else []
                )
                return make_forward_pairs_epoch(
                    nt_train=bundle.nt_train,
                    ot_train=[],
                    clause_aug=nt_clauses,
                    lex_aug=bundle.lex_aug,
                    bt_pairs=pat_sample,
                    ccl_lexicon=ccl_lexicon,
                    v1_drafts=v1_drafts if use_v1_draft else None,
                    draft_vocab_pool=draft_vocab_pool,
                    draft_drop_p=0.25,
                    nt_repeat=1,
                    loan_drop_p=0.05,
                    morpheme_drop_p=0.02,
                    rng=rng,
                    greekify=True,
                    unaccented_hints=unaccented_target,
                    native_gloss_hints=native_gloss_hints,
                    target_tag=tgt_grc_tag,
                )

            ft_model, ft_hist = train_seq2seq(
                config=cfg,
                tokenizer=tokenizer,
                train_pairs_fn=_nt_ft_epoch_v3,
                dev_pairs=dev_fwd_pairs,
                epochs=ft_epochs,
                lr=1.0e-4,
                seed=seed_val + 1,
                device=dev,
                init_model=fwd_model,
                select_by_chrf=True,
                patience=5,
                log_fn=typer.echo,
            )
            if ft_hist.best_dev_chrf >= hist.best_dev_chrf:
                fwd_model = ft_model
                typer.echo(
                    f"  Kept NT fine-tuned checkpoint "
                    f"(dev_chrF={ft_hist.best_dev_chrf:.4f})"
                )
            else:
                typer.echo(
                    f"  Kept joint checkpoint (joint {hist.best_dev_chrf:.4f} > "
                    f"ft {ft_hist.best_dev_chrf:.4f})"
                )

        fwd_model.save_safetensors(out_w, fp16=True)
        forward_models.append(
            KoineSeq2SeqTransformer.load_safetensors(out_w, cfg, device="cpu")
        )

    # Tune noisy-channel reranking weights strictly on 51-verse NT DEV split
    base_w = RerankWeights(
        w_fwd=1.0,
        w_rev=0.35 if reverse_model is not None else 0.0,
        w_lm=0.20 if koine_lm is not None else 0.0,
        w_loan=0.60,
        len_alpha=0.9,
        beam_size=8,
        use_v1_draft=use_v1_draft and v1_fast is not None,
        include_v1_candidate=include_v1,
    )
    dec = NeuralKoineDecoder(
        tokenizer=tokenizer,
        forward_models=forward_models,
        ccl_lexicon=ccl_lexicon,
        analyzer_inflector=analyzer_inflector,
        reverse_model=reverse_model.to("cpu") if reverse_model is not None else None,
        koine_lm=koine_lm.to("cpu") if koine_lm is not None else None,
        v1_decoder=v1_fast,
        weights=base_w,
        device="cpu",
        accent_restorer=accent_restorer,
        native_gloss_hints=native_gloss_hints,
        greekify_source=True,
        unaccented_target=unaccented_target,
        use_target_lang_tag=use_tgt_tag,
    )
    typer.echo("Tuning noisy-channel rerank weights strictly on NT DEV (51 verses)...")
    tuned_w = tune_rerank_weights_on_dev(decoder=dec, dev_units=bundle.nt_dev_units)
    tuned_w.save_json(target_dir / "rerank-weights.json")

    write_training_provenance(
        out_dir=target_dir,
        model_id=target_dir.name,
        split_path=split_path,
        ccl_xml_path=ccl_xml,
        nt_train_ids=[ex.unit_id for ex in bundle.nt_train],
        ot_train_ids=[ex.unit_id for ex in bundle.ot_train],
        excluded_holdout_ids=holdout_ids,
        excluded_dev_ids=dev_ids,
        excluded_synoptic_ids=excluded_synoptic,
        extra={
            "seeds": len(forward_models),
            "epochs": epochs,
            "ft_epochs": ft_epochs,
            "pretrain_epochs": pretrain_epochs,
            "greekify_source": True,
            "unaccented_target": unaccented_target,
            "use_ot_dp": use_ot_dp,
            "use_gloss_bridge": use_gloss_bridge,
            "native_gloss_hint_count": len(bundle.native_gloss_hints),
            "use_apophthegmata": use_apophthegmata,
            "apop_unit_count": len(bundle.apop_coptic_units),
            "use_patristic_bt": use_patristic_bt,
            "swete_bt_count": len(swete_bt_pairs),
            "patristic_bt_count": len(patristic_bt_pairs),
            "train_lm": train_lm,
            "use_v1_draft": use_v1_draft,
            "include_v1": include_v1,
        },
    )

    sz_mb = sum(p.stat().st_size for p in target_dir.glob("*")) / (1024 * 1024)
    typer.echo(
        f"OK: froze model-v3 ({len(forward_models)} fwd seeds, "
        f"rev={reverse_model is not None}, lm={koine_lm is not None}, "
        f"total={sz_mb:.1f} MiB, weights={tuned_w}) to {target_dir}"
    )


@model_app.command("build-v4")
def build_model_v4(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Output model directory (default: data/mappings/model-v4)",
    ),
    seeds: int = typer.Option(
        3, "--seeds", min=1, max=5, help="Number of forward ensemble seeds"
    ),
    epochs: int = typer.Option(
        22, "--epochs", min=2, max=60, help="Joint training epochs per forward seed"
    ),
    ft_epochs: int = typer.Option(
        6, "--ft-epochs", min=0, max=30, help="Balanced NT+OT fine-tuning epochs"
    ),
    pretrain_epochs: int = typer.Option(
        6, "--pretrain-epochs", min=0, max=30, help="Span-denoising pretraining epochs"
    ),
    coverage_loss_weight: float = typer.Option(
        0.50, "--cov-loss-weight", min=0.0, help="Cross-attention coverage loss weight"
    ),
    align_loss_weight: float = typer.Option(
        0.30,
        "--align-loss-weight",
        min=0.0,
        help="Guided lexical alignment loss weight",
    ),
    use_patristic_bt: bool = typer.Option(
        True,
        "--use-patristic-bt/--no-use-patristic-bt",
        help="Enable faithfulness-filtered & leak-checked patristic BT + LM",
    ),
    use_apop_denoise: bool = typer.Option(
        True,
        "--use-apop-denoise/--no-use-apop-denoise",
        help="Include Sahidic Apophthegmata Patrum in Coptic denoising pretraining",
    ),
    use_eng_multitask: bool = typer.Option(
        False,
        "--use-eng-multitask/--no-use-eng-multitask",
        help="Optional ablation: include cop/grc->eng multi-task pairs",
    ),
    use_v1_draft: bool = typer.Option(
        True,
        "--use-v1-draft/--no-use-v1-draft",
        help="Condition forward encoder on leak-free VSTAT-aware v1 draft",
    ),
    reuse_aux: bool = typer.Option(
        False,
        "--reuse-aux/--no-reuse-aux",
        help="Reuse existing analyzer, aligner, tokenizer, rev, BT & LM",
    ),
    reuse_seeds: bool = typer.Option(
        False,
        "--reuse-seeds/--no-reuse-seeds",
        help="Reuse already trained forward-*.safetensors seeds",
    ),
    vocab_size: int = typer.Option(8000, "--vocab-size", min=512),
    d_model: int = typer.Option(384, "--d-model", min=64),
    layers: int = typer.Option(5, "--layers", min=2),
    device: str = typer.Option(
        "auto", "--device", help="Training device: auto|mps|cpu"
    ),
    exclude_synoptic_thomas: bool = typer.Option(
        False,
        "--exclude-synoptic-thomas/--include-synoptic-thomas",
        help="Optional ablation: exclude 63 NT Synoptic parallels",
    ),
) -> None:
    """Train faithfulness-first zero-leakage model-v4 (coverage + constrained beam)."""
    import dataclasses
    import json
    import random
    from collections.abc import Sequence

    from zetodos_thomas.neural.corpus import (
        SPECIAL_USER_SYMBOLS_V3,
        build_parallel_corpus_bundle,
        make_sub_coptic_unit,
        normalize_greek_for_model,
        normalize_greek_target,
        serialize_coptic_morphemes,
    )
    from zetodos_thomas.neural.decoder_v2 import (
        NeuralKoineDecoder,
        RerankWeights,
        _seq_lcs_len,
        _strip_bare,
        load_analyzer_inflector,
        save_analyzer_shards,
        tune_rerank_weights_on_dev,
    )
    from zetodos_thomas.neural.tokenizer import BilingualTokenizer
    from zetodos_thomas.neural.trainer import (
        generate_backtranslations,
        make_faithful_forward_pairs_epoch,
        make_forward_pairs_epoch,
        pretrain_denoising,
        resolve_device,
        train_causal_koine_lm,
        train_seq2seq,
        train_seq2seq_faithful,
    )
    from zetodos_thomas.neural.transformer import (
        KoineCausalLM,
        KoineCausalLMConfig,
        KoineSeq2SeqTransformer,
        Seq2SeqConfig,
    )

    root = require_repo_root(repo_root)
    downloads = root / "data" / "downloads"
    split_path = resolve_nt_split_path(root)
    v1_dir = root / MODEL_REL
    target_dir = out_dir or (root / MODEL_V4_REL)
    target_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    typer.echo(f"Training faithfulness-first model-v4 on device={dev} -> {target_dir}")

    split = load_nt_split(split_path)
    holdout_ids = {
        uid
        for p in split.passages
        if p.role is PassageRole.HOLDOUT
        for uid in p.unit_ids
    }
    dev_ids = {
        uid for p in split.passages if p.role is PassageRole.DEV for uid in p.unit_ids
    }
    excluded_from_model = all_excluded_nt_ids(
        split, include_dev=True, exclude_synoptic_thomas=exclude_synoptic_thomas
    )
    excluded_synoptic = (
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS if exclude_synoptic_thomas else frozenset()
    )
    ccl_xml = downloads / "ccl-ddglc" / CCL_TEI_FILENAME
    ccl_manifest = root / CCL_MANIFEST_REL
    ccl_expected_sha = (
        load_manifest_sha256(ccl_manifest) if ccl_manifest.is_file() else None
    )
    ot_zip = downloads / "sahidic-ot" / SAHIDIC_OT_ZIP_FILENAME

    # 1. Train & freeze upgraded CopticAnalyzer, CCL, GreekInflector & VSTAT eflomal
    ccl_lexicon = CclLexicon.from_tei_xml(ccl_xml, expected_sha256=ccl_expected_sha)
    ccl_lexicon.save_json(target_dir / "ccl-loanwords.json")

    if not reuse_aux or not (target_dir / "coptic-meta.json").is_file():
        typer.echo("1/7 Training upgraded leak-free CopticAnalyzer (OT + CCL)...")
        coptic_analyzer = CopticAnalyzer.train(ot_zip=ot_zip, ccl_xml=ccl_xml)
        coptic_analyzer.save_shards(target_dir)
    else:
        typer.echo("1/7 Reusing existing model-v4 CopticAnalyzer shards...")
        coptic_analyzer = CopticAnalyzer.load_shards(target_dir)

    if (
        not reuse_aux
        or not (target_dir / "greek-analyzer-0.json").is_file()
        or not (target_dir / "greek-paradigms-0.json").is_file()
    ):
        typer.echo("2/7 Building leak-free GreekInflector (analyzer + paradigms)...")
        inflector_full = GreekInflector.build(
            morphgnt_dir=downloads / "morphgnt-sblgnt",
            exclude_unit_ids=excluded_from_model,
            ccl_lexicon=ccl_lexicon,
            swete_dir=downloads / "lxx-swete",
        )
        save_analyzer_shards(inflector_full, target_dir)
        inflector_full.save_shards(target_dir)
    else:
        typer.echo("2/7 Reusing existing GreekInflector shards...")
    analyzer_inflector = load_analyzer_inflector(target_dir)
    paradigm_inflector = GreekInflector.load_shards(target_dir)

    if not reuse_aux or not any(target_dir.glob("lemma-align-*.json")):
        typer.echo("  Building VSTAT-separated leak-free LexicalTranslationModel...")
        lexical_model = LexicalTranslationModel.build_full(
            downloads_dir=downloads,
            split_path=split_path,
            analyzer=coptic_analyzer,
            inflector=analyzer_inflector,
            ccl_lexicon=ccl_lexicon,
            nt_weight=3.0,
            repo_root=root,
            exclude_synoptic_thomas=exclude_synoptic_thomas,
        )
        lexical_model.save_shards(target_dir)
    else:
        typer.echo("  Reusing VSTAT-separated LexicalTranslationModel shards...")
        lexical_model = LexicalTranslationModel.load_shards(target_dir)

    # 3. Build faithfulness-first bundle (PREPs, POS glosses, 72m windows, patristics)
    tgt_grc_tag = "<TGT_GRC>" if use_eng_multitask else None
    typer.echo(
        "3/7 Building faithfulness-first model-v4 corpus bundle "
        "(VSTAT glosses + canonical PREPs + multi-verse windows)..."
    )
    bundle = build_parallel_corpus_bundle(
        downloads_dir=downloads,
        split_path=split_path,
        coptic_analyzer=coptic_analyzer,
        ccl_lexicon=ccl_lexicon,
        exclude_synoptic_thomas=exclude_synoptic_thomas,
        recover_ot_versification=True,
        greekify=True,
        unaccented_target=False,
        include_apophthegmata=use_apop_denoise or use_eng_multitask,
        include_patristics=use_patristic_bt,
        include_gloss_bridge=True,
        target_tag=tgt_grc_tag,
        lexical_model=lexical_model,
        include_multiverse_windows=True,
    )
    native_gloss_hints = bundle.native_gloss_hints
    (target_dir / "native-gloss-hints.json").write_text(
        json.dumps(
            dict(sorted(native_gloss_hints.items())),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(
        f"  Dataset: nt_train={len(bundle.nt_train)}, nt_dev={len(bundle.nt_dev)}, "
        f"ot_train={len(bundle.ot_train)}, clause_aug={len(bundle.clause_aug)}, "
        f"multiverse={len(bundle.multiverse_aug)}, lex_aug={len(bundle.lex_aug)}, "
        f"gloss_hints={len(native_gloss_hints)}, "
        f"swete_unaligned={len(bundle.swete_unaligned_greek)}, "
        f"patristic={len(bundle.patristic_greek)}, apop={len(bundle.apop_coptic_units)}"
    )

    # 4. Train or reuse shared Greekified SentencePiece BPE tokenizer
    tok_model = target_dir / "tokenizer.model"
    if reuse_aux and tok_model.is_file():
        typer.echo("4/7 Reusing existing Greekified SentencePiece tokenizer...")
        tokenizer = BilingualTokenizer.load(tok_model)
    else:
        typer.echo(f"4/7 Training unified Greekified BPE tokenizer (V={vocab_size})...")
        tok_texts: list[str] = list(bundle.all_clean_greek)
        for ex in (*bundle.nt_train, *bundle.ot_train):
            s_txt, _ = serialize_coptic_morphemes(
                ex.coptic_morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag=ex.domain_tag,
                greekify=True,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
                canonicalize_preps=True,
            )
            tok_texts.append(s_txt)
        for ap_u in bundle.apop_coptic_units:
            s_txt, _ = serialize_coptic_morphemes(
                ap_u.morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag="<DOM_APOP>",
                greekify=True,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
                canonicalize_preps=True,
            )
            tok_texts.append(s_txt)
        for s_l, g_l in bundle.lex_aug:
            tok_texts.append(s_l)
            tok_texts.append(g_l)
        tokenizer = BilingualTokenizer.train(
            texts=tok_texts,
            out_prefix=target_dir / "tokenizer",
            vocab_size=vocab_size,
            user_defined_symbols=SPECIAL_USER_SYMBOLS_V3,
        )

    cfg = Seq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=6 if d_model % 6 == 0 else 4,
        d_ff=d_model * 4,
        num_enc_layers=layers,
        num_dec_layers=layers,
        dropout=0.25,
        attn_dropout=0.10,
        greekify_source=True,
        unaccented_target=False,
        use_gloss_hints=True,
        use_target_lang_tag=use_eng_multitask,
        canonicalize_preps=True,
    )
    (target_dir / "seq2seq-config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    # Precompute leak-free VSTAT-aware v1 drafts
    v1_drafts: dict[str, str] = {}
    draft_vocab_pool: list[str] = []
    v1_draft_cache = downloads / ".cache-v1-drafts-v4.json"
    v1_fast: KoineBeamDecoder | None = None
    if use_v1_draft and (v1_dir / "decoder-weights.json").is_file():
        v1_fast = KoineBeamDecoder.from_model_dir(v1_dir)
        v1_fast.lexicon = lexical_model
        v1_fast.inflector = paradigm_inflector
        v1_fast.weights = dataclasses.replace(v1_fast.weights, beam_size=2)
    if use_v1_draft and v1_fast is not None:
        all_exs = (
            *bundle.nt_train,
            *bundle.ot_train,
            *bundle.clause_aug,
            *bundle.multiverse_aug,
            *bundle.nt_dev,
        )
        if v1_draft_cache.is_file():
            v1_drafts = {
                str(k): str(v)
                for k, v in json.loads(
                    v1_draft_cache.read_text(encoding="utf-8")
                ).items()
            }
        missing_exs = [ex for ex in all_exs if ex.unit_id not in v1_drafts]
        if missing_exs:
            typer.echo(
                f"  Precomputing leak-free VSTAT-aware v1 drafts for "
                f"{len(missing_exs)} examples (beam=2)..."
            )
            for ex in missing_exs:
                tmp_u = make_sub_coptic_unit(
                    ex.coptic_morphemes, unit_id=ex.unit_id, source_id="train"
                )
                v1_drafts[ex.unit_id] = normalize_greek_for_model(
                    v1_fast.decode_unit(tmp_u).greek_text,
                    unaccented=False,
                )
            v1_draft_cache.write_text(
                json.dumps(v1_drafts, ensure_ascii=False), encoding="utf-8"
            )
        vocab_seen: set[str] = set()
        for g_txt in bundle.all_clean_greek[:12000]:
            for w in g_txt.split():
                if len(w) >= 3 and w not in vocab_seen:
                    vocab_seen.add(w)
                    draft_vocab_pool.append(w)
                    if len(draft_vocab_pool) >= 4000:
                        break
            if len(draft_vocab_pool) >= 4000:
                break

    dev_pure_pairs: list[tuple[str, str]] = []
    dev_fwd_pairs: list[tuple[str, str]] = []
    for ex in bundle.nt_dev:
        s_pure, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes,
            ccl_lexicon=ccl_lexicon,
            domain_tag="<DOM_NT>",
            greekify=True,
            native_gloss_hints=native_gloss_hints,
            canonicalize_preps=True,
        )
        dev_pure_pairs.append((s_pure, ex.greek_target))
        s_fwd, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes,
            ccl_lexicon=ccl_lexicon,
            domain_tag="<DOM_NT>",
            v1_draft=v1_drafts.get(ex.unit_id) if use_v1_draft else None,
            greekify=True,
            native_gloss_hints=native_gloss_hints,
            target_tag=tgt_grc_tag,
            canonicalize_preps=True,
        )
        dev_fwd_pairs.append((s_fwd, ex.greek_target))

    # Build faithfulness-aware NT DEV scorer (ordered F1 + recall + length fidelity)
    _content_gpos = {"N-", "V-", "A-", "D-"}
    dev_gold_lems = [
        [_strip_bare(t.lemma) for t in g_u.tokens if t.morph.pos in _content_gpos]
        for _, g_u in bundle.nt_dev_units
    ]
    dev_gold_surfs = [
        [
            ns
            for t in g_u.tokens
            if t.morph.pos != "X-"
            and (ns := normalize_greek_target(t.normalized or t.surface).lower())
        ]
        for _, g_u in bundle.nt_dev_units
    ]

    def _faithful_dev_score(hyps: Sequence[str]) -> float:
        lm_m = lm_p = lm_g = sf_m = sf_p = sf_g = 0
        for hyp, g_l, g_s in zip(hyps, dev_gold_lems, dev_gold_surfs, strict=False):
            p_l: list[str] = []
            p_s = normalize_greek_target(hyp).lower().split() if hyp else []
            for w in hyp.split():
                lem, mf, _ = analyzer_inflector.analyze_surface(w)[0]
                if mf.pos in _content_gpos:
                    p_l.append(_strip_bare(lem))
            lm_m += _seq_lcs_len(p_l, g_l)
            lm_p += len(p_l)
            lm_g += len(g_l)
            sf_m += _seq_lcs_len(p_s, g_s)
            sf_p += len(p_s)
            sf_g += len(g_s)
        lf1 = 2 * lm_m / max(lm_p + lm_g, 1)
        lrec = lm_m / max(lm_g, 1)
        sf1 = 2 * sf_m / max(sf_p + sf_g, 1)
        ratio = lm_p / max(lm_g, 1)
        len_fid = max(0.0, 1.0 - abs(1.0 - ratio))
        return 0.45 * lf1 + 0.30 * sf1 + 0.15 * lrec + 0.10 * len_fid

    # 5. Clean Reverse Koine -> Greekified Coptic model & Faithfulness-Filtered BT
    bt_cache = downloads / f".cache-bt-v4-pat{int(use_patristic_bt)}.json"
    rev_path = target_dir / "reverse.safetensors"
    swete_bt_pairs: list[tuple[str, str]] = []
    patristic_bt_pairs: list[tuple[str, str]] = []
    if reuse_aux and rev_path.is_file() and bt_cache.is_file():
        typer.echo("5/7 Reusing clean reverse model & faithfulness-filtered BT...")
        reverse_model = KoineSeq2SeqTransformer.load_safetensors(
            rev_path, cfg, device="cpu"
        )
        cached_bt = json.loads(bt_cache.read_text(encoding="utf-8"))
        swete_bt_pairs = [(str(a), str(b)) for a, b in cached_bt.get("swete", [])]
        patristic_bt_pairs = [
            (str(a), str(b)) for a, b in cached_bt.get("patristic", [])
        ]
    else:
        typer.echo("5/7 Training clean reverse Koine -> Greekified Coptic model...")
        dev_rev_pairs = [(f"<REV_COPTIC> {g}", s) for s, g in dev_pure_pairs]

        def _rev_epoch_v4(rng: random.Random) -> list[tuple[str, str]]:
            fwd = make_forward_pairs_epoch(
                nt_train=bundle.nt_train,
                ot_train=bundle.ot_train,
                clause_aug=bundle.clause_aug,
                lex_aug=[],
                bt_pairs=[],
                ccl_lexicon=ccl_lexicon,
                nt_repeat=2,
                loan_drop_p=0.0,
                morpheme_drop_p=0.0,
                rng=rng,
                greekify=True,
                unaccented_hints=False,
                native_gloss_hints=native_gloss_hints,
                target_tag=None,
                canonicalize_preps=True,
            )
            return [(f"<REV_COPTIC> {g}", s) for s, g in fwd]

        reverse_model, rev_hist = train_seq2seq(
            config=cfg,
            tokenizer=tokenizer,
            train_pairs_fn=_rev_epoch_v4,
            dev_pairs=dev_rev_pairs,
            epochs=max(12, epochs // 2),
            lr=6e-4,
            seed=777,
            device=dev,
            select_by_chrf=False,
            patience=5,
            log_fn=typer.echo,
        )
        typer.echo(
            f"  Reverse best_ep={rev_hist.best_epoch}, "
            f"dev_loss={rev_hist.best_dev_loss:.4f}"
        )
        reverse_model.save_safetensors(rev_path, fp16=True)

        typer.echo(
            f"  Back-translating {len(bundle.swete_unaligned_greek)} Swete LXX "
            "(with strict faithfulness filter)..."
        )
        swete_bt_pairs = generate_backtranslations(
            reverse_model=reverse_model,
            tokenizer=tokenizer,
            greek_texts=bundle.swete_unaligned_greek,
            device=dev,
            domain_tag="<DOM_BT>",
            target_tag=tgt_grc_tag,
            filter_faithful=True,
        )
        if use_patristic_bt and bundle.patristic_greek:
            typer.echo(
                f"  Back-translating {len(bundle.patristic_greek)} leak-filtered "
                "Patristic Koine sentences (with strict faithfulness filter)..."
            )
            patristic_bt_pairs = generate_backtranslations(
                reverse_model=reverse_model,
                tokenizer=tokenizer,
                greek_texts=bundle.patristic_greek,
                device=dev,
                domain_tag="<DOM_PAT>",
                target_tag=tgt_grc_tag,
                filter_faithful=True,
            )
        bt_cache.write_text(
            json.dumps(
                {"swete": swete_bt_pairs, "patristic": patristic_bt_pairs},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        typer.echo(
            f"  Faithful BT kept: swete={len(swete_bt_pairs)}, "
            f"patristic={len(patristic_bt_pairs)}"
        )

    all_bt_pairs = [*swete_bt_pairs, *patristic_bt_pairs]

    # 6. Standalone 4-layer Koine Causal LM (NT + Swete + leak-filtered Patristics)
    lm_path = target_dir / "koine-lm.safetensors"
    lm_cfg_path = target_dir / "koine-lm-config.json"
    lm_cfg = KoineCausalLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=6 if d_model % 6 == 0 else 4,
        d_ff=d_model * 4,
        num_layers=4,
        unaccented_target=False,
    )
    if reuse_aux and lm_path.is_file() and lm_cfg_path.is_file():
        typer.echo("6/7 Reusing standalone Koine Causal LM...")
        koine_lm = KoineCausalLM.load_safetensors(lm_path, lm_cfg, device="cpu")
    else:
        typer.echo(
            f"6/7 Training standalone Koine Causal LM "
            f"({len(bundle.all_clean_greek)} clean Koine sentences)..."
        )
        lm_cfg_path.write_text(
            json.dumps(lm_cfg.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        koine_lm, lm_loss = train_causal_koine_lm(
            config=lm_cfg,
            tokenizer=tokenizer,
            train_greek_texts=bundle.all_clean_greek,
            dev_greek_texts=[ex.greek_target for ex in bundle.nt_dev],
            epochs=12,
            lr=6e-4,
            seed=303,
            device=dev,
            log_fn=typer.echo,
        )
        koine_lm.save_safetensors(lm_path, fp16=True)
        typer.echo(f"  Koine LM best dev_loss={lm_loss:.4f}")

    # 7. Denoising pretraining + 1..N coverage/alignment-supervised forward seeds
    if not reuse_seeds:
        for old_f in target_dir.glob("forward-*.safetensors"):
            old_f.unlink()
    else:
        for old_f in target_dir.glob("forward-*.safetensors"):
            stem = old_f.stem.removeprefix("forward-")
            if stem.isdigit() and int(stem) >= seeds:
                old_f.unlink()

    mono_pool: list[str] = []
    if pretrain_epochs > 0:
        mono_pool.extend(bundle.all_clean_greek)
        for c_u in bundle.all_clean_coptic_units:
            dom = (
                "<DOM_APOP>"
                if c_u.source_id == "scriptorium-apophthegmata"
                else "<DOM_OT>"
            )
            s_txt, _ = serialize_coptic_morphemes(
                c_u.morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag=dom,
                greekify=True,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
                canonicalize_preps=True,
            )
            mono_pool.append(s_txt)

    nt_clauses = [c for c in bundle.clause_aug if c.domain_tag == "<DOM_NT>"]
    nt_multiverse = [m for m in bundle.multiverse_aug if m.domain_tag == "<DOM_NT>"]
    forward_models: list[KoineSeq2SeqTransformer] = []

    for s_idx in range(seeds):
        seed_val = 42 + s_idx * 100
        out_w = target_dir / f"forward-{s_idx}.safetensors"
        if reuse_seeds and out_w.is_file():
            typer.echo(
                f"7/7 Reusing forward seed {s_idx + 1}/{seeds} ({out_w.name})..."
            )
            forward_models.append(
                KoineSeq2SeqTransformer.load_safetensors(out_w, cfg, device="cpu")
            )
            continue
        typer.echo(
            f"7/7 Training faithful seed {s_idx + 1}/{seeds} (seed={seed_val})..."
        )
        init_m: KoineSeq2SeqTransformer | None = None
        if pretrain_epochs > 0:
            typer.echo(
                f"  Pretraining denoising ({pretrain_epochs} ep, "
                f"{len(mono_pool)} Greekified+Koine mono lines)..."
            )
            init_m = pretrain_denoising(
                model=KoineSeq2SeqTransformer(cfg),
                tokenizer=tokenizer,
                mono_texts=mono_pool,
                epochs=pretrain_epochs,
                lr=6e-4,
                seed=seed_val,
                device=dev,
                log_fn=typer.echo,
            )

        def _fwd_epoch_v4(rng: random.Random) -> list:
            return make_faithful_forward_pairs_epoch(
                nt_train=bundle.nt_train,
                ot_train=bundle.ot_train,
                clause_aug=bundle.clause_aug,
                multiverse_aug=bundle.multiverse_aug,
                lex_aug=bundle.lex_aug,
                bt_pairs=all_bt_pairs,
                ccl_lexicon=ccl_lexicon,
                lexical_model=lexical_model,
                v1_drafts=v1_drafts if use_v1_draft else None,
                draft_vocab_pool=draft_vocab_pool,
                draft_drop_p=0.20,
                nt_repeat=2,
                max_bt_per_epoch=10000,
                rng=rng,
                greekify=True,
                native_gloss_hints=native_gloss_hints,
                target_tag=tgt_grc_tag,
                canonicalize_preps=True,
            )

        fwd_model, hist = train_seq2seq_faithful(
            config=cfg,
            tokenizer=tokenizer,
            train_pairs_fn=_fwd_epoch_v4,
            dev_pairs=dev_fwd_pairs,
            epochs=epochs,
            lr=6e-4,
            max_len=320,
            seed=seed_val,
            device=dev,
            init_model=init_m,
            inflector=analyzer_inflector,
            coverage_loss_weight=coverage_loss_weight,
            align_loss_weight=align_loss_weight,
            dev_score_fn=_faithful_dev_score,
            patience=8,
            log_fn=typer.echo,
        )
        typer.echo(
            f"  Joint phase best_ep={hist.best_epoch}, "
            f"dev_score={hist.best_dev_chrf:.4f}, dev_loss={hist.best_dev_loss:.4f}"
        )

        if ft_epochs > 0:
            typer.echo(
                f"  Balanced NT+OT+Lex retention fine-tuning ({ft_epochs} ep)..."
            )

            def _ft_epoch_v4(rng: random.Random) -> list:
                ot_ret = (
                    rng.sample(bundle.ot_train, min(len(bundle.ot_train), 2500))
                    if bundle.ot_train
                    else []
                )
                pat_ret = (
                    rng.sample(patristic_bt_pairs, min(len(patristic_bt_pairs), 800))
                    if patristic_bt_pairs
                    else []
                )
                return make_faithful_forward_pairs_epoch(
                    nt_train=bundle.nt_train,
                    ot_train=ot_ret,
                    clause_aug=nt_clauses,
                    multiverse_aug=nt_multiverse,
                    lex_aug=bundle.lex_aug,
                    bt_pairs=pat_ret,
                    ccl_lexicon=ccl_lexicon,
                    lexical_model=lexical_model,
                    v1_drafts=v1_drafts if use_v1_draft else None,
                    draft_vocab_pool=draft_vocab_pool,
                    draft_drop_p=0.15,
                    nt_repeat=1,
                    max_bt_per_epoch=800,
                    rng=rng,
                    greekify=True,
                    native_gloss_hints=native_gloss_hints,
                    target_tag=tgt_grc_tag,
                    canonicalize_preps=True,
                )

            ft_model, ft_hist = train_seq2seq_faithful(
                config=cfg,
                tokenizer=tokenizer,
                train_pairs_fn=_ft_epoch_v4,
                dev_pairs=dev_fwd_pairs,
                epochs=ft_epochs,
                lr=8.0e-5,
                max_len=320,
                seed=seed_val + 1,
                device=dev,
                init_model=fwd_model,
                inflector=analyzer_inflector,
                coverage_loss_weight=coverage_loss_weight,
                align_loss_weight=align_loss_weight,
                dev_score_fn=_faithful_dev_score,
                patience=5,
                log_fn=typer.echo,
            )
            if ft_hist.best_dev_chrf >= hist.best_dev_chrf:
                fwd_model = ft_model
                typer.echo(
                    f"  Kept balanced fine-tuned checkpoint "
                    f"(dev_score={ft_hist.best_dev_chrf:.4f})"
                )
            else:
                typer.echo(
                    f"  Kept joint checkpoint (joint {hist.best_dev_chrf:.4f} > "
                    f"ft {ft_hist.best_dev_chrf:.4f})"
                )

        fwd_model.save_safetensors(out_w, fp16=True)
        forward_models.append(
            KoineSeq2SeqTransformer.load_safetensors(out_w, cfg, device="cpu")
        )

    # Tune faithfulness-aware noisy-channel + coverage rerank weights on 51-verse NT DEV
    base_w = RerankWeights(
        w_fwd=1.0,
        w_rev=0.65,
        w_lm=0.10,
        w_loan=1.5,
        w_cov=0.0,
        w_bipartite_cov=1.2,
        w_attn_cov=0.35,
        w_under_len=1.0,
        len_alpha=1.0,
        beam_size=8,
        max_morphemes=72,
        min_tail_morphemes=12,
        use_v1_draft=use_v1_draft and v1_fast is not None,
        include_v1_candidate=False,
        use_constrained_beam=True,
        use_coverage_repair=True,
        multi_morpheme_align=True,
    )
    dec = NeuralKoineDecoder(
        tokenizer=tokenizer,
        forward_models=forward_models,
        ccl_lexicon=ccl_lexicon,
        analyzer_inflector=analyzer_inflector,
        paradigm_inflector=paradigm_inflector,
        lexical_model=lexical_model,
        reverse_model=reverse_model.to("cpu"),
        koine_lm=koine_lm.to("cpu"),
        v1_decoder=v1_fast,
        weights=base_w,
        device="cpu",
        native_gloss_hints=native_gloss_hints,
        greekify_source=True,
        unaccented_target=False,
        use_target_lang_tag=use_eng_multitask,
        canonicalize_preps=True,
    )
    typer.echo(
        "Tuning faithfulness + coverage rerank weights strictly on NT DEV (51v)..."
    )
    tuned_w = tune_rerank_weights_on_dev(
        decoder=dec, dev_units=bundle.nt_dev_units, tune_faithfulness=True
    )
    tuned_w.save_json(target_dir / "rerank-weights.json")

    write_training_provenance(
        out_dir=target_dir,
        model_id=target_dir.name,
        split_path=split_path,
        ccl_xml_path=ccl_xml,
        nt_train_ids=[ex.unit_id for ex in bundle.nt_train],
        ot_train_ids=[ex.unit_id for ex in bundle.ot_train],
        excluded_holdout_ids=holdout_ids,
        excluded_dev_ids=dev_ids,
        excluded_synoptic_ids=excluded_synoptic,
        extra={
            "seeds": len(forward_models),
            "epochs": epochs,
            "ft_epochs": ft_epochs,
            "pretrain_epochs": pretrain_epochs,
            "coverage_loss_weight": coverage_loss_weight,
            "align_loss_weight": align_loss_weight,
            "greekify_source": True,
            "canonicalize_preps": True,
            "vstat_separated_lexicon": True,
            "native_gloss_hint_count": len(native_gloss_hints),
            "multiverse_window_count": len(bundle.multiverse_aug),
            "swete_unaligned_pre_filter_count": len(bundle.swete_unaligned_greek),
            "swete_bt_count": len(swete_bt_pairs),
            "patristic_pre_filter_count": len(bundle.patristic_greek),
            "patristic_bt_count": len(patristic_bt_pairs),
            "bt_filter_faithful": True,
            "bt_filter_ratio_bounds": [0.42, 1.30],
            "patristic_nt_5gram_filtered": True,
            "use_constrained_beam": True,
            "max_morphemes": tuned_w.max_morphemes,
            "use_v1_draft": use_v1_draft,
        },
    )

    sz_mb = sum(p.stat().st_size for p in target_dir.glob("*")) / (1024 * 1024)
    typer.echo(
        f"OK: froze model-v4 ({len(forward_models)} fwd seeds, "
        f"total={sz_mb:.1f} MiB, weights={tuned_w}) to {target_dir}"
    )
