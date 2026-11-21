# SPDX-License-Identifier: Apache-2.0
"""``thomas model`` Typer subcommands: build & inspect offline model-v1."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.align.lemma_aligner import LexicalTranslationModel
from zetodos_thomas.ingest.download import CCL_TEI_FILENAME, SAHIDIC_OT_ZIP_FILENAME
from zetodos_thomas.ingest.holdouts import load_nt_split
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt, pair_horner_morphgnt
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
    excluded_from_model = holdout_ids | dev_ids

    ot_zip = downloads / "sahidic-ot" / SAHIDIC_OT_ZIP_FILENAME
    ccl_xml = downloads / "ccl-ddglc" / CCL_TEI_FILENAME
    morphgnt_dir = downloads / "morphgnt-sblgnt"

    typer.echo("1/6 Training CopticAnalyzer (sahidic.ot + CCL v1.2)...")
    coptic_analyzer = CopticAnalyzer.train(ot_zip=ot_zip, ccl_xml=ccl_xml)
    c_shards = coptic_analyzer.save_shards(target_dir)

    typer.echo("2/6 Building CCL/DDGLC loanword & compound index...")
    ccl_lexicon = CclLexicon.from_tei_xml(ccl_xml)
    ccl_lexicon.save_json(target_dir / "ccl-loanwords.json")

    typer.echo(
        "3/6 Building Koine Greek paradigm inflector (+ CCL/Swete LXX, no holdout)..."
    )
    inflector = GreekInflector.build(
        morphgnt_dir=morphgnt_dir,
        exclude_unit_ids=holdout_ids,
        ccl_lexicon=ccl_lexicon,
        swete_dir=downloads / "lxx-swete",
    )
    g_shards = inflector.save_shards(target_dir)

    typer.echo(
        "4/6 Running seeded bidirectional eflomal on NT train (3x) + OT x LXX..."
    )
    lexicon = LexicalTranslationModel.build_full(
        downloads_dir=downloads,
        split_path=split_path,
        analyzer=coptic_analyzer,
        inflector=inflector,
        ccl_lexicon=ccl_lexicon,
        nt_weight=3.0,
        repo_root=root,
    )
    l_shards = lexicon.save_shards(target_dir)

    typer.echo(
        "5/6 Extracting idioms/valency & training 3-gram Koine LM (train split)..."
    )
    morphgnt_units = load_morphgnt_nt(morphgnt_dir)
    idioms = IdiomAndValencyTable.build_from_morphgnt(
        morphgnt_units.values(),
        exclude_unit_ids=excluded_from_model,
    )
    idioms.save_json(target_dir / "idiom-valency.json")

    lm = KoineNgramLM.train(
        downloads_dir=downloads,
        split_path=split_path,
        inflector=inflector,
        nt_weight=3.0,
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
    raw_nt = pair_horner_morphgnt(
        sword_root=downloads / "sword",
        morphgnt_dir=morphgnt_dir,
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

    # 1. Load leak-free CopticAnalyzer & CCL loanwords, build offline Greek analyzer
    coptic_analyzer = CopticAnalyzer.load_shards(v1_dir)
    ccl_lexicon = CclLexicon.load_json(v1_dir / "ccl-loanwords.json")
    ccl_lexicon.save_json(target_dir / "ccl-loanwords.json")

    if not reuse_aux or not (target_dir / "greek-analyzer-0.json").is_file():
        typer.echo("1/6 Building leak-free Greek surface analyzer (no holdout)...")
        inflector = GreekInflector.build(
            morphgnt_dir=downloads / "morphgnt-sblgnt",
            exclude_unit_ids=holdout_ids,
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

    sz_mb = sum(p.stat().st_size for p in target_dir.glob("*")) / (1024 * 1024)
    typer.echo(
        f"OK: froze model-v2 ({len(forward_models)} fwd seeds, "
        f"rev={reverse_model is not None}, lm={koine_lm is not None}, "
        f"total={sz_mb:.1f} MiB, weights={tuned_w}) to {target_dir}"
    )
