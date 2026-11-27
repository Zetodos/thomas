# SPDX-License-Identifier: Apache-2.0
"""Deterministic neural ensemble beam decoder with noisy-channel reranking."""

from __future__ import annotations

import dataclasses
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor

from zetodos_thomas.align.lemma_aligner import LexicalTranslationModel
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.neural.accent_restorer import (
    KoineAccentRestorer,
    _accent_class,
    _bare_key,
)
from zetodos_thomas.neural.constrained_beam import (
    CONTENT_COPTIC_POS,
    assign_bound_group_coptic_seqs,
    bipartite_lexicon_coverage,
    build_chunk_lexical_constraints,
    build_coptic_content_candidate_sets,
    build_missing_content_constraints,
    constrained_ensemble_beam_search,
    count_spurious_adjacent_word_repeats,
    filter_hard_loan_hyps,
    has_local_loop_repeat,
    is_bare_coptic_copy_word,
    wu_attention_coverage_penalty,
)
from zetodos_thomas.neural.corpus import (
    make_sub_coptic_unit,
    normalize_greek_for_model,
    normalize_greek_target,
    resolve_morpheme_loan_hint,
    serialize_coptic_morphemes,
    split_coptic_unit_for_neural,
)
from zetodos_thomas.neural.tokenizer import BilingualTokenizer
from zetodos_thomas.neural.transformer import (
    KoineCausalLM,
    KoineCausalLMConfig,
    KoineSeq2SeqTransformer,
    Seq2SeqConfig,
)
from zetodos_thomas.reconstruct.decoder import KoineBeamDecoder
from zetodos_thomas.schemas.generated import (
    GeneratedCorpus,
    GeneratedToken,
    GeneratedUnit,
    TokenAlternative,
)
from zetodos_thomas.schemas.linguistic import (
    CopticMorpheme,
    CopticUnit,
    GreekMorphFeatures,
    GreekUnit,
)


class RerankWeights(BaseModel):
    """Noisy-channel reranking weights tuned strictly on 51-verse NT DEV split."""

    w_fwd: float = 1.0
    w_rev: float = 0.0
    w_lm: float = 0.0
    w_loan: float = 0.6
    w_cov: float = 0.0
    w_bipartite_cov: float = 0.0
    w_attn_cov: float = 0.0
    w_under_len: float = 0.0
    len_alpha: float = 0.9
    beam_size: int = 8
    max_morphemes: int = 36
    min_tail_morphemes: int = 0
    use_v1_draft: bool = False
    include_v1_candidate: bool = False
    use_constrained_beam: bool = False
    use_coverage_repair: bool = False
    multi_morpheme_align: bool = False

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> RerankWeights:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_analyzer_shards(
    inflector: GreekInflector, out_dir: Path, *, max_bytes: int = 650_000
) -> list[Path]:
    """Persist MorphGNT-train+Swete surface analyzer tables in <1024 KiB shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("greek-analyzer-*.json"):
        old.unlink()
    files: list[Path] = []

    items = sorted(inflector._surface_analyses.items())
    shard: dict[str, list[str | float]] = {}
    shard_idx = 0
    est = 2
    for surf, parses in items:
        if not parses:
            continue
        lem, sig, lp = parses[0]
        row = [lem, sig, round(lp, 3)]
        row_bytes = (
            len(surf.encode("utf-8"))
            + len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
            + 6
        )
        if est + row_bytes > max_bytes and shard:
            p = out_dir / f"greek-analyzer-{shard_idx}.json"
            p.write_text(
                json.dumps(shard, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            files.append(p)
            shard = {}
            shard_idx += 1
            est = 2
        shard[surf] = row
        est += row_bytes
    if shard:
        p = out_dir / f"greek-analyzer-{shard_idx}.json"
        p.write_text(
            json.dumps(shard, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        files.append(p)

    known_p = out_dir / "greek-known-lemmas.json"
    known_p.write_text(
        json.dumps(
            {k: [v[0], v[1]] for k, v in sorted(inflector._known_lemmas.items())},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    files.append(known_p)
    return files


def load_analyzer_inflector(model_dir: Path) -> GreekInflector:
    """Load offline surface analyzer (MorphGNT train + Swete, zero holdout)."""
    surface_analyses: dict[str, list[tuple[str, str, float]]] = {}
    for p in sorted(model_dir.glob("greek-analyzer-*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for surf, row in data.items():
            surface_analyses[surf] = [(str(row[0]), str(row[1]), float(row[2]))]
    known_p = model_dir / "greek-known-lemmas.json"
    known_lemmas: dict[str, tuple[str, str]] = {}
    if known_p.is_file():
        kdata = json.loads(known_p.read_text(encoding="utf-8"))
        known_lemmas = {k: (str(v[0]), str(v[1])) for k, v in kdata.items()}
    return GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={},
        surface_analyses=surface_analyses,
        known_lemmas=known_lemmas,
    )


def _strip_bare(word: str) -> str:
    nfd = unicodedata.normalize("NFD", word)
    return (
        "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        .lower()
        .replace("ς", "σ")
    )


def _has_trigram_repeat(seq: list[int], next_id: int) -> bool:
    return has_local_loop_repeat(seq, next_id)


class NeuralKoineDecoder:
    """Deterministic offline ensemble NMT + noisy-channel Koine Greek decoder."""

    def __init__(
        self,
        *,
        tokenizer: BilingualTokenizer,
        forward_models: list[KoineSeq2SeqTransformer],
        ccl_lexicon: CclLexicon,
        analyzer_inflector: GreekInflector,
        reverse_model: KoineSeq2SeqTransformer | None = None,
        koine_lm: KoineCausalLM | None = None,
        v1_decoder: KoineBeamDecoder | None = None,
        weights: RerankWeights | None = None,
        device: torch.device | str = "cpu",
        accent_restorer: KoineAccentRestorer | None = None,
        native_gloss_hints: dict[str, str] | None = None,
        greekify_source: bool = False,
        unaccented_target: bool = False,
        use_target_lang_tag: bool = False,
        paradigm_inflector: GreekInflector | None = None,
        lexical_model: LexicalTranslationModel | None = None,
        canonicalize_preps: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.forward_models = forward_models
        self.ccl_lexicon = ccl_lexicon
        self.inflector = analyzer_inflector
        self.paradigm_inflector = paradigm_inflector or (
            v1_decoder.inflector if v1_decoder is not None else analyzer_inflector
        )
        self.lexical_model = lexical_model or (
            v1_decoder.lexicon if v1_decoder is not None else None
        )
        self.reverse_model = reverse_model
        self.koine_lm = koine_lm
        self.v1_decoder = v1_decoder
        self.weights = weights or RerankWeights()
        self.device = torch.device(device)
        self.accent_restorer = accent_restorer
        self.native_gloss_hints = native_gloss_hints
        self._greekify = greekify_source
        self._unaccented = unaccented_target
        self._tgt_grc_tag: str | None = "<TGT_GRC>" if use_target_lang_tag else None
        self._canonicalize_preps = canonicalize_preps
        # Precompute bare prefixes of all zero-holdout known Greek surfaces and
        # lemmas so beam search can immediately penalize dead subword prefixes
        # (e.g. `δαπανήσει`, `φορτίοντίον`, `ἐξαλείπτας`) without delaying until
        # word close.
        known_prefixes: set[str] = set()
        for s_key in self.inflector._surface_analyses:
            b_s = _strip_bare(s_key)
            for k_len in range(1, len(b_s) + 1):
                known_prefixes.add(b_s[:k_len])
        for l_key in self.inflector._known_lemmas:
            b_l = _strip_bare(l_key)
            for k_len in range(1, len(b_l) + 1):
                known_prefixes.add(b_l[:k_len])
        self._known_bare_prefixes: frozenset[str] = frozenset(known_prefixes)

        for m in self.forward_models:
            m.to(self.device)
            m.eval()
        if self.reverse_model is not None:
            self.reverse_model.to(self.device)
            self.reverse_model.eval()
        if self.koine_lm is not None:
            self.koine_lm.to(self.device)
            self.koine_lm.eval()

    def _norm_model_greek(self, text: str) -> str:
        return normalize_greek_for_model(text, unaccented=self._unaccented)

    def _restore_surface_sentence(self, text: str) -> str:
        if self._unaccented and self.accent_restorer is not None:
            return self.accent_restorer.restore_sentence(text)
        return normalize_greek_target(text)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        *,
        v1_model_dir: Path | None = None,
        device: str = "cpu",
    ) -> NeuralKoineDecoder:
        """Load frozen model-v2 or model-v3 weights and offline analyzer tables."""
        tokenizer = BilingualTokenizer.load(model_dir / "tokenizer.model")
        cfg = Seq2SeqConfig.model_validate(
            json.loads((model_dir / "seq2seq-config.json").read_text(encoding="utf-8"))
        )
        fwd_paths = sorted(model_dir.glob("forward-*.safetensors"))
        if not fwd_paths:
            msg = f"No forward-*.safetensors weights found in {model_dir}"
            raise FileNotFoundError(msg)
        forward_models = [
            KoineSeq2SeqTransformer.load_safetensors(p, cfg, device=device)
            for p in fwd_paths
        ]

        rev_path = model_dir / "reverse.safetensors"
        reverse_model = (
            KoineSeq2SeqTransformer.load_safetensors(rev_path, cfg, device=device)
            if rev_path.is_file()
            else None
        )

        lm_path = model_dir / "koine-lm.safetensors"
        lm_cfg_path = model_dir / "koine-lm-config.json"
        koine_lm = (
            KoineCausalLM.load_safetensors(
                lm_path,
                KoineCausalLMConfig.model_validate(
                    json.loads(lm_cfg_path.read_text(encoding="utf-8"))
                ),
                device=device,
            )
            if lm_path.is_file() and lm_cfg_path.is_file()
            else None
        )

        ccl_path = model_dir / "ccl-loanwords.json"
        if not ccl_path.is_file() and v1_model_dir is not None:
            ccl_path = v1_model_dir / "ccl-loanwords.json"
        ccl_lexicon = CclLexicon.load_json(ccl_path)
        analyzer_inflector = load_analyzer_inflector(model_dir)

        accent_restorer: KoineAccentRestorer | None = None
        if cfg.unaccented_target and any(model_dir.glob("accent-uni-*.json")):
            accent_restorer = KoineAccentRestorer.load_shards(model_dir)

        native_gloss_hints: dict[str, str] | None = None
        gloss_path = model_dir / "native-gloss-hints.json"
        if cfg.use_gloss_hints and gloss_path.is_file():
            raw_gh = json.loads(gloss_path.read_text(encoding="utf-8"))
            if isinstance(raw_gh, dict):
                native_gloss_hints = {str(k): str(v) for k, v in raw_gh.items()}

        w_path = model_dir / "rerank-weights.json"
        weights = (
            RerankWeights.load_json(w_path) if w_path.is_file() else RerankWeights()
        )

        need_v1 = (
            weights.use_v1_draft
            or weights.include_v1_candidate
            or weights.use_constrained_beam
            or abs(weights.w_bipartite_cov) > 1e-6
        )
        resolved_v1_dir = v1_model_dir
        if resolved_v1_dir is None and need_v1:
            sibling = model_dir.parent / "model-v1"
            if (sibling / "decoder-weights.json").is_file():
                resolved_v1_dir = sibling
        v1_dec: KoineBeamDecoder | None = None
        if (
            need_v1
            and resolved_v1_dir is not None
            and (resolved_v1_dir / "decoder-weights.json").is_file()
        ):
            v1_dec = KoineBeamDecoder.from_model_dir(resolved_v1_dir)
            v1_dec.weights = dataclasses.replace(v1_dec.weights, beam_size=2)
        paradigm_inf: GreekInflector | None = None
        if any(model_dir.glob("greek-paradigms-*.json")):
            paradigm_inf = GreekInflector.load_shards(model_dir)
        elif v1_dec is not None:
            paradigm_inf = v1_dec.inflector
        lex_model: LexicalTranslationModel | None = None
        if any(model_dir.glob("lemma-align-*.json")):
            lex_model = LexicalTranslationModel.load_shards(model_dir)
        elif v1_dec is not None:
            lex_model = v1_dec.lexicon
        if v1_dec is not None:
            if lex_model is not None and any(model_dir.glob("lemma-align-*.json")):
                v1_dec.lexicon = lex_model
            if paradigm_inf is not None and any(
                model_dir.glob("greek-paradigms-*.json")
            ):
                v1_dec.inflector = paradigm_inf
        return cls(
            tokenizer=tokenizer,
            forward_models=forward_models,
            ccl_lexicon=ccl_lexicon,
            analyzer_inflector=analyzer_inflector,
            reverse_model=reverse_model,
            koine_lm=koine_lm,
            v1_decoder=v1_dec,
            weights=weights,
            device=device,
            accent_restorer=accent_restorer,
            native_gloss_hints=native_gloss_hints,
            greekify_source=cfg.greekify_source,
            unaccented_target=cfg.unaccented_target,
            use_target_lang_tag=cfg.use_target_lang_tag,
            paradigm_inflector=paradigm_inf,
            lexical_model=lex_model,
            canonicalize_preps=getattr(cfg, "canonicalize_preps", False),
        )

    @torch.no_grad()
    def _score_seq2seq(
        self,
        models: Sequence[KoineSeq2SeqTransformer],
        src_ids: list[int],
        tgt_seqs: list[list[int]],
    ) -> list[float]:
        """Compute ensemble log-likelihood ``log P(tgt | src)`` for each candidate."""
        if not tgt_seqs:
            return []
        src_t = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        pad_id = self.tokenizer.pad_id
        max_l = max(len(s) for s in tgt_seqs)
        if max_l < 2:
            return [0.0 for _ in tgt_seqs]
        tgt_b = torch.full(
            (len(tgt_seqs), max_l), pad_id, dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(tgt_seqs):
            tgt_b[i, : len(seq)] = torch.tensor(
                seq, dtype=torch.long, device=self.device
            )
        tgt_in = tgt_b[:, :-1]
        tgt_out = tgt_b[:, 1:]
        mask = tgt_out.ne(pad_id).float()

        log_prob_stack: list[Tensor] = []
        for m in models:
            mem, spm_mask = m.encode(src_t)
            mem_exp = mem.expand(len(tgt_seqs), -1, -1)
            spm_exp = spm_mask.expand(len(tgt_seqs), -1)
            logits, _ = m.decode(tgt_in, mem_exp, spm_exp, return_cross_attn=False)
            lp = F.log_softmax(logits, dim=-1)
            log_prob_stack.append(lp)

        if len(log_prob_stack) == 1:
            ens_lp = log_prob_stack[0]
        else:
            ens_lp = torch.logsumexp(
                torch.stack(log_prob_stack, dim=0), dim=0
            ) - math.log(len(log_prob_stack))
        tok_lp = ens_lp.gather(2, tgt_out.unsqueeze(-1)).squeeze(-1)
        totals = (tok_lp * mask).sum(dim=1)
        return [float(x) for x in totals.cpu().tolist()]

    @torch.no_grad()
    def _ensemble_beam_search(
        self,
        src_ids: list[int],
        *,
        beam_size: int,
        max_new_tokens: int,
        min_new_tokens: int = 1,
        len_alpha: float,
        word_penalty_fn: Callable[[tuple[int, ...]], float] | None = None,
        prefix_penalty_fn: Callable[[tuple[int, ...]], float] | None = None,
    ) -> list[tuple[list[int], float]]:
        """Deterministic probability-mixture ensemble beam search."""
        src_t = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        encoded = [m.encode(src_t) for m in self.forward_models]
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        n_models = len(self.forward_models)
        word_starts = self.tokenizer.word_start_flags()

        active: list[tuple[list[int], float]] = [([bos_id], 0.0)]
        finished: list[tuple[list[int], float]] = []

        def _last_wtup(s: list[int]) -> tuple[int, ...]:
            n = len(s)
            if n <= 1:
                return ()
            idx = n - 1
            while idx > 1 and not word_starts[s[idx]]:
                idx -= 1
            return tuple(s[idx:])

        for _ in range(max_new_tokens):
            if not active:
                break
            cur_bsz = len(active)
            seq_len = len(active[0][0])
            tgt_t = torch.tensor(
                [seq for seq, _ in active], dtype=torch.long, device=self.device
            )
            lp_list: list[Tensor] = []
            for m, (mem, spm_mask) in zip(self.forward_models, encoded, strict=True):
                logits, _ = m.decode(
                    tgt_t,
                    mem.expand(cur_bsz, -1, -1),
                    spm_mask.expand(cur_bsz, -1),
                    return_cross_attn=False,
                )
                lp_list.append(F.log_softmax(logits[:, -1, :], dim=-1))

            if n_models == 1:
                step_lp = lp_list[0]
            else:
                step_lp = torch.logsumexp(
                    torch.stack(lp_list, dim=0), dim=0
                ) - math.log(n_models)

            top_vals, top_idx = torch.topk(
                step_lp, k=min(beam_size * 2, step_lp.size(-1)), dim=-1
            )
            top_vals_cpu = top_vals.cpu().tolist()
            top_idx_cpu = top_idx.cpu().tolist()

            next_cands: list[tuple[list[int], float]] = []
            for b_i, (seq, score) in enumerate(active):
                cur_w = (
                    _last_wtup(seq)
                    if len(seq) > 1
                    and (word_penalty_fn is not None or prefix_penalty_fn is not None)
                    else ()
                )
                closed_w_pen = (
                    word_penalty_fn(cur_w)
                    if word_penalty_fn is not None and cur_w
                    else 0.0
                )
                for val, tid in zip(top_vals_cpu[b_i], top_idx_cpu[b_i], strict=True):
                    itid = int(tid)
                    if itid == self.tokenizer.pad_id or itid == bos_id:
                        continue
                    if itid != eos_id and _has_trigram_repeat(seq, itid):
                        continue
                    if (
                        itid != eos_id
                        and not word_starts[itid]
                        and len(seq) > 1
                        and seq[-1] == itid
                    ):
                        continue
                    new_seq = [*seq, itid]
                    if itid == eos_id:
                        if seq_len < min_new_tokens:
                            continue
                        new_score = score + closed_w_pen + float(val)
                        norm_sc = new_score / (max(seq_len, 1) ** len_alpha)
                        finished.append((new_seq, norm_sc))
                    else:
                        if word_starts[itid]:
                            w_pen = closed_w_pen + (
                                prefix_penalty_fn((itid,))
                                if prefix_penalty_fn is not None
                                else 0.0
                            )
                        else:
                            w_pen = (
                                prefix_penalty_fn((*cur_w, itid))
                                if prefix_penalty_fn is not None
                                else 0.0
                            )
                        next_cands.append((new_seq, score + w_pen + float(val)))

            next_cands.sort(
                key=lambda item: item[1] / (max(len(item[0]) - 1, 1) ** len_alpha),
                reverse=True,
            )
            active = next_cands[:beam_size]
            if len(finished) >= beam_size * 2:
                best_fin = max(sc for _, sc in finished)
                best_act = (
                    max(sc / (max(len(s) - 1, 1) ** len_alpha) for s, sc in active)
                    if active
                    else -1e9
                )
                if best_act < best_fin - 0.5:
                    break

        for seq, sc in active:
            closed = [*seq, eos_id]
            finished.append((closed, sc / (max(len(seq), 1) ** len_alpha)))

        finished.sort(key=lambda item: item[1], reverse=True)
        # Deduplicate exact token sequences
        dedup: list[tuple[list[int], float]] = []
        seen: set[tuple[int, ...]] = set()
        for seq, sc in finished:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                dedup.append((seq, sc))
            if len(dedup) >= beam_size:
                break
        return dedup

    @torch.no_grad()
    def _extract_cross_attention(
        self, src_ids: list[int], tgt_ids: list[int]
    ) -> list[list[float]]:
        """Compute averaged encoder-decoder cross-attention `[T_tgt, T_src]`."""
        src_t = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        tgt_in = torch.tensor([tgt_ids[:-1]], dtype=torch.long, device=self.device)
        attns: list[Tensor] = []
        for m in self.forward_models:
            _, ca = m(src_t, tgt_in, return_cross_attn=True)
            if ca is not None:
                attns.append(ca.squeeze(0))
        if not attns:
            return []
        avg = torch.stack(attns, dim=0).mean(dim=0)
        return [[float(v) for v in row] for row in avg.cpu().tolist()]

    def _rerank_candidates(
        self,
        *,
        src_ids: list[int],
        coptic_only_text: str,
        morphemes: Sequence[CopticMorpheme],
        beam_hyps: list[tuple[list[int], float]],
        v1_candidate_text: str | None = None,
        tok_to_seq: Sequence[int | None] | None = None,
    ) -> list[tuple[list[int], str, float]]:
        """Score & sort candidate hypotheses with noisy-channel + coverage reranking."""
        w = self.weights
        cands_ids: list[list[int]] = [seq for seq, _ in beam_hyps]
        cands_txt: list[str] = [
            self._norm_model_greek(self.tokenizer.decode(seq)) for seq in cands_ids
        ]
        if v1_candidate_text and w.include_v1_candidate:
            v1_norm = self._norm_model_greek(v1_candidate_text)
            if v1_norm and v1_norm not in cands_txt:
                cands_ids.append(self.tokenizer.encode(v1_norm))
                cands_txt.append(v1_norm)

        # Forward score under ensemble
        fwd_raw = self._score_seq2seq(self.forward_models, src_ids, cands_ids)

        # Reverse noisy-channel score: log P_rev(pure Coptic | Greek)
        rev_scores = [0.0 for _ in cands_ids]
        if self.reverse_model is not None and abs(w.w_rev) > 1e-6:
            c_tgt_ids = self.tokenizer.encode(coptic_only_text)
            c_len_norm = max(len(c_tgt_ids) - 1, 1) ** w.len_alpha
            for idx, g_txt in enumerate(cands_txt):
                if not g_txt:
                    rev_scores[idx] = -20.0
                    continue
                g_src = self.tokenizer.encode(f"<REV_COPTIC> {g_txt}")
                rv = self._score_seq2seq([self.reverse_model], g_src, [c_tgt_ids])[0]
                rev_scores[idx] = rv / c_len_norm

        # Standalone Koine Greek Neural LM score
        lm_scores = [0.0 for _ in cands_ids]
        if self.koine_lm is not None and abs(w.w_lm) > 1e-6:
            raw_lm = self.koine_lm.score_token_seqs(cands_ids, device=self.device)
            for idx, (seq, val) in enumerate(zip(cands_ids, raw_lm, strict=True)):
                lm_scores[idx] = val / (max(len(seq) - 1, 1) ** w.len_alpha)

        from zetodos_thomas.ingest.normalize import normalize_coptic

        # Expected Greek loanwords (multiset) from CCL/DDGLC & draft coverage set
        expected_loans: Counter[str] = Counter()
        loan_inflected_bare: set[str] = set()
        for m in morphemes:
            g_hint = resolve_morpheme_loan_hint(m, self.ccl_lexicon)
            if g_hint:
                expected_loans[_strip_bare(g_hint)] += 1
                g_pos_h = (
                    "V-"
                    if m.pos in {"V", "VSTAT", "VIMP"}
                    else "N-"
                    if m.pos in {"N", "NPROP"}
                    else None
                )
                for form in self.paradigm_inflector.all_inflected_forms(
                    g_hint, pos=g_pos_h, max_forms=64
                ):
                    loan_inflected_bare.add(_strip_bare(form))
        n_exp_loans = sum(expected_loans.values())

        content_pos = {"N-", "V-", "A-", "D-", "X-"}
        draft_content_lems: set[str] = set()
        if v1_candidate_text and abs(w.w_cov) > 1e-6:
            for dw in normalize_greek_target(v1_candidate_text).split():
                d_lem, d_mf, _ = self.inflector.analyze_surface(dw)[0]
                if d_mf.pos in content_pos:
                    draft_content_lems.add(_strip_bare(d_lem))

        # Multiset/synonym-aware per-morpheme bilingual lexicon coverage
        coptic_cand_sets: list[tuple[int, set[str]]] = []
        if abs(w.w_bipartite_cov) > 1e-6 or w.use_constrained_beam:
            coptic_cand_sets = build_coptic_content_candidate_sets(
                morphemes,
                ccl_lexicon=self.ccl_lexicon,
                lexical_model=self.lexical_model,
                native_gloss_hints=self.native_gloss_hints,
            )
        known_cand_bare: set[str] = set(expected_loans) | loan_inflected_bare
        for _sq, cset in coptic_cand_sets:
            known_cand_bare.update(cset)
        chunk_has_nprop = any(m.pos == "NPROP" for m in morphemes)
        chunk_lems = {
            normalize_coptic(m.lemma or m.norm).replace("\u2cff", "") for m in morphemes
        }
        forbid_hemera = "ⲉϩⲉ" in chunk_lems and "ϩⲟⲟⲩ" not in chunk_lems

        content_seqs = [m.seq for m in morphemes if m.pos in CONTENT_COPTIC_POS]
        n_cop_content = len(content_seqs)
        need_attn_cov = (
            abs(w.w_attn_cov) > 1e-6 and tok_to_seq is not None and bool(content_seqs)
        )

        ranked: list[tuple[list[int], str, float]] = []
        for idx, (seq, g_txt) in enumerate(zip(cands_ids, cands_txt, strict=True)):
            fwd_n = fwd_raw[idx] / (max(len(seq) - 1, 1) ** w.len_alpha)
            loan_bonus = 0.0
            cov_bonus = 0.0
            bip_bonus = 0.0
            under_pen = 0.0
            attn_cov = 0.0
            bare_copy_pen = 0.0
            if g_txt and (
                expected_loans
                or draft_content_lems
                or coptic_cand_sets
                or abs(w.w_under_len) > 1e-6
                or w.use_constrained_beam
            ):
                g_lems: set[str] = set()
                g_lem_counts: Counter[str] = Counter()
                g_content_seq: list[str] = []
                restored_cand = self._restore_surface_sentence(g_txt)
                cand_words = restored_cand.split()
                if w.use_constrained_beam:
                    bare_copy_pen += 0.8 * count_spurious_adjacent_word_repeats(
                        cand_words, morphemes
                    )
                for gw in cand_words:
                    is_bare_copy = (
                        not self._unaccented
                        and w.use_constrained_beam
                        and is_bare_coptic_copy_word(gw)
                    )
                    if is_bare_copy:
                        bare_copy_pen += 1.5
                    p0 = self.inflector.analyze_surface(gw)[0]
                    b_lem = _strip_bare(p0[0])
                    b_surf = _strip_bare(gw)
                    if w.use_constrained_beam and forbid_hemera and b_lem == "ημερα":
                        bare_copy_pen += 2.0
                    elif (
                        w.use_constrained_beam
                        and not is_bare_copy
                        and p0[1].pos == "UNK"
                        and b_lem not in known_cand_bare
                        and b_surf not in known_cand_bare
                        and not (gw[0].isupper() and chunk_has_nprop)
                    ):
                        # Penalize Greek surfaces that GreekInflector cannot parse.
                        bare_copy_pen += 0.80
                    if not is_bare_copy:
                        g_lems.add(b_lem)
                        g_lems.add(b_surf)
                        g_lem_counts[b_lem] += 1
                        if b_surf != b_lem:
                            g_lem_counts[b_surf] += 1
                    if p0[1].pos in content_pos:
                        g_content_seq.append(b_lem)
                if n_exp_loans:
                    hit_n = sum(
                        min(cnt, g_lem_counts[el]) for el, cnt in expected_loans.items()
                    )
                    loan_bonus = hit_n / n_exp_loans
                if draft_content_lems:
                    cov_n = sum(1 for dl in draft_content_lems if dl in g_lems)
                    cov_bonus = cov_n / len(draft_content_lems)
                if coptic_cand_sets:
                    bip_bonus = bipartite_lexicon_coverage(
                        coptic_cand_sets, g_content_seq
                    )
                if abs(w.w_under_len) > 1e-6 and n_cop_content > 0:
                    # Penalize dropping below 0.85x or hallucinating >1.30x
                    # Coptic content count (prevents Pass-2 padding loops).
                    min_exp = max(1.0, 0.85 * n_cop_content)
                    max_exp = max(n_cop_content + 3.0, 1.30 * n_cop_content)
                    if len(g_content_seq) < min_exp:
                        under_pen = (min_exp - len(g_content_seq)) / n_cop_content
                    elif len(g_content_seq) > max_exp:
                        under_pen = (len(g_content_seq) - max_exp) / n_cop_content
            if need_attn_cov and len(seq) >= 2 and tok_to_seq is not None:
                c_attn = self._extract_cross_attention(src_ids, seq)
                attn_cov = wu_attention_coverage_penalty(
                    c_attn, tok_to_seq, content_seqs, tau=0.45
                )

            total = (
                w.w_fwd * fwd_n
                + w.w_rev * rev_scores[idx]
                + w.w_lm * lm_scores[idx]
                + w.w_loan * loan_bonus
                + w.w_cov * cov_bonus
                + w.w_bipartite_cov * bip_bonus
                + w.w_attn_cov * attn_cov
                - w.w_under_len * under_pen
                - bare_copy_pen
            )
            ranked.append((seq, g_txt, total))

        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked

    def _build_beam_penalty_fns(
        self,
        morphemes: Sequence[CopticMorpheme],
        constraints: Sequence[object],
    ) -> tuple[
        Callable[[tuple[int, ...]], float] | None,
        Callable[[tuple[int, ...]], float] | None,
    ]:
        """Build memoized beam-time completed-word & live-prefix penalty fns."""
        if not self.weights.use_constrained_beam:
            return None, None
        from zetodos_thomas.ingest.normalize import normalize_coptic

        allowed_bare: set[str] = set()
        for m in morphemes:
            gh = resolve_morpheme_loan_hint(m, self.ccl_lexicon)
            if gh:
                allowed_bare.add(_strip_bare(gh))
                g_pos_h = (
                    "V-"
                    if m.pos in {"V", "VSTAT", "VIMP"}
                    else "N-"
                    if m.pos in {"N", "NPROP"}
                    else None
                )
                for form in self.paradigm_inflector.all_inflected_forms(
                    gh, pos=g_pos_h, max_forms=64
                ):
                    allowed_bare.add(_strip_bare(form))
        for ct in constraints:
            bs = getattr(ct, "bare_surfaces", None)
            if bs:
                allowed_bare.update(bs)
        csets = build_coptic_content_candidate_sets(
            morphemes,
            ccl_lexicon=self.ccl_lexicon,
            lexical_model=self.lexical_model,
            native_gloss_hints=self.native_gloss_hints,
        )
        for _sq, cset in csets:
            allowed_bare.update(cset)
        allowed_prefixes: set[str] = set()
        for ab in allowed_bare:
            for k_len in range(1, len(ab) + 1):
                allowed_prefixes.add(ab[:k_len])
        chunk_has_nprop = any(m.pos == "NPROP" for m in morphemes)
        chunk_lems = {
            normalize_coptic(m.lemma or m.norm).replace("\u2cff", "") for m in morphemes
        }
        forbid_hemera = "ⲉϩⲉ" in chunk_lems and "ϩⲟⲟⲩ" not in chunk_lems
        known_prefixes = self._known_bare_prefixes
        w_cache: dict[tuple[int, ...], float] = {}
        p_cache: dict[tuple[int, ...], float] = {}

        def _word_pen(w_tup: tuple[int, ...]) -> float:
            if not w_tup:
                return 0.0
            hit = w_cache.get(w_tup)
            if hit is not None:
                return hit
            raw = self._norm_model_greek(self.tokenizer.decode(list(w_tup)))
            if not raw:
                w_cache[w_tup] = 0.0
                return 0.0
            word = self._restore_surface_sentence(raw).strip()
            if not word or " " in word:
                w_cache[w_tup] = 0.0
                return 0.0
            lem, mf, _ = self.inflector.analyze_surface(word)[0]
            b_lem = _strip_bare(lem)
            b_surf = _strip_bare(word)
            pen = 0.0
            if forbid_hemera and b_lem == "ημερα":
                pen -= 8.0
            elif (
                mf.pos == "UNK"
                and b_lem not in allowed_bare
                and b_surf not in allowed_bare
                and not (word[0].isupper() and chunk_has_nprop)
            ):
                pen -= 2.5
            w_cache[w_tup] = pen
            return pen

        def _prefix_pen(w_tup: tuple[int, ...]) -> float:
            if not w_tup:
                return 0.0
            hit = p_cache.get(w_tup)
            if hit is not None:
                return hit
            raw = self._norm_model_greek(self.tokenizer.decode(list(w_tup))).strip()
            if not raw or " " in raw:
                p_cache[w_tup] = 0.0
                return 0.0
            bare = _strip_bare(raw)
            if not bare or not any("α" <= ch <= "ω" for ch in bare):
                p_cache[w_tup] = 0.0
                return 0.0
            pen = 0.0
            if forbid_hemera and bare.startswith("ημερ"):
                pen -= 8.0
            elif chunk_has_nprop and raw[0].isupper():
                pen = 0.0
            elif bare not in known_prefixes and bare not in allowed_prefixes:
                # Allow regular stem-suffix inflections (e.g. `κρυπτοῦ`) that
                # GreekInflector parses via morphology rules even if absent
                # from the literal MorphGNT-train+Swete surface table.
                restored_w = self._restore_surface_sentence(raw).strip()
                if (
                    not restored_w
                    or " " in restored_w
                    or self.inflector.analyze_surface(restored_w)[0][1].pos == "UNK"
                ):
                    pen -= 3.0
            p_cache[w_tup] = pen
            return pen

        return _word_pen, _prefix_pen

    def _decode_chunk(
        self,
        morphemes: Sequence[CopticMorpheme],
        *,
        domain_tag: str = "<DOM_NT>",
        v1_chunk_greek: str | None = None,
    ) -> tuple[list[GeneratedToken], list[str]]:
        """Decode one Coptic morpheme chunk into aligned ``GeneratedToken``s."""
        coptic_only_text, _ = serialize_coptic_morphemes(
            morphemes,
            ccl_lexicon=self.ccl_lexicon,
            domain_tag=domain_tag,
            greekify=self._greekify,
            unaccented_hints=self._unaccented,
            native_gloss_hints=self.native_gloss_hints,
            canonicalize_preps=self._canonicalize_preps,
        )
        draft_arg = v1_chunk_greek if self.weights.use_v1_draft else None
        src_text, char_spans = serialize_coptic_morphemes(
            morphemes,
            ccl_lexicon=self.ccl_lexicon,
            domain_tag=domain_tag,
            v1_draft=draft_arg,
            greekify=self._greekify,
            unaccented_hints=self._unaccented,
            native_gloss_hints=self.native_gloss_hints,
            target_tag=self._tgt_grc_tag,
            canonicalize_preps=self._canonicalize_preps,
        )
        src_ids, tok_to_seq = self.tokenizer.encode_with_coptic_seq_map(
            src_text, char_spans
        )
        n_non_punct = sum(1 for m in morphemes if m.pos != "PUNCT")
        min_new = max(2, int(n_non_punct * 0.45)) if n_non_punct >= 5 else 1
        max_new = min(135, max(24, int(len(morphemes) * 1.85) + 18))
        est_tgt = max(int(n_non_punct * 0.85), 8)
        if self.weights.use_constrained_beam:
            constraints = build_chunk_lexical_constraints(
                morphemes,
                ccl_lexicon=self.ccl_lexicon,
                inflector=self.paradigm_inflector,
                tokenizer=self.tokenizer,
                lexical_model=self.lexical_model,
                unaccented=self._unaccented,
            )
        else:
            constraints = []
        word_pen_fn, prefix_pen_fn = self._build_beam_penalty_fns(
            morphemes, constraints
        )
        if constraints:
            beam_hyps = constrained_ensemble_beam_search(
                forward_models=self.forward_models,
                tokenizer=self.tokenizer,
                src_ids=src_ids,
                constraints=constraints,
                beam_size=self.weights.beam_size,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                len_alpha=self.weights.len_alpha,
                device=self.device,
                est_tgt_len=est_tgt,
                word_penalty_fn=word_pen_fn,
                prefix_penalty_fn=prefix_pen_fn,
            )
        else:
            beam_hyps = self._ensemble_beam_search(
                src_ids,
                beam_size=self.weights.beam_size,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                len_alpha=self.weights.len_alpha,
                word_penalty_fn=word_pen_fn,
                prefix_penalty_fn=prefix_pen_fn,
            )
        ranked = self._rerank_candidates(
            src_ids=src_ids,
            coptic_only_text=coptic_only_text,
            morphemes=morphemes,
            beam_hyps=beam_hyps,
            v1_candidate_text=v1_chunk_greek,
            tok_to_seq=tok_to_seq,
        )
        # Pass-2 Coverage Repair: if pass-1 dropped content Coptic morphemes,
        # add positional synonym tries and re-rank jointly (and if pass-2
        # introduced a newly dropped target such as a tail verb, repair once).
        if (
            self.weights.use_coverage_repair
            and self.lexical_model is not None
            and ranked
            and ranked[0][1]
        ):
            p1_restored = self._restore_surface_sentence(ranked[0][1])
            p1_content_lems: list[str] = []
            for gw in p1_restored.split():
                lem, mf, _ = self.inflector.analyze_surface(gw)[0]
                if mf.pos in {"N-", "V-", "A-", "D-", "X-"}:
                    p1_content_lems.append(_strip_bare(lem))
            missing_tries = build_missing_content_constraints(
                morphemes,
                existing_constraints=constraints,
                predicted_greek_lemmas=p1_content_lems,
                ccl_lexicon=self.ccl_lexicon,
                lexical_model=self.lexical_model,
                inflector=self.paradigm_inflector,
                tokenizer=self.tokenizer,
                native_gloss_hints=self.native_gloss_hints,
                unaccented=self._unaccented,
                min_prob=0.50,
                max_added=None,
            )
            if missing_tries:
                all_p2_c = [*constraints, *missing_tries]
                word_pen_p2, prefix_pen_p2 = self._build_beam_penalty_fns(
                    morphemes, all_p2_c
                )
                repair_hyps = constrained_ensemble_beam_search(
                    forward_models=self.forward_models,
                    tokenizer=self.tokenizer,
                    src_ids=src_ids,
                    constraints=all_p2_c,
                    beam_size=self.weights.beam_size,
                    max_new_tokens=max_new,
                    min_new_tokens=min_new,
                    len_alpha=self.weights.len_alpha,
                    device=self.device,
                    est_tgt_len=est_tgt,
                    word_penalty_fn=word_pen_p2,
                    prefix_penalty_fn=prefix_pen_p2,
                )
                merged_hyps: list[tuple[list[int], float]] = list(beam_hyps)
                seen_h = {tuple(s) for s, _ in merged_hyps}
                for s, sc in repair_hyps:
                    if tuple(s) not in seen_h:
                        seen_h.add(tuple(s))
                        merged_hyps.append((s, sc))
                merged_hyps = filter_hard_loan_hyps(
                    merged_hyps,
                    constraints,
                    self.tokenizer.word_start_flags(),
                    eos_id=self.tokenizer.eos_id,
                )
                ranked = self._rerank_candidates(
                    src_ids=src_ids,
                    coptic_only_text=coptic_only_text,
                    morphemes=morphemes,
                    beam_hyps=merged_hyps,
                    v1_candidate_text=v1_chunk_greek,
                    tok_to_seq=tok_to_seq,
                )
                # Follow-up repair pass ONLY if Pass-2 newly dropped a target
                # that Pass-1 originally translated (e.g. tail verb `ⲕⲱ`).
                if ranked and ranked[0][1] and len(all_p2_c) <= 8:
                    p2_restored = self._restore_surface_sentence(ranked[0][1])
                    p2_lems = [
                        _strip_bare(lem)
                        for gw in p2_restored.split()
                        for lem, mf, _ in [self.inflector.analyze_surface(gw)[0]]
                        if mf.pos in {"N-", "V-", "A-", "D-", "X-"}
                    ]
                    missing_p3 = build_missing_content_constraints(
                        morphemes,
                        existing_constraints=all_p2_c,
                        predicted_greek_lemmas=p2_lems,
                        ccl_lexicon=self.ccl_lexicon,
                        lexical_model=self.lexical_model,
                        inflector=self.paradigm_inflector,
                        tokenizer=self.tokenizer,
                        native_gloss_hints=self.native_gloss_hints,
                        unaccented=self._unaccented,
                        min_prob=0.50,
                        max_added=2,
                    )
                    if missing_p3:
                        all_p3_c = [*all_p2_c, *missing_p3]
                        word_pen_p3, prefix_pen_p3 = self._build_beam_penalty_fns(
                            morphemes, all_p3_c
                        )
                        p3_hyps = constrained_ensemble_beam_search(
                            forward_models=self.forward_models,
                            tokenizer=self.tokenizer,
                            src_ids=src_ids,
                            constraints=all_p3_c,
                            beam_size=self.weights.beam_size,
                            max_new_tokens=max_new,
                            min_new_tokens=min_new,
                            len_alpha=self.weights.len_alpha,
                            device=self.device,
                            est_tgt_len=est_tgt,
                            word_penalty_fn=word_pen_p3,
                            prefix_penalty_fn=prefix_pen_p3,
                        )
                        for s, sc in p3_hyps:
                            if tuple(s) not in seen_h:
                                seen_h.add(tuple(s))
                                merged_hyps.append((s, sc))
                        merged_hyps = filter_hard_loan_hyps(
                            merged_hyps,
                            constraints,
                            self.tokenizer.word_start_flags(),
                            eos_id=self.tokenizer.eos_id,
                        )
                        ranked = self._rerank_candidates(
                            src_ids=src_ids,
                            coptic_only_text=coptic_only_text,
                            morphemes=morphemes,
                            beam_hyps=merged_hyps,
                            v1_candidate_text=v1_chunk_greek,
                            tok_to_seq=tok_to_seq,
                        )
        if not ranked or not ranked[0][1]:
            # Zero-null guarantee fallback
            fallback_seqs = [m.seq for m in morphemes[:1]] or [0]
            return (
                [
                    GeneratedToken(
                        seq=0,
                        surface="λόγος",
                        lemma="λόγος",
                        morph=GreekMorphFeatures(pos="N-"),
                        coptic_seqs=fallback_seqs,
                        provenance="N",
                        confidence=0.1,
                    )
                ],
                ["λόγος"],
            )

        best_ids, _best_txt, best_score = ranked[0]
        attn = self._extract_cross_attention(src_ids, best_ids)
        word_groups = self.tokenizer.piece_word_groups(best_ids)
        chunk_seqs = [m.seq for m in morphemes if m.pos != "PUNCT"] or [
            m.seq for m in morphemes
        ]
        n_words = max(len(word_groups), 1)

        # Pre-analyze secondary beams (+ v1 draft backup) for Top-3 alternatives
        alt_beams: list[list[tuple[str, str, str, float]]] = []
        for _a_ids, a_txt, a_sc in ranked[1:6]:
            a_words: list[tuple[str, str, str, float]] = []
            for aw in self._restore_surface_sentence(a_txt).split():
                lem, mf, _ = self.inflector.analyze_surface(aw)[0]
                a_words.append((aw, lem, mf.signature(), round(a_sc, 3)))
            if a_words:
                alt_beams.append(a_words)
        if v1_chunk_greek:
            v1_words: list[tuple[str, str, str, float]] = []
            for vw in normalize_greek_target(v1_chunk_greek).split():
                lem, mf, _ = self.inflector.analyze_surface(vw)[0]
                v1_words.append((vw, lem, mf.signature(), round(best_score - 0.5, 3)))
            if v1_words:
                alt_beams.append(v1_words)

        loan_bare_set: set[str] = set()
        loan_surface_to_meta: dict[str, tuple[str, str]] = {}
        for m in morphemes:
            g_hint = resolve_morpheme_loan_hint(m, self.ccl_lexicon)
            if g_hint:
                loan_bare_set.add(_strip_bare(g_hint))
                g_pos_h = (
                    "V-"
                    if m.pos in {"V", "VSTAT", "VIMP"}
                    else "N-"
                    if m.pos in {"N", "NPROP"}
                    else "A-"
                    if m.pos == "ADJ"
                    else "D-"
                )
                for form in self.paradigm_inflector.all_inflected_forms(
                    g_hint, pos=g_pos_h, max_forms=64
                ):
                    loan_surface_to_meta.setdefault(
                        _strip_bare(form), (g_hint, g_pos_h)
                    )

        # Precompute per-source-column mean attention to suppress Transformer sinks
        n_rows = len(attn)
        n_cols = len(attn[0]) if attn else 0
        col_mean: list[float] = [0.0] * n_cols
        if n_rows > 0 and n_cols > 0:
            for r in attn:
                for c_i, val in enumerate(r):
                    col_mean[c_i] += val
            inv_r = 1.0 / n_rows
            col_mean = [v * inv_r for v in col_mean]

        seq_to_rel: dict[int, float] = {
            sq: (idx + 0.5) / max(len(chunk_seqs), 1)
            for idx, sq in enumerate(chunk_seqs)
        }

        norm_words = [self._norm_model_greek(wr) for wr, _piece_pos in word_groups]
        bare_for_ctx = [_bare_key(w) for w in norm_words]
        prev_acc = "BOS"

        out_tokens: list[GeneratedToken] = []
        for w_idx, ((_word_raw, piece_positions), model_word) in enumerate(
            zip(word_groups, norm_words, strict=True)
        ):
            if not model_word:
                continue
            if self._unaccented and self.accent_restorer is not None:
                prev_b = (
                    bare_for_ctx[w_idx - 1]
                    if w_idx > 0 and bare_for_ctx[w_idx - 1]
                    else "<BOS>"
                )
                next_b = (
                    bare_for_ctx[w_idx + 1]
                    if w_idx + 1 < len(bare_for_ctx) and bare_for_ctx[w_idx + 1]
                    else "<EOS>"
                )
                word = self.accent_restorer.restore_word(
                    model_word,
                    prev_bare=prev_b,
                    next_bare=next_b,
                    prev_accent=prev_acc,
                )
                prev_acc = _accent_class(word)
            else:
                word = normalize_greek_target(model_word)
            if not word:
                continue
            # De-sinked cross-attention with gentle monotonic prior -> Top-1 Coptic seq
            w_frac = (w_idx + 0.5) / n_words
            seq_mass: dict[int, float] = {}
            for t_pos in piece_positions:
                row_idx = (
                    t_pos - 1
                )  # target token at pos `t_pos` was predicted at `t_pos - 1`
                if 0 <= row_idx < len(attn):
                    row = attn[row_idx]
                    for s_pos, weight in enumerate(row):
                        if s_pos < len(tok_to_seq) and tok_to_seq[s_pos] is not None:
                            c_sq = tok_to_seq[s_pos]
                            if c_sq is not None:
                                excess = max(
                                    0.0, float(weight) - 0.75 * col_mean[s_pos]
                                )
                                if excess <= 0.0:
                                    continue
                                s_frac = seq_to_rel.get(c_sq, w_frac)
                                mono_w = math.exp(
                                    -0.5 * ((s_frac - w_frac) / 0.30) ** 2
                                )
                                seq_mass[c_sq] = (
                                    seq_mass.get(c_sq, 0.0) + excess * mono_w
                                )
            if seq_mass:
                top_seqs = [
                    sorted(seq_mass.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                ]
            else:
                # Monotonic proportional fallback strictly inside this chunk's seqs
                frac_idx = min(
                    len(chunk_seqs) - 1,
                    int((w_idx / n_words) * len(chunk_seqs)),
                )
                top_seqs = [chunk_seqs[frac_idx]]

            lem, mf, _lp = self.inflector.analyze_surface(word)[0]
            if mf.pos == "UNK":
                loan_meta = loan_surface_to_meta.get(_strip_bare(word))
                if loan_meta is not None:
                    lem, loan_pos = loan_meta
                    mf = GreekMorphFeatures(pos=loan_pos)
                else:
                    # Keep unparseable non-loan surfaces honestly marked UNK
                    # rather than masking them as N-.
                    mf = GreekMorphFeatures(pos="UNK")
                    lem = word.lower()
            prov = "L" if _strip_bare(lem) in loan_bare_set else "N"

            # Gather up to 3 distinct alternative lemmas from sibling beams
            rel_pos = w_idx / n_words
            alts: list[TokenAlternative] = []
            seen_alt_lem = {_strip_bare(lem)}
            for ab in alt_beams:
                ab_n = max(len(ab), 1)
                for a_i, (a_surf, a_lem, a_sig, a_sc) in enumerate(ab):
                    if abs((a_i / ab_n) - rel_pos) <= 0.25:
                        bl = _strip_bare(a_lem)
                        if bl not in seen_alt_lem and a_sig.startswith(
                            ("N-", "V-", "A-", "D-")
                        ):
                            seen_alt_lem.add(bl)
                            alts.append(
                                TokenAlternative(
                                    surface=a_surf,
                                    lemma=a_lem,
                                    morph_sig=a_sig,
                                    score=a_sc,
                                    provenance="N",
                                )
                            )
                            if len(alts) >= 3:
                                break
                if len(alts) >= 3:
                    break

            out_tokens.append(
                GeneratedToken(
                    seq=w_idx,
                    surface=word,
                    lemma=lem,
                    morph=mf,
                    coptic_seqs=sorted(top_seqs),
                    provenance=prov,
                    confidence=round(
                        1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, best_score)))), 4
                    ),
                    alternatives=alts,
                )
            )

        if self.weights.multi_morpheme_align and out_tokens:
            prim = [t.coptic_seqs[0] for t in out_tokens if t.coptic_seqs]
            if len(prim) == len(out_tokens):
                spans = assign_bound_group_coptic_seqs(prim, morphemes)
                out_tokens = [
                    tok.model_copy(update={"coptic_seqs": sp})
                    for tok, sp in zip(out_tokens, spans, strict=True)
                ]

        alt_strings = [
            self._restore_surface_sentence(txt) for _, txt, _ in ranked[:3] if txt
        ]
        return out_tokens, alt_strings

    def _v1_draft_for_chunk(
        self, unit: CopticUnit, ch_ms: Sequence[CopticMorpheme]
    ) -> str | None:
        if self.v1_decoder is None:
            return None
        if len(ch_ms) == len(unit.morphemes):
            return normalize_greek_target(self.v1_decoder.decode_unit(unit).greek_text)
        sub_unit = make_sub_coptic_unit(
            ch_ms, unit_id=unit.unit_id, source_id=unit.source_id
        )
        return normalize_greek_target(self.v1_decoder.decode_unit(sub_unit).greek_text)

    def decode_unit(self, unit: CopticUnit) -> GeneratedUnit:
        """Deterministically reconstruct Koine Greek for one Coptic unit."""
        chunks = split_coptic_unit_for_neural(
            unit,
            max_morphemes=self.weights.max_morphemes,
            min_tail_morphemes=self.weights.min_tail_morphemes,
        )
        all_tokens: list[GeneratedToken] = []
        all_alts: list[str] = []
        for ch_ms in chunks:
            v1_greek = self._v1_draft_for_chunk(unit, ch_ms)
            ch_toks, ch_alts = self._decode_chunk(
                ch_ms,
                domain_tag="<DOM_NT>",
                v1_chunk_greek=v1_greek,
            )
            for tok in ch_toks:
                all_tokens.append(tok.model_copy(update={"seq": len(all_tokens)}))
            if ch_alts:
                all_alts.append(ch_alts[0])

        greek_text = " ".join(t.surface for t in all_tokens)
        return GeneratedUnit(
            unit_id=unit.unit_id,
            coptic_source_id=unit.source_id,
            coptic_text=unit.normalized_text,
            greek_text=greek_text,
            tokens=all_tokens,
            clause_alternatives=all_alts[:4],
        )

    def decode_corpus(
        self, units: Sequence[CopticUnit], *, model_id: str = "model-v2"
    ) -> GeneratedCorpus:
        """Deterministically reconstruct every unit in ``units``."""
        return GeneratedCorpus(
            id=f"thomas-generated-{model_id.removeprefix('model-')}",
            title=f"Generated Koine Greek for Gospel of Thomas ({model_id}, Cat C)",
            model_id=model_id,
            units=[self.decode_unit(u) for u in units],
        )


def _seq_lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    """Order-preserving Longest Common Subsequence length for dev rerank tuning."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[len(b)]


def tune_rerank_weights_on_dev(
    *,
    decoder: NeuralKoineDecoder,
    dev_units: Sequence[tuple[CopticUnit, GreekUnit]],
    tune_faithfulness: bool = False,
) -> RerankWeights:
    """Grid-search noisy-channel rerank weights via ordered LCS on 51-verse NT DEV."""
    content_pos = {"N-", "V-", "A-", "D-"}
    # Row: (t_len, fwd, rev, c_len, lm, lb, cb, bip, attn_c, under_p,
    #       lm_lcs, lm_p, sf_lcs, sf_p)
    cached: list[
        tuple[
            list[
                tuple[
                    int,
                    float,
                    float,
                    int,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    int,
                    int,
                    int,
                    int,
                ]
            ],
            int,
            int,
        ]
    ] = []

    w_base = decoder.weights
    for c_unit, g_unit in dev_units:
        ms = list(c_unit.morphemes)
        if not ms or len(ms) > 96:
            continue
        coptic_only, _ = serialize_coptic_morphemes(
            ms,
            ccl_lexicon=decoder.ccl_lexicon,
            domain_tag="<DOM_NT>",
            greekify=decoder._greekify,
            unaccented_hints=decoder._unaccented,
            native_gloss_hints=decoder.native_gloss_hints,
            canonicalize_preps=decoder._canonicalize_preps,
        )
        v1_t = decoder._v1_draft_for_chunk(c_unit, ms)
        src_text, char_spans = serialize_coptic_morphemes(
            ms,
            ccl_lexicon=decoder.ccl_lexicon,
            domain_tag="<DOM_NT>",
            v1_draft=v1_t if w_base.use_v1_draft else None,
            greekify=decoder._greekify,
            unaccented_hints=decoder._unaccented,
            native_gloss_hints=decoder.native_gloss_hints,
            target_tag=decoder._tgt_grc_tag,
            canonicalize_preps=decoder._canonicalize_preps,
        )
        src_ids, tok_to_seq = decoder.tokenizer.encode_with_coptic_seq_map(
            src_text, char_spans
        )
        n_np = sum(1 for m in ms if m.pos != "PUNCT")
        min_new = max(2, int(n_np * 0.45)) if n_np >= 5 else 1
        max_new = min(220, max(24, int(len(ms) * 3.0) + 16))
        est_tgt = max(int(n_np * 0.85), 8)
        if w_base.use_constrained_beam:
            c_tries = build_chunk_lexical_constraints(
                ms,
                ccl_lexicon=decoder.ccl_lexicon,
                inflector=decoder.paradigm_inflector,
                tokenizer=decoder.tokenizer,
                lexical_model=decoder.lexical_model,
                unaccented=decoder._unaccented,
            )
        else:
            c_tries = []
        wp_fn, pp_fn = decoder._build_beam_penalty_fns(ms, c_tries)
        if c_tries:
            beam_hyps = constrained_ensemble_beam_search(
                forward_models=decoder.forward_models,
                tokenizer=decoder.tokenizer,
                src_ids=src_ids,
                constraints=c_tries,
                beam_size=w_base.beam_size,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                len_alpha=w_base.len_alpha,
                device=decoder.device,
                est_tgt_len=est_tgt,
                word_penalty_fn=wp_fn,
                prefix_penalty_fn=pp_fn,
            )
        else:
            beam_hyps = decoder._ensemble_beam_search(
                src_ids,
                beam_size=w_base.beam_size,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                len_alpha=w_base.len_alpha,
                word_penalty_fn=wp_fn,
                prefix_penalty_fn=pp_fn,
            )
        if (
            w_base.use_coverage_repair
            and decoder.lexical_model is not None
            and beam_hyps
        ):
            p1_txt = decoder._norm_model_greek(
                decoder.tokenizer.decode(beam_hyps[0][0])
            )
            p1_rest = decoder._restore_surface_sentence(p1_txt) if p1_txt else ""
            p1_lems: list[str] = []
            for gw in p1_rest.split():
                lem, mf, _ = decoder.inflector.analyze_surface(gw)[0]
                if mf.pos in {"N-", "V-", "A-", "D-", "X-"}:
                    p1_lems.append(_strip_bare(lem))
            miss_tries = build_missing_content_constraints(
                ms,
                existing_constraints=c_tries,
                predicted_greek_lemmas=p1_lems,
                ccl_lexicon=decoder.ccl_lexicon,
                lexical_model=decoder.lexical_model,
                inflector=decoder.paradigm_inflector,
                tokenizer=decoder.tokenizer,
                native_gloss_hints=decoder.native_gloss_hints,
                unaccented=decoder._unaccented,
                min_prob=0.50,
                max_added=None,
            )
            if miss_tries:
                all_p2_t = [*c_tries, *miss_tries]
                wp_p2, pp_p2 = decoder._build_beam_penalty_fns(ms, all_p2_t)
                rep_hyps = constrained_ensemble_beam_search(
                    forward_models=decoder.forward_models,
                    tokenizer=decoder.tokenizer,
                    src_ids=src_ids,
                    constraints=all_p2_t,
                    beam_size=w_base.beam_size,
                    max_new_tokens=max_new,
                    min_new_tokens=min_new,
                    len_alpha=w_base.len_alpha,
                    device=decoder.device,
                    est_tgt_len=est_tgt,
                    word_penalty_fn=wp_p2,
                    prefix_penalty_fn=pp_p2,
                )
                seen_s = {tuple(s) for s, _ in beam_hyps}
                for s, sc in rep_hyps:
                    if tuple(s) not in seen_s:
                        seen_s.add(tuple(s))
                        beam_hyps.append((s, sc))
                beam_hyps = filter_hard_loan_hyps(
                    beam_hyps,
                    c_tries,
                    decoder.tokenizer.word_start_flags(),
                    eos_id=decoder.tokenizer.eos_id,
                )
        c_ids = [s for s, _ in beam_hyps]
        c_txts = [decoder._norm_model_greek(decoder.tokenizer.decode(s)) for s in c_ids]
        if v1_t and w_base.include_v1_candidate:
            v1_m = decoder._norm_model_greek(v1_t)
            if v1_m and v1_m not in c_txts:
                c_ids.append(decoder.tokenizer.encode(v1_m))
                c_txts.append(v1_m)

        fwd_raw = decoder._score_seq2seq(decoder.forward_models, src_ids, c_ids)
        c_tgt_ids = decoder.tokenizer.encode(coptic_only)
        c_len = max(len(c_tgt_ids) - 1, 1)
        rev_raw = [0.0 for _ in c_ids]
        if decoder.reverse_model is not None:
            for idx, gt in enumerate(c_txts):
                if gt:
                    gs = decoder.tokenizer.encode(f"<REV_COPTIC> {gt}")
                    rev_raw[idx] = decoder._score_seq2seq(
                        [decoder.reverse_model], gs, [c_tgt_ids]
                    )[0]
        lm_raw = [0.0 for _ in c_ids]
        if decoder.koine_lm is not None:
            lm_raw = decoder.koine_lm.score_token_seqs(c_ids, device=decoder.device)

        exp_loans: Counter[str] = Counter()
        for m in ms:
            if (gh := resolve_morpheme_loan_hint(m, decoder.ccl_lexicon)) is not None:
                exp_loans[_strip_bare(gh)] += 1
        n_exp_loans = sum(exp_loans.values())
        draft_lems: set[str] = set()
        if v1_t:
            for dw in normalize_greek_target(v1_t).split():
                d_lem, d_mf, _ = decoder.inflector.analyze_surface(dw)[0]
                if d_mf.pos in content_pos:
                    draft_lems.add(_strip_bare(d_lem))

        coptic_cand_sets = (
            build_coptic_content_candidate_sets(
                ms,
                ccl_lexicon=decoder.ccl_lexicon,
                lexical_model=decoder.lexical_model,
                native_gloss_hints=decoder.native_gloss_hints,
            )
            if tune_faithfulness
            else []
        )
        content_seqs = [m.seq for m in ms if m.pos in CONTENT_COPTIC_POS]
        n_cop_content = len(content_seqs)

        g_lem_seq = [
            _strip_bare(t.lemma) for t in g_unit.tokens if t.morph.pos in content_pos
        ]
        g_surf_seq = [
            norm_s
            for t in g_unit.tokens
            if t.morph.pos != "X-"
            and (norm_s := normalize_greek_target(t.normalized or t.surface).lower())
        ]

        cand_rows: list[
            tuple[
                int,
                float,
                float,
                int,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                int,
                int,
                int,
                int,
            ]
        ] = []
        for idx, (seq, gt) in enumerate(zip(c_ids, c_txts, strict=True)):
            t_len = max(len(seq) - 1, 1)
            restored_gt = decoder._restore_surface_sentence(gt) if gt else ""
            lb = 0.0
            cb = 0.0
            bip = 0.0
            attn_c = 0.0
            under_p = 0.0
            bare_pen = 0.0
            p_lem_seq: list[str] = []
            p_surf_seq = restored_gt.lower().split() if restored_gt else []
            if restored_gt:
                gl: set[str] = set()
                gl_counts: Counter[str] = Counter()
                r_words = restored_gt.split()
                if w_base.use_constrained_beam:
                    bare_pen += 0.8 * count_spurious_adjacent_word_repeats(r_words, ms)
                from zetodos_thomas.ingest.normalize import normalize_coptic

                known_cand_bare: set[str] = set(exp_loans)
                for _sq, cset in coptic_cand_sets:
                    known_cand_bare.update(cset)
                chunk_has_nprop = any(m.pos == "NPROP" for m in ms)
                chunk_lems_t = {
                    normalize_coptic(m.lemma or m.norm).replace("\u2cff", "")
                    for m in ms
                }
                forbid_hem_t = "ⲉϩⲉ" in chunk_lems_t and "ϩⲟⲟⲩ" not in chunk_lems_t
                for w in r_words:
                    is_bc = (
                        not decoder._unaccented
                        and w_base.use_constrained_beam
                        and is_bare_coptic_copy_word(w)
                    )
                    if is_bc:
                        bare_pen += 1.5
                    lem, mf, _ = decoder.inflector.analyze_surface(w)[0]
                    bl = _strip_bare(lem)
                    bs = _strip_bare(w)
                    if w_base.use_constrained_beam and forbid_hem_t and bl == "ημερα":
                        bare_pen += 2.0
                    elif (
                        w_base.use_constrained_beam
                        and not is_bc
                        and mf.pos == "UNK"
                        and bl not in known_cand_bare
                        and bs not in known_cand_bare
                        and not (w[0].isupper() and chunk_has_nprop)
                    ):
                        bare_pen += 0.80
                    if not is_bc:
                        gl.add(bl)
                        gl.add(bs)
                        gl_counts[bl] += 1
                        if bs != bl:
                            gl_counts[bs] += 1
                    if mf.pos in {"N-", "V-", "A-", "D-", "X-"}:
                        p_lem_seq.append(bl)
                if n_exp_loans:
                    lb = (
                        sum(min(cnt, gl_counts[el]) for el, cnt in exp_loans.items())
                        / n_exp_loans
                    )
                if draft_lems:
                    cb = sum(1 for dl in draft_lems if dl in gl) / len(draft_lems)
                if coptic_cand_sets:
                    bip = bipartite_lexicon_coverage(coptic_cand_sets, p_lem_seq)
                if tune_faithfulness and n_cop_content > 0:
                    min_exp = max(1.0, 0.85 * n_cop_content)
                    max_exp = max(n_cop_content + 3.0, 1.30 * n_cop_content)
                    if len(p_lem_seq) < min_exp:
                        under_p = (min_exp - len(p_lem_seq)) / n_cop_content
                    elif len(p_lem_seq) > max_exp:
                        under_p = (len(p_lem_seq) - max_exp) / n_cop_content
            if tune_faithfulness and len(seq) >= 2 and content_seqs:
                c_attn = decoder._extract_cross_attention(src_ids, seq)
                attn_c = wu_attention_coverage_penalty(
                    c_attn, tok_to_seq, content_seqs, tau=0.45
                )
            cand_rows.append(
                (
                    t_len,
                    fwd_raw[idx],
                    rev_raw[idx],
                    c_len,
                    lm_raw[idx],
                    lb,
                    cb,
                    bip,
                    attn_c,
                    under_p,
                    bare_pen,
                    _seq_lcs_len(p_lem_seq, g_lem_seq),
                    len(p_lem_seq),
                    _seq_lcs_len(p_surf_seq, g_surf_seq),
                    len(p_surf_seq),
                )
            )
        cached.append((cand_rows, len(g_lem_seq), len(g_surf_seq)))

    if not tune_faithfulness:
        alpha_grid = (1.0, 0.9, 0.8, 1.1)
        rev_grid = (
            (0.25, 0.5, 0.0, 0.75) if decoder.reverse_model is not None else (0.0,)
        )
        lm_grid = (0.15, 0.30, 0.0, 0.45) if decoder.koine_lm is not None else (0.0,)
        loan_grid = (0.8, 0.4, 1.2, 0.0)
        cov_grid = (0.0, 0.25, 0.5, 0.8) if w_base.use_v1_draft else (0.0,)
        bip_grid = (0.0,)
        attn_grid = (0.0,)
        under_grid = (0.0,)
    else:
        alpha_grid = (0.85, 0.95, 1.0)
        rev_grid = (0.35, 0.65, 0.95) if decoder.reverse_model is not None else (0.0,)
        lm_grid = (0.0, 0.10, 0.20) if decoder.koine_lm is not None else (0.0,)
        loan_grid = (1.2, 1.8)
        cov_grid = (0.0, 0.4) if w_base.use_v1_draft else (0.0,)
        bip_grid = (0.6, 1.2, 1.8)
        attn_grid = (0.15, 0.35, 0.60)
        under_grid = (0.5, 1.2)

    best_obj = -1.0
    best_w = w_base
    for alpha in alpha_grid:
        for wr in rev_grid:
            for wl in lm_grid:
                for wln in loan_grid:
                    for wcv in cov_grid:
                        for wbip in bip_grid:
                            for wat in attn_grid:
                                for wun in under_grid:
                                    lm_m = lm_p = lm_g = 0
                                    sf_m = sf_p = sf_g = 0
                                    for cand_rows, n_g_lem, n_g_surf in cached:
                                        best_r = max(
                                            cand_rows,
                                            key=lambda r: (
                                                r[1] / (r[0] ** alpha)
                                                + wr * (r[2] / (r[3] ** alpha))
                                                + wl * (r[4] / (r[0] ** alpha))
                                                + wln * r[5]
                                                + wcv * r[6]
                                                + wbip * r[7]
                                                + wat * r[8]
                                                - wun * r[9]
                                                - r[10]
                                            ),
                                        )
                                        lm_m += best_r[11]
                                        lm_p += best_r[12]
                                        lm_g += n_g_lem
                                        sf_m += best_r[13]
                                        sf_p += best_r[14]
                                        sf_g += n_g_surf
                                    lf1 = 2 * lm_m / max(lm_p + lm_g, 1)
                                    sf1 = 2 * sf_m / max(sf_p + sf_g, 1)
                                    if tune_faithfulness:
                                        lrec = lm_m / max(lm_g, 1)
                                        ratio = lm_p / max(lm_g, 1)
                                        len_fid = max(0.0, 1.0 - abs(1.0 - ratio))
                                        obj = (
                                            0.45 * lf1
                                            + 0.25 * sf1
                                            + 0.20 * lrec
                                            + 0.10 * len_fid
                                        )
                                    else:
                                        obj = 0.60 * lf1 + 0.40 * sf1
                                    if obj > best_obj:
                                        best_obj = obj
                                        best_w = w_base.model_copy(
                                            update={
                                                "len_alpha": alpha,
                                                "w_rev": wr,
                                                "w_lm": wl,
                                                "w_loan": wln,
                                                "w_cov": wcv,
                                                "w_bipartite_cov": wbip,
                                                "w_attn_cov": wat,
                                                "w_under_len": wun,
                                            }
                                        )
    decoder.weights = best_w
    return best_w
