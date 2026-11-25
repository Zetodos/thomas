# SPDX-License-Identifier: Apache-2.0
"""Deterministic neural ensemble beam decoder with noisy-channel reranking."""

from __future__ import annotations

import dataclasses
import json
import math
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor

from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.neural.accent_restorer import (
    KoineAccentRestorer,
    _accent_class,
    _bare_key,
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
    len_alpha: float = 0.9
    beam_size: int = 8
    use_v1_draft: bool = False
    include_v1_candidate: bool = False

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
    if len(seq) < 5:
        return False
    tri = (seq[-2], seq[-1], next_id)
    for i in range(len(seq) - 2):
        if (seq[i], seq[i + 1], seq[i + 2]) == tri:
            return True
    return False


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
    ) -> None:
        self.tokenizer = tokenizer
        self.forward_models = forward_models
        self.ccl_lexicon = ccl_lexicon
        self.inflector = analyzer_inflector
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

        need_v1 = weights.use_v1_draft or weights.include_v1_candidate
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
    ) -> list[tuple[list[int], float]]:
        """Deterministic probability-mixture ensemble beam search."""
        src_t = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        encoded = [m.encode(src_t) for m in self.forward_models]
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        n_models = len(self.forward_models)

        active: list[tuple[list[int], float]] = [([bos_id], 0.0)]
        finished: list[tuple[list[int], float]] = []

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
                for val, tid in zip(top_vals_cpu[b_i], top_idx_cpu[b_i], strict=True):
                    itid = int(tid)
                    if itid == self.tokenizer.pad_id or itid == bos_id:
                        continue
                    if itid != eos_id and _has_trigram_repeat(seq, itid):
                        continue
                    new_seq = [*seq, itid]
                    new_score = score + float(val)
                    if itid == eos_id:
                        if seq_len < min_new_tokens:
                            continue
                        norm_sc = new_score / (max(seq_len, 1) ** len_alpha)
                        finished.append((new_seq, norm_sc))
                    else:
                        next_cands.append((new_seq, new_score))

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
    ) -> list[tuple[list[int], str, float]]:
        """Score & sort candidate hypotheses with noisy-channel reranking."""
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

        # Expected Greek loanwords from CCL/DDGLC & draft content-lemma coverage set
        expected_loans: set[str] = set()
        for m in morphemes:
            g_hint = resolve_morpheme_loan_hint(m, self.ccl_lexicon)
            if g_hint:
                expected_loans.add(_strip_bare(g_hint))

        content_pos = {"N-", "V-", "A-", "D-"}
        draft_content_lems: set[str] = set()
        if v1_candidate_text and abs(w.w_cov) > 1e-6:
            for dw in normalize_greek_target(v1_candidate_text).split():
                d_lem, d_mf, _ = self.inflector.analyze_surface(dw)[0]
                if d_mf.pos in content_pos:
                    draft_content_lems.add(_strip_bare(d_lem))

        ranked: list[tuple[list[int], str, float]] = []
        for idx, (seq, g_txt) in enumerate(zip(cands_ids, cands_txt, strict=True)):
            fwd_n = fwd_raw[idx] / (max(len(seq) - 1, 1) ** w.len_alpha)
            loan_bonus = 0.0
            cov_bonus = 0.0
            if (expected_loans or draft_content_lems) and g_txt:
                g_lems = set()
                restored_cand = self._restore_surface_sentence(g_txt)
                for gw in restored_cand.split():
                    p0 = self.inflector.analyze_surface(gw)[0]
                    g_lems.add(_strip_bare(p0[0]))
                    g_lems.add(_strip_bare(gw))
                if expected_loans:
                    hit_n = sum(1 for el in expected_loans if el in g_lems)
                    loan_bonus = hit_n / len(expected_loans)
                if draft_content_lems:
                    cov_n = sum(1 for dl in draft_content_lems if dl in g_lems)
                    cov_bonus = cov_n / len(draft_content_lems)
            total = (
                w.w_fwd * fwd_n
                + w.w_rev * rev_scores[idx]
                + w.w_lm * lm_scores[idx]
                + w.w_loan * loan_bonus
                + w.w_cov * cov_bonus
            )
            ranked.append((seq, g_txt, total))

        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked

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
        )
        src_ids, tok_to_seq = self.tokenizer.encode_with_coptic_seq_map(
            src_text, char_spans
        )
        n_non_punct = sum(1 for m in morphemes if m.pos != "PUNCT")
        min_new = max(2, int(n_non_punct * 0.45)) if n_non_punct >= 5 else 1
        max_new = min(220, max(24, int(len(morphemes) * 3.0) + 16))
        beam_hyps = self._ensemble_beam_search(
            src_ids,
            beam_size=self.weights.beam_size,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            len_alpha=self.weights.len_alpha,
        )
        ranked = self._rerank_candidates(
            src_ids=src_ids,
            coptic_only_text=coptic_only_text,
            morphemes=morphemes,
            beam_hyps=beam_hyps,
            v1_candidate_text=v1_chunk_greek,
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
        for m in morphemes:
            g_hint = resolve_morpheme_loan_hint(m, self.ccl_lexicon)
            if g_hint:
                loan_bare_set.add(_strip_bare(g_hint))

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
                mf = GreekMorphFeatures(pos="N-")
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
        chunks = split_coptic_unit_for_neural(unit, max_morphemes=36)
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
) -> RerankWeights:
    """Grid-search noisy-channel rerank weights via ordered LCS on 51-verse NT DEV."""
    # Precompute candidate sets & unnormalized log-likelihoods once for fast search
    content_pos = {"N-", "V-", "A-", "D-"}
    # Cand row: (t_len, fwd, rev, c_len, lm, lb, cb, lm_lcs, lm_p, sf_lcs, sf_p)
    cached: list[
        tuple[
            list[
                tuple[int, float, float, int, float, float, float, int, int, int, int]
            ],
            int,
            int,
        ]
    ] = []

    w_base = decoder.weights
    for c_unit, g_unit in dev_units:
        ms = list(c_unit.morphemes)
        if not ms or len(ms) > 80:
            continue
        coptic_only, _ = serialize_coptic_morphemes(
            ms,
            ccl_lexicon=decoder.ccl_lexicon,
            domain_tag="<DOM_NT>",
            greekify=decoder._greekify,
            unaccented_hints=decoder._unaccented,
            native_gloss_hints=decoder.native_gloss_hints,
        )
        v1_t = decoder._v1_draft_for_chunk(c_unit, ms)
        src_text, _ = serialize_coptic_morphemes(
            ms,
            ccl_lexicon=decoder.ccl_lexicon,
            domain_tag="<DOM_NT>",
            v1_draft=v1_t if w_base.use_v1_draft else None,
            greekify=decoder._greekify,
            unaccented_hints=decoder._unaccented,
            native_gloss_hints=decoder.native_gloss_hints,
            target_tag=decoder._tgt_grc_tag,
        )
        src_ids = decoder.tokenizer.encode(src_text)
        n_np = sum(1 for m in ms if m.pos != "PUNCT")
        min_new = max(2, int(n_np * 0.45)) if n_np >= 5 else 1
        max_new = min(220, max(24, int(len(ms) * 3.0) + 16))
        beam_hyps = decoder._ensemble_beam_search(
            src_ids,
            beam_size=w_base.beam_size,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            len_alpha=w_base.len_alpha,
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

        exp_loans = {
            _strip_bare(gh)
            for m in ms
            if (gh := resolve_morpheme_loan_hint(m, decoder.ccl_lexicon)) is not None
        }
        draft_lems: set[str] = set()
        if v1_t:
            for dw in normalize_greek_target(v1_t).split():
                d_lem, d_mf, _ = decoder.inflector.analyze_surface(dw)[0]
                if d_mf.pos in content_pos:
                    draft_lems.add(_strip_bare(d_lem))

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
            tuple[int, float, float, int, float, float, float, int, int, int, int]
        ] = []
        for idx, (seq, gt) in enumerate(zip(c_ids, c_txts, strict=True)):
            t_len = max(len(seq) - 1, 1)
            restored_gt = decoder._restore_surface_sentence(gt) if gt else ""
            lb = 0.0
            cb = 0.0
            p_lem_seq: list[str] = []
            p_surf_seq = restored_gt.lower().split() if restored_gt else []
            if restored_gt:
                gl: set[str] = set()
                for w in restored_gt.split():
                    lem, mf, _ = decoder.inflector.analyze_surface(w)[0]
                    bl = _strip_bare(lem)
                    gl.add(bl)
                    gl.add(_strip_bare(w))
                    if mf.pos in content_pos:
                        p_lem_seq.append(bl)
                if exp_loans:
                    lb = sum(1 for el in exp_loans if el in gl) / len(exp_loans)
                if draft_lems:
                    cb = sum(1 for dl in draft_lems if dl in gl) / len(draft_lems)
            cand_rows.append(
                (
                    t_len,
                    fwd_raw[idx],
                    rev_raw[idx],
                    c_len,
                    lm_raw[idx],
                    lb,
                    cb,
                    _seq_lcs_len(p_lem_seq, g_lem_seq),
                    len(p_lem_seq),
                    _seq_lcs_len(p_surf_seq, g_surf_seq),
                    len(p_surf_seq),
                )
            )
        cached.append((cand_rows, len(g_lem_seq), len(g_surf_seq)))

    alpha_grid = (1.0, 0.9, 0.8, 1.1)
    rev_grid = (0.25, 0.5, 0.0, 0.75) if decoder.reverse_model is not None else (0.0,)
    lm_grid = (0.15, 0.30, 0.0, 0.45) if decoder.koine_lm is not None else (0.0,)
    loan_grid = (0.8, 0.4, 1.2, 0.0)
    cov_grid = (0.0, 0.25, 0.5, 0.8) if w_base.use_v1_draft else (0.0,)

    best_obj = -1.0
    best_w = w_base
    for alpha in alpha_grid:
        for wr in rev_grid:
            for wl in lm_grid:
                for wln in loan_grid:
                    for wcv in cov_grid:
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
                                ),
                            )
                            lm_m += best_r[7]
                            lm_p += best_r[8]
                            lm_g += n_g_lem
                            sf_m += best_r[9]
                            sf_p += best_r[10]
                            sf_g += n_g_surf
                        lf1 = 2 * lm_m / max(lm_p + lm_g, 1)
                        sf1 = 2 * sf_m / max(sf_p + sf_g, 1)
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
                                }
                            )
    decoder.weights = best_w
    return best_w
