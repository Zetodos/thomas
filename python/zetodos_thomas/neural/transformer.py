# SPDX-License-Identifier: Apache-2.0
"""Compact Pre-LN Encoder-Decoder Transformer & Causal Koine LM (from scratch)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


class Seq2SeqConfig(BaseModel):
    """Serializable architecture configuration for ``KoineSeq2SeqTransformer``."""

    vocab_size: int = 8000
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    num_enc_layers: int = 5
    num_dec_layers: int = 5
    dropout: float = 0.30
    attn_dropout: float = 0.15
    max_len: int = 512
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    greekify_source: bool = False
    unaccented_target: bool = False
    use_gloss_hints: bool = False
    use_target_lang_tag: bool = False
    canonicalize_preps: bool = False


class KoineCausalLMConfig(BaseModel):
    """Serializable configuration for standalone from-scratch Koine Causal LM."""

    vocab_size: int = 8000
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    num_layers: int = 4
    dropout: float = 0.25
    attn_dropout: float = 0.15
    max_len: int = 512
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unaccented_target: bool = False


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        pe_buf = self.get_buffer("pe")
        return x + pe_buf[:, : x.size(1), :].to(dtype=x.dtype)


class _PreLNEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, src_key_padding_mask: Tensor | None) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class _PreLNDecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        *,
        tgt_causal_mask: Tensor,
        tgt_key_padding_mask: Tensor | None,
        memory_key_padding_mask: Tensor | None,
        need_cross_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        h = self.norm1(x)
        sa_out, _ = self.self_attn(
            h,
            h,
            h,
            attn_mask=tgt_causal_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(sa_out)

        h2 = self.norm2(x)
        ca_out, ca_weights = self.cross_attn(
            h2,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_cross_weights,
            average_attn_weights=True,
        )
        x = x + self.drop2(ca_out)
        x = x + self.ffn(self.norm3(x))
        return x, ca_weights


class KoineSeq2SeqTransformer(nn.Module):
    """From-scratch Pre-LN Encoder-Decoder Transformer with cross-attn export."""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.scale = math.sqrt(config.d_model)
        self.embed = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.pos_enc = _SinusoidalPositionalEncoding(
            config.d_model, max_len=max(config.max_len * 2, 1024)
        )
        self.embed_drop = nn.Dropout(config.dropout)
        self.encoder_layers = nn.ModuleList(
            [
                _PreLNEncoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.attn_dropout,
                )
                for _ in range(config.num_enc_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(config.d_model)
        self.decoder_layers = nn.ModuleList(
            [
                _PreLNDecoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.attn_dropout,
                )
                for _ in range(config.num_dec_layers)
            ]
        )
        self.dec_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying between shared BPE embedding and output projection
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=self.config.d_model**-0.5)
                if mod.padding_idx is not None:
                    with torch.no_grad():
                        mod.weight[mod.padding_idx].zero_()

    def encode(self, src_ids: Tensor) -> tuple[Tensor, Tensor]:
        src_pad_mask = src_ids.eq(self.config.pad_id)
        x = self.embed_drop(self.pos_enc(self.embed(src_ids) * self.scale))
        for layer in self.encoder_layers:
            x = layer(x, src_pad_mask)
        return self.enc_norm(x), src_pad_mask

    def decode(
        self,
        tgt_ids: Tensor,
        memory: Tensor,
        src_pad_mask: Tensor,
        *,
        return_cross_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        t_len = tgt_ids.size(1)
        tgt_pad_mask = tgt_ids.eq(self.config.pad_id)
        causal_mask = torch.triu(
            torch.ones((t_len, t_len), device=tgt_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.embed_drop(self.pos_enc(self.embed(tgt_ids) * self.scale))
        n_dec = len(self.decoder_layers)
        attn_accum: list[Tensor] = []
        for idx, layer in enumerate(self.decoder_layers):
            want_w = return_cross_attn and idx >= max(0, n_dec - 2)
            x, w = layer(
                x,
                memory,
                tgt_causal_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
                need_cross_weights=want_w,
            )
            if want_w and w is not None:
                attn_accum.append(w)
        logits = self.lm_head(self.dec_norm(x))
        avg_attn = torch.stack(attn_accum, dim=0).mean(dim=0) if attn_accum else None
        return logits, avg_attn

    def forward(
        self,
        src_ids: Tensor,
        tgt_in_ids: Tensor,
        *,
        return_cross_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        memory, src_pad_mask = self.encode(src_ids)
        return self.decode(
            tgt_in_ids,
            memory,
            src_pad_mask,
            return_cross_attn=return_cross_attn,
        )

    def save_safetensors(self, path: Path, *, fp16: bool = True) -> None:
        """Persist weights safely (de-tying lm_head.weight for safetensors)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_state = self.state_dict()
        tensors: dict[str, Tensor] = {}
        for k, v in raw_state.items():
            if k == "lm_head.weight":
                continue  # tied to embed.weight
            t = v.detach().cpu().contiguous()
            if fp16 and t.is_floating_point():
                t = t.half()
            tensors[k] = t
        save_file(tensors, str(path))

    @classmethod
    def load_safetensors(
        cls,
        path: Path,
        config: Seq2SeqConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> KoineSeq2SeqTransformer:
        model = cls(config)
        loaded = load_file(str(path), device="cpu")
        float_state = {
            k: (v.float() if v.is_floating_point() else v) for k, v in loaded.items()
        }
        float_state["lm_head.weight"] = float_state["embed.weight"]
        model.load_state_dict(float_state, strict=True)
        model.lm_head.weight = model.embed.weight
        model.to(device)
        model.eval()
        return model


class _PreLNCausalBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: Tensor, causal_mask: Tensor, key_padding_mask: Tensor | None
    ) -> Tensor:
        h = self.norm1(x)
        out, _ = self.self_attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(out)
        x = x + self.ffn(self.norm2(x))
        return x


class KoineCausalLM(nn.Module):
    """Standalone from-scratch Koine Greek causal LM for noisy-channel reranking."""

    def __init__(self, config: KoineCausalLMConfig) -> None:
        super().__init__()
        self.config = config
        self.scale = math.sqrt(config.d_model)
        self.embed = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.pos_enc = _SinusoidalPositionalEncoding(
            config.d_model, max_len=max(config.max_len * 2, 1024)
        )
        self.embed_drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                _PreLNCausalBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.attn_dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=self.config.d_model**-0.5)
                if mod.padding_idx is not None:
                    with torch.no_grad():
                        mod.weight[mod.padding_idx].zero_()

    def forward(self, input_ids: Tensor) -> Tensor:
        t_len = input_ids.size(1)
        pad_mask = input_ids.eq(self.config.pad_id)
        causal_mask = torch.triu(
            torch.ones((t_len, t_len), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.embed_drop(self.pos_enc(self.embed(input_ids) * self.scale))
        for layer in self.layers:
            x = layer(x, causal_mask, pad_mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def score_token_seqs(
        self, seqs: list[list[int]], *, device: torch.device | str = "cpu"
    ) -> list[float]:
        """Return total log-probability for each ``[BOS, ..., EOS]`` token sequence."""
        if not seqs:
            return []
        max_l = max(len(s) for s in seqs)
        if max_l < 2:
            return [0.0 for _ in seqs]
        batch = torch.full(
            (len(seqs), max_l), self.config.pad_id, dtype=torch.long, device=device
        )
        for idx, seq in enumerate(seqs):
            batch[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        logits = self.forward(batch[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        targets = batch[:, 1:]
        tok_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        mask = targets.ne(self.config.pad_id).float()
        totals = (tok_lp * mask).sum(dim=1)
        return [float(x) for x in totals.cpu().tolist()]

    def save_safetensors(self, path: Path, *, fp16: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, Tensor] = {}
        for k, v in self.state_dict().items():
            if k == "lm_head.weight":
                continue
            t = v.detach().cpu().contiguous()
            if fp16 and t.is_floating_point():
                t = t.half()
            tensors[k] = t
        save_file(tensors, str(path))

    @classmethod
    def load_safetensors(
        cls,
        path: Path,
        config: KoineCausalLMConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> KoineCausalLM:
        model = cls(config)
        loaded = load_file(str(path), device="cpu")
        float_state = {
            k: (v.float() if v.is_floating_point() else v) for k, v in loaded.items()
        }
        float_state["lm_head.weight"] = float_state["embed.weight"]
        model.load_state_dict(float_state, strict=True)
        model.lm_head.weight = model.embed.weight
        model.to(device)
        model.eval()
        return model
