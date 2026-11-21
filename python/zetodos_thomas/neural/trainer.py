# SPDX-License-Identifier: Apache-2.0
"""MPS/CPU training loops for from-scratch Seq2Seq NMT, Back-Translation & LM."""

from __future__ import annotations

import copy
import gc
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.neural.corpus import (
    ParallelExample,
    normalize_greek_target,
    serialize_coptic_morphemes,
)
from zetodos_thomas.neural.tokenizer import BilingualTokenizer
from zetodos_thomas.neural.transformer import (
    KoineCausalLM,
    KoineCausalLMConfig,
    KoineSeq2SeqTransformer,
    Seq2SeqConfig,
)


def seed_everything(seed: int) -> None:
    """Set deterministic seeds across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve 'auto', 'mps', or 'cpu' cleanly."""
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _clear_mps_cache(device: torch.device) -> None:
    """Release fragmented Apple Metal buffers during long training runs."""
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _quick_chrf(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Lightweight char 1-6 chrF (beta=2) for NT dev checkpoint selection."""
    tot_p = tot_r = 0.0
    n_orders = 6
    for order in range(1, n_orders + 1):
        matched = pred_n = gold_n = 0
        for ref, hyp in zip(refs, hyps, strict=True):
            r_s = ref.replace(" ", "")
            h_s = hyp.replace(" ", "")
            rc = Counter(
                r_s[i : i + order] for i in range(max(0, len(r_s) - order + 1))
            )
            hc = Counter(
                h_s[i : i + order] for i in range(max(0, len(h_s) - order + 1))
            )
            matched += sum((rc & hc).values())
            pred_n += sum(hc.values())
            gold_n += sum(rc.values())
        tot_p += matched / max(pred_n, 1)
        tot_r += matched / max(gold_n, 1)
    avg_p = tot_p / n_orders
    avg_r = tot_r / n_orders
    if avg_p + avg_r <= 0.0:
        return 0.0
    beta2 = 4.0
    return (1.0 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)


def _iter_dynamic_pair_batches(
    pairs: list[tuple[list[int], list[int]]],
    *,
    max_tokens: int,
    max_batch: int,
    pad_id: int,
    rng: random.Random,
    shuffle: bool = True,
) -> list[tuple[Tensor, Tensor]]:
    """Length-bucketed dynamic batching for fast, low-padding Transformer training."""
    indexed = list(pairs)
    if shuffle:
        # Sort with tiny random noise so batches stay tight in length yet varied
        indexed.sort(key=lambda item: len(item[0]) + len(item[1]) + rng.randint(0, 12))
    else:
        indexed.sort(key=lambda item: len(item[0]) + len(item[1]))

    batches_raw: list[list[tuple[list[int], list[int]]]] = []
    cur: list[tuple[list[int], list[int]]] = []
    cur_max_s = 0
    cur_max_t = 0
    for s_ids, t_ids in indexed:
        nxt_s = max(cur_max_s, len(s_ids))
        nxt_t = max(cur_max_t, len(t_ids))
        if cur and (
            (len(cur) + 1) * (nxt_s + nxt_t) > max_tokens or len(cur) >= max_batch
        ):
            batches_raw.append(cur)
            cur = []
            cur_max_s = 0
            cur_max_t = 0
        cur.append((s_ids, t_ids))
        cur_max_s = max(cur_max_s, len(s_ids))
        cur_max_t = max(cur_max_t, len(t_ids))
    if cur:
        batches_raw.append(cur)

    if shuffle:
        rng.shuffle(batches_raw)

    tensors: list[tuple[Tensor, Tensor]] = []
    for batch in batches_raw:
        bsz = len(batch)
        ms = max(len(s) for s, _ in batch)
        mt = max(len(t) for _, t in batch)
        src_t = torch.full((bsz, ms), pad_id, dtype=torch.long)
        tgt_t = torch.full((bsz, mt), pad_id, dtype=torch.long)
        for i, (s, t) in enumerate(batch):
            src_t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            tgt_t[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        tensors.append((src_t, tgt_t))
    return tensors


@torch.no_grad()
def _greedy_decode_batch(
    model: KoineSeq2SeqTransformer,
    src_batch: Tensor,
    *,
    max_new_tokens: int,
) -> list[list[int]]:
    """Fast batched greedy decoding for dev validation & back-translation."""
    model.eval()
    device = src_batch.device
    bsz = src_batch.size(0)
    memory, src_pad_mask = model.encode(src_batch)
    ys = torch.full((bsz, 1), model.config.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model.decode(ys, memory, src_pad_mask, return_cross_attn=False)
        next_tok = logits[:, -1, :].argmax(dim=-1)
        next_tok = torch.where(
            finished,
            torch.full_like(next_tok, model.config.pad_id),
            next_tok,
        )
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | next_tok.eq(model.config.eos_id)
        if bool(finished.all()):
            break

    out: list[list[int]] = []
    for row in ys.cpu().tolist():
        seq: list[int] = []
        for tid in row[1:]:
            if tid == model.config.eos_id or tid == model.config.pad_id:
                break
            seq.append(int(tid))
        out.append(seq)
    return out


def _apply_span_noise(
    ids: list[int],
    *,
    mask_id: int,
    mask_prob: float,
    rng: random.Random,
) -> list[int]:
    """mBART-style contiguous span masking on interior token IDs."""
    if len(ids) <= 4:
        return list(ids)
    bos, body, eos = ids[0], ids[1:-1], ids[-1]
    out: list[int] = [bos]
    i = 0
    n = len(body)
    while i < n:
        if rng.random() < mask_prob:
            span_len = min(n - i, rng.choice((1, 1, 2, 2, 3)))
            out.append(mask_id)
            i += span_len
        else:
            out.append(body[i])
            i += 1
    out.append(eos)
    return out


@dataclass
class TrainHistory:
    best_epoch: int
    best_dev_loss: float
    best_dev_chrf: float


def pretrain_denoising(
    *,
    model: KoineSeq2SeqTransformer,
    tokenizer: BilingualTokenizer,
    mono_texts: Sequence[str],
    epochs: int = 8,
    lr: float = 6e-4,
    max_tokens: int = 3072,
    max_len: int = 256,
    seed: int = 17,
    device: torch.device,
    log_fn: Callable[[str], None] | None = None,
) -> KoineSeq2SeqTransformer:
    """mBART-style span-masking denoising pretraining on clean mono Greek+Coptic."""
    seed_everything(seed)
    rng = random.Random(seed)
    model.to(device)

    encoded_clean: list[list[int]] = []
    for txt in mono_texts:
        ids = tokenizer.encode(txt)
        if 4 <= len(ids) <= max_len:
            encoded_clean.append(ids)
    if not encoded_clean:
        return model

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.05
    )
    est_steps_per_ep = max(1, len(encoded_clean) // 48)
    total_steps = max(1, epochs * est_steps_per_ep)
    warmup = max(20, int(0.1 * total_steps))
    step = 0

    for ep in range(1, epochs + 1):
        model.train()
        noisy_pairs = [
            (
                _apply_span_noise(
                    seq, mask_id=tokenizer.mask_id, mask_prob=0.18, rng=rng
                ),
                seq,
            )
            for seq in encoded_clean
        ]
        batches = _iter_dynamic_pair_batches(
            noisy_pairs,
            max_tokens=max_tokens,
            max_batch=64,
            pad_id=model.config.pad_id,
            rng=rng,
            shuffle=True,
        )
        tot_loss = 0.0
        tot_toks = 0
        for src_b, tgt_b in batches:
            step += 1
            if step <= warmup:
                cur_lr = lr * (step / warmup)
            else:
                prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
                cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)
            tgt_in = tgt_b[:, :-1]
            tgt_out = tgt_b[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(src_b, tgt_in, return_cross_attn=False)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=model.config.pad_id,
                label_smoothing=0.1,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ntok = int(tgt_out.ne(model.config.pad_id).sum().item())
            tot_loss += float(loss.item()) * ntok
            tot_toks += ntok
            del logits, loss
            if step % 150 == 0:
                _clear_mps_cache(device)

        _clear_mps_cache(device)
        if log_fn is not None:
            log_fn(
                f"  [denoise ep {ep}/{epochs}] loss={tot_loss / max(tot_toks, 1):.4f}"
            )
    model.to("cpu")
    del optimizer
    gc.collect()
    _clear_mps_cache(device)
    return model


def train_seq2seq(
    *,
    config: Seq2SeqConfig,
    tokenizer: BilingualTokenizer,
    train_pairs_fn: Callable[[random.Random], list[tuple[str, str]]],
    dev_pairs: Sequence[tuple[str, str]],
    epochs: int,
    lr: float = 6e-4,
    min_lr_ratio: float = 0.05,
    max_tokens: int = 3072,
    max_len: int = 280,
    seed: int = 42,
    device: torch.device,
    init_model: KoineSeq2SeqTransformer | None = None,
    select_by_chrf: bool = True,
    patience: int = 8,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[KoineSeq2SeqTransformer, TrainHistory]:
    """Train or fine-tune ``KoineSeq2SeqTransformer`` selecting best on NT dev."""
    seed_everything(seed)
    rng = random.Random(seed)
    model = (
        copy.deepcopy(init_model)
        if init_model is not None
        else KoineSeq2SeqTransformer(config)
    )
    model.to(device)

    dev_encoded: list[tuple[list[int], list[int], str]] = []
    for s_txt, t_txt in dev_pairs:
        s_ids = tokenizer.encode(s_txt)[:max_len]
        t_ids = tokenizer.encode(t_txt)[:max_len]
        if len(s_ids) >= 2 and len(t_ids) >= 2:
            dev_encoded.append((s_ids, t_ids, t_txt))
    dev_batches = _iter_dynamic_pair_batches(
        [(s, t) for s, t, _ in dev_encoded],
        max_tokens=max_tokens,
        max_batch=32,
        pad_id=config.pad_id,
        rng=random.Random(0),
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.05
    )

    # Probe epoch 1 pair count for cosine schedule
    first_raw = train_pairs_fn(rng)
    est_batches = max(1, len(first_raw) // 36)
    total_steps = max(1, epochs * est_batches)
    warmup = max(40, int(0.08 * total_steps))
    step = 0

    best_state = copy.deepcopy(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )
    best_dev_loss = float("inf")
    best_dev_chrf = -1.0
    best_epoch = 0
    stale_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        raw_pairs = first_raw if ep == 1 else train_pairs_fn(rng)
        encoded_train: list[tuple[list[int], list[int]]] = []
        for s_txt, t_txt in raw_pairs:
            s_ids = tokenizer.encode(s_txt)
            t_ids = tokenizer.encode(t_txt)
            if 2 <= len(s_ids) <= max_len and 2 <= len(t_ids) <= max_len:
                encoded_train.append((s_ids, t_ids))

        batches = _iter_dynamic_pair_batches(
            encoded_train,
            max_tokens=max_tokens,
            max_batch=64,
            pad_id=config.pad_id,
            rng=rng,
            shuffle=True,
        )
        tot_loss = 0.0
        tot_toks = 0
        for src_b, tgt_b in batches:
            step += 1
            if step <= warmup:
                cur_lr = lr * (step / warmup)
            else:
                prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
                cur_lr = lr * (
                    min_lr_ratio
                    + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
                )
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)
            tgt_in = tgt_b[:, :-1]
            tgt_out = tgt_b[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(src_b, tgt_in, return_cross_attn=False)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=config.pad_id,
                label_smoothing=0.1,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ntok = int(tgt_out.ne(config.pad_id).sum().item())
            tot_loss += float(loss.item()) * ntok
            tot_toks += ntok
            del logits, loss
            if step % 150 == 0:
                _clear_mps_cache(device)

        _clear_mps_cache(device)
        # Evaluate strictly on NT dev split
        model.eval()
        dev_loss_sum = 0.0
        dev_toks = 0
        with torch.no_grad():
            for src_b, tgt_b in dev_batches:
                src_b = src_b.to(device)
                tgt_b = tgt_b.to(device)
                logits, _ = model(src_b, tgt_b[:, :-1], return_cross_attn=False)
                loss_d = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_b[:, 1:].reshape(-1),
                    ignore_index=config.pad_id,
                    reduction="sum",
                )
                dev_loss_sum += float(loss_d.item())
                dev_toks += int(tgt_b[:, 1:].ne(config.pad_id).sum().item())
        dev_loss = dev_loss_sum / max(dev_toks, 1)

        dev_chrf = 0.0
        if select_by_chrf and dev_encoded:
            refs: list[str] = []
            hyps: list[str] = []
            for start in range(0, len(dev_encoded), 32):
                chunk = dev_encoded[start : start + 32]
                ms = max(len(s) for s, _, _ in chunk)
                sb = torch.full((len(chunk), ms), config.pad_id, dtype=torch.long)
                for idx_c, (s, _, r_txt) in enumerate(chunk):
                    sb[idx_c, : len(s)] = torch.tensor(s, dtype=torch.long)
                    refs.append(r_txt)
                pred_seqs = _greedy_decode_batch(
                    model, sb.to(device), max_new_tokens=min(140, max_len)
                )
                for p_ids in pred_seqs:
                    hyps.append(normalize_greek_target(tokenizer.decode(p_ids)))
            dev_chrf = _quick_chrf(refs, hyps)

        improved = (
            (dev_chrf > best_dev_chrf + 1e-4)
            if select_by_chrf
            else (dev_loss < best_dev_loss - 1e-4)
        )
        if improved:
            best_dev_loss = dev_loss
            best_dev_chrf = dev_chrf
            best_epoch = ep
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if log_fn is not None:
            tr_loss = tot_loss / max(tot_toks, 1)
            log_fn(
                f"  [ep {ep:02d}/{epochs}] train_loss={tr_loss:.4f} "
                f"dev_loss={dev_loss:.4f} dev_chrF={dev_chrf:.4f} "
                f"(best_ep={best_epoch}, best_chrF={best_dev_chrf:.4f})"
            )
        if stale_epochs >= patience:
            break
        _clear_mps_cache(device)

    model.to("cpu")
    model.load_state_dict(best_state, strict=True)
    model.lm_head.weight = model.embed.weight
    model.eval()
    del optimizer
    gc.collect()
    _clear_mps_cache(device)
    return model, TrainHistory(
        best_epoch=best_epoch,
        best_dev_loss=round(best_dev_loss, 4),
        best_dev_chrf=round(best_dev_chrf, 4),
    )


@torch.no_grad()
def generate_backtranslations(
    *,
    reverse_model: KoineSeq2SeqTransformer,
    tokenizer: BilingualTokenizer,
    greek_texts: Sequence[str],
    batch_size: int = 64,
    max_len: int = 240,
    device: torch.device,
) -> list[tuple[str, str]]:
    """Back-translate clean Swete LXX Koine sentences into synthetic Coptic inputs."""
    reverse_model.to(device)
    reverse_model.eval()

    encoded: list[tuple[list[int], str]] = []
    for g_txt in greek_texts:
        s_ids = tokenizer.encode(f"<REV_COPTIC> {g_txt}")
        if 4 <= len(s_ids) <= max_len:
            encoded.append((s_ids, g_txt))
    encoded.sort(key=lambda item: len(item[0]))

    bt_pairs: list[tuple[str, str]] = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        ms = max(len(s) for s, _ in chunk)
        sb = torch.full((len(chunk), ms), reverse_model.config.pad_id, dtype=torch.long)
        for i, (s, _) in enumerate(chunk):
            sb[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        pred_seqs = _greedy_decode_batch(
            reverse_model, sb.to(device), max_new_tokens=min(220, ms * 3)
        )
        for p_ids, (_, g_txt) in zip(pred_seqs, chunk, strict=True):
            c_txt = tokenizer.decode(p_ids).strip()
            # Filter degenerate empty or runaway-loop synthetic outputs
            c_toks = c_txt.split()
            g_toks = g_txt.split()
            if len(c_toks) < 3 or len(c_toks) > max(180, len(g_toks) * 9):
                continue
            if len(set(c_toks)) < max(2, len(c_toks) // 5):
                continue
            # Ensure domain prefix is cleanly <DOM_BT>
            for dom in ("<DOM_NT>", "<DOM_OT>", "<DOM_BT>", "<DOM_LEX>"):
                if c_txt.startswith(dom):
                    c_txt = c_txt[len(dom) :].strip()
                    break
            bt_pairs.append((f"<DOM_BT> {c_txt}", g_txt))
        if (start // batch_size) % 25 == 0:
            _clear_mps_cache(device)
    reverse_model.to("cpu")
    _clear_mps_cache(device)
    return bt_pairs


def train_causal_koine_lm(
    *,
    config: KoineCausalLMConfig,
    tokenizer: BilingualTokenizer,
    train_greek_texts: Sequence[str],
    dev_greek_texts: Sequence[str],
    epochs: int = 12,
    lr: float = 6e-4,
    max_tokens: int = 4096,
    max_len: int = 256,
    seed: int = 101,
    device: torch.device,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[KoineCausalLM, float]:
    """Train standalone from-scratch Koine Greek causal Transformer LM."""
    seed_everything(seed)
    rng = random.Random(seed)
    lm = KoineCausalLM(config).to(device)

    train_seqs = [
        ids
        for t in train_greek_texts
        if 3 <= len(ids := tokenizer.encode(t)) <= max_len
    ]
    dev_seqs = [
        ids for t in dev_greek_texts if 3 <= len(ids := tokenizer.encode(t)) <= max_len
    ]

    optimizer = torch.optim.AdamW(
        lm.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.05
    )
    est_batches = max(1, len(train_seqs) // 48)
    total_steps = max(1, epochs * est_batches)
    warmup = max(30, int(0.08 * total_steps))
    step = 0

    best_state = copy.deepcopy(
        {k: v.detach().cpu() for k, v in lm.state_dict().items()}
    )
    best_dev_loss = float("inf")

    for ep in range(1, epochs + 1):
        lm.train()
        order = list(train_seqs)
        order.sort(key=lambda s: len(s) + rng.randint(0, 10))
        batches: list[list[list[int]]] = []
        cur: list[list[int]] = []
        cur_m = 0
        for s in order:
            nm = max(cur_m, len(s))
            if cur and ((len(cur) + 1) * nm > max_tokens or len(cur) >= 64):
                batches.append(cur)
                cur = []
                cur_m = 0
            cur.append(s)
            cur_m = max(cur_m, len(s))
        if cur:
            batches.append(cur)
        rng.shuffle(batches)

        tot_loss = 0.0
        tot_toks = 0
        for b in batches:
            step += 1
            if step <= warmup:
                cur_lr = lr * (step / warmup)
            else:
                prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
                cur_lr = lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            ml = max(len(s) for s in b)
            x = torch.full((len(b), ml), config.pad_id, dtype=torch.long)
            for i, s in enumerate(b):
                x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = lm(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
                ignore_index=config.pad_id,
                label_smoothing=0.05,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            optimizer.step()
            nt = int(x[:, 1:].ne(config.pad_id).sum().item())
            tot_loss += float(loss.item()) * nt
            tot_toks += nt
            del logits, loss
            if step % 150 == 0:
                _clear_mps_cache(device)

        _clear_mps_cache(device)
        lm.eval()
        dev_sum = 0.0
        dev_nt = 0
        with torch.no_grad():
            for start in range(0, len(dev_seqs), 32):
                b = dev_seqs[start : start + 32]
                ml = max(len(s) for s in b)
                x = torch.full((len(b), ml), config.pad_id, dtype=torch.long)
                for i, s in enumerate(b):
                    x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                x = x.to(device)
                logits = lm(x[:, :-1])
                loss_d = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1),
                    ignore_index=config.pad_id,
                    reduction="sum",
                )
                dev_sum += float(loss_d.item())
                dev_nt += int(x[:, 1:].ne(config.pad_id).sum().item())
        dev_loss = dev_sum / max(dev_nt, 1)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in lm.state_dict().items()}
            )
        if log_fn is not None:
            tr_l = tot_loss / max(tot_toks, 1)
            log_fn(
                f"  [lm ep {ep:02d}/{epochs}] train_loss={tr_l:.4f} "
                f"dev_loss={dev_loss:.4f} (ppl={math.exp(min(dev_loss, 20.0)):.2f})"
            )

    lm.to("cpu")
    lm.load_state_dict(best_state, strict=True)
    lm.lm_head.weight = lm.embed.weight
    lm.eval()
    del optimizer
    gc.collect()
    _clear_mps_cache(device)
    return lm, best_dev_loss


def corrupt_v1_draft(
    draft: str,
    *,
    vocab_pool: Sequence[str],
    rng: random.Random,
    drop_p: float = 0.15,
    swap_p: float = 0.10,
    sub_p: float = 0.06,
) -> str:
    """Apply stochastic noise to model-v1 draft to prevent copy-overfitting."""
    words = draft.split()
    if len(words) <= 2:
        return draft
    out: list[str] = []
    n_pool = len(vocab_pool)
    for w in words:
        if rng.random() < drop_p:
            continue
        if n_pool > 0 and rng.random() < sub_p:
            out.append(vocab_pool[rng.randrange(n_pool)])
        else:
            out.append(w)
    if not out:
        return draft
    # Occasional adjacent word swaps so the neural decoder learns true Koine order
    i = 0
    while i + 1 < len(out):
        if rng.random() < swap_p:
            out[i], out[i + 1] = out[i + 1], out[i]
            i += 2
        else:
            i += 1
    return " ".join(out)


def make_forward_pairs_epoch(
    *,
    nt_train: Sequence[ParallelExample],
    ot_train: Sequence[ParallelExample],
    clause_aug: Sequence[ParallelExample],
    lex_aug: Sequence[tuple[str, str]],
    bt_pairs: Sequence[tuple[str, str]],
    ccl_lexicon: CclLexicon,
    v1_drafts: Mapping[str, str] | None = None,
    draft_vocab_pool: Sequence[str] = (),
    draft_drop_p: float = 0.25,
    nt_repeat: int = 2,
    loan_drop_p: float = 0.05,
    morpheme_drop_p: float = 0.03,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Build one randomized-augmentation epoch of Coptic -> Koine training pairs."""

    def _pick_draft(unit_id: str) -> str | None:
        if not v1_drafts:
            return None
        raw = v1_drafts.get(unit_id)
        if not raw or rng.random() < draft_drop_p:
            return None
        return corrupt_v1_draft(raw, vocab_pool=draft_vocab_pool, rng=rng)

    pairs: list[tuple[str, str]] = []
    for _ in range(nt_repeat):
        for ex in nt_train:
            s_txt, _ = serialize_coptic_morphemes(
                ex.coptic_morphemes,
                ccl_lexicon=ccl_lexicon,
                domain_tag=ex.domain_tag,
                v1_draft=_pick_draft(ex.unit_id),
                loan_drop_p=loan_drop_p,
                morpheme_drop_p=morpheme_drop_p,
                rng=rng,
            )
            pairs.append((s_txt, ex.greek_target))
    for ex in ot_train:
        s_txt, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes,
            ccl_lexicon=ccl_lexicon,
            domain_tag=ex.domain_tag,
            v1_draft=_pick_draft(ex.unit_id),
            loan_drop_p=loan_drop_p,
            morpheme_drop_p=morpheme_drop_p,
            rng=rng,
        )
        pairs.append((s_txt, ex.greek_target))
    for ex in clause_aug:
        s_txt, _ = serialize_coptic_morphemes(
            ex.coptic_morphemes,
            ccl_lexicon=ccl_lexicon,
            domain_tag=ex.domain_tag,
            v1_draft=_pick_draft(ex.unit_id),
            loan_drop_p=loan_drop_p,
            morpheme_drop_p=0.0,
            rng=rng,
        )
        pairs.append((s_txt, ex.greek_target))
    pairs.extend(bt_pairs)
    # Sample ~35% of CCL lexical pairs per epoch to reinforce rare Greek lemmas
    if lex_aug:
        k_lex = min(len(lex_aug), 2500)
        pairs.extend(rng.sample(list(lex_aug), k_lex))
    return pairs
