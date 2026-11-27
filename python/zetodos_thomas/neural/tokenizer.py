# SPDX-License-Identifier: Apache-2.0
"""Deterministic shared Coptic + Koine Greek SentencePiece BPE tokenizer."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import sentencepiece as spm

from zetodos_thomas.neural.corpus import SPECIAL_USER_SYMBOLS

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class BilingualTokenizer:
    """Thin typed wrapper over a shared Coptic+Greek SentencePiece BPE model."""

    def __init__(self, sp: spm.SentencePieceProcessor, model_path: Path) -> None:
        self._sp = sp
        self.model_path = model_path
        self.pad_id: int = int(sp.pad_id())
        self.bos_id: int = int(sp.bos_id())
        self.eos_id: int = int(sp.eos_id())
        self.unk_id: int = int(sp.unk_id())
        self.vocab_size: int = int(sp.get_piece_size())
        self.mask_id: int = int(sp.piece_to_id("<MASK>"))
        self._word_start_flags: list[bool] | None = None

    def word_start_flags(self) -> list[bool]:
        """Return bool flag per vocab ID indicating a word-initial ``▁`` piece."""
        if self._word_start_flags is None:
            flags = [False] * self.vocab_size
            for tid in range(self.vocab_size):
                if tid in {self.pad_id, self.bos_id, self.eos_id}:
                    continue
                piece = str(self._sp.id_to_piece(tid))
                if piece.startswith("\u2581"):
                    flags[tid] = True
            self._word_start_flags = flags
        return self._word_start_flags

    def id_to_piece(self, tid: int) -> str:
        """Return the raw SentencePiece string for token ID ``tid``."""
        return str(self._sp.id_to_piece(int(tid)))

    @classmethod
    def train(
        cls,
        *,
        texts: Iterable[str],
        out_prefix: Path,
        vocab_size: int = 8000,
        user_defined_symbols: Sequence[str] | None = None,
    ) -> BilingualTokenizer:
        """Train a deterministic SentencePiece BPE model strictly on train texts."""
        symbols = (
            list(user_defined_symbols)
            if user_defined_symbols is not None
            else list(SPECIAL_USER_SYMBOLS)
        )
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=True
        ) as tmp:
            for line in texts:
                clean = line.strip()
                if clean:
                    tmp.write(clean + "\n")
            tmp.flush()
            spm.SentencePieceTrainer.train(
                input=tmp.name,
                model_prefix=str(out_prefix),
                model_type="bpe",
                vocab_size=vocab_size,
                character_coverage=1.0,
                normalization_rule_name="identity",
                pad_id=PAD_ID,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                unk_id=UNK_ID,
                user_defined_symbols=symbols,
                byte_fallback=True,
                shuffle_input_sentence=False,
                num_threads=1,
                minloglevel=2,
            )
        model_file = out_prefix.with_suffix(".model")
        return cls.load(model_file)

    def has_piece(self, piece: str) -> bool:
        """Return True if ``piece`` is an explicit non-UNK vocabulary symbol."""
        return int(self._sp.piece_to_id(piece)) != self.unk_id

    @classmethod
    def load(cls, model_path: Path) -> BilingualTokenizer:
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        return cls(sp=sp, model_path=model_path)

    def encode(
        self, text: str, *, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        ids: list[int] = list(self._sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        filtered = [
            int(i) for i in ids if int(i) not in {self.pad_id, self.bos_id, self.eos_id}
        ]
        return str(self._sp.decode(filtered)).strip()

    def encode_with_coptic_seq_map(
        self,
        text: str,
        morpheme_char_spans: Sequence[tuple[int, int, int]],
    ) -> tuple[list[int], list[int | None]]:
        """Encode Coptic text and map every encoder token index to its Coptic seq."""
        mapping: dict[str, Any] = self._sp.encode_as_offset_mapping(
            text, return_bytes=False
        )
        raw_ids: list[int] = list(mapping["ids"])
        offsets: Sequence[tuple[int, int]] = mapping["offsets"]

        ids: list[int] = [self.bos_id]
        seq_map: list[int | None] = [None]
        span_idx = 0
        n_spans = len(morpheme_char_spans)

        for tid, (c_beg, c_end) in zip(raw_ids, offsets, strict=True):
            ids.append(int(tid))
            mid = (int(c_beg) + max(int(c_end) - 1, int(c_beg))) // 2
            while span_idx < n_spans and morpheme_char_spans[span_idx][2] <= c_beg:
                span_idx += 1
            matched_seq: int | None = None
            for k in range(span_idx, min(span_idx + 3, n_spans)):
                m_seq, s_beg, s_end = morpheme_char_spans[k]
                if s_beg <= mid < s_end or (c_beg < s_end and c_end > s_beg):
                    matched_seq = m_seq
                    break
            seq_map.append(matched_seq)

        ids.append(self.eos_id)
        seq_map.append(None)
        return ids, seq_map

    def piece_word_groups(self, ids: Sequence[int]) -> list[tuple[str, list[int]]]:
        """Group target subword token indices into decoded surface words."""
        # Collect visible token positions alongside their raw SentencePiece strings
        words: list[tuple[str, list[int]]] = []
        cur_ids: list[int] = []
        cur_pos: list[int] = []
        for pos, tid in enumerate(ids):
            itid = int(tid)
            if itid in {self.pad_id, self.bos_id, self.eos_id}:
                continue
            piece: str = str(self._sp.id_to_piece(itid))
            starts_new = piece.startswith("\u2581")
            if starts_new and cur_ids:
                w = str(self._sp.decode(cur_ids)).strip()
                if w:
                    words.append((w, list(cur_pos)))
                cur_ids = []
                cur_pos = []
            cur_ids.append(itid)
            cur_pos.append(pos)
        if cur_ids:
            w = str(self._sp.decode(cur_ids)).strip()
            if w:
                words.append((w, list(cur_pos)))
        return words
