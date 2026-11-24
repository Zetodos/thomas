# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline polytonic accent & capitalization restorer for Koine Greek.

Trained strictly without NT holdout, NT dev, Dilley Thomas, or P.Oxy. witnesses
(using MorphGNT TRAIN + Swete LXX + pure CCL + optional Apostolic Fathers /
First1KGreek). Restores unaccented lowercase Koine decoder outputs to acute-only
polytonic NFC Greek (matching MorphGNT normalized surfaces) before lemmatization.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from pathlib import Path

from zetodos_thomas.ingest.holdouts import assert_clean_training_unit
from zetodos_thomas.ingest.koine_patristics import load_all_patristic_koine_sentences
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt
from zetodos_thomas.ingest.ot_swete import parse_swete_file
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon

_GREEK_VOWELS = frozenset({"α", "ε", "η", "ι", "ο", "υ", "ω"})
_GREEK_DIPHTHONGS = frozenset({"αι", "αυ", "ει", "ευ", "ηυ", "οι", "ου", "υι"})
_ROUGH_INITIAL_PREFIXES = (
    "υπ",
    "υμ",
    "υι",
    "υδ",
    "υγ",
    "υλ",
    "υψ",
    "υστ",
    "ημ",
    "ηγεμ",
    "ηλικ",
    "ημερ",
    "ησυχ",
    "ιερ",
    "ικαν",
    "ινα",
    "ιππ",
    "ιστ",
    "οδ",
    "ολ",
    "ομ",
    "οπ",
    "ορ",
    "οσ",
    "οτ",
    "ουτ",
    "εαυτ",
    "εβδ",
    "εικ",
    "εκατ",
    "ελλην",
    "ελκ",
    "ελπ",
    "εν",
    "εξ",
    "επτ",
    "ερμ",
    "εστ",
    "ετερ",
    "ετοιμ",
    "ευρ",
    "εωσ",
    "αγι",
    "αδ",
    "αιμ",
    "αιρ",
    "αμ",
    "απασ",
    "απαντ",
    "απλ",
    "απτ",
    "αρμ",
    "αρπ",
    "ωρ",
    "ωσ",
)
_SMOOTH_COMBINING = "\u0313"
_ROUGH_COMBINING = "\u0314"
_ACUTE_COMBINING = "\u0301"
_CIRCUMFLEX_COMBINING = "\u0342"


def _accent_class(polytonic_word: str) -> str:
    """Classify a restored polytonic Greek word's accent position (for enclitics)."""
    if not polytonic_word or polytonic_word == "<BOS>":
        return "BOS"
    bare = _bare_key(polytonic_word)
    nuclei = _split_syllable_nuclei(bare)
    if not nuclei:
        return "NONE"
    # Walk NFD characters tracking which syllable nucleus carries acute/circumflex
    nfd = unicodedata.normalize("NFD", polytonic_word.lower())
    base_idx = -1
    accent_base = -1
    accent_kind = ""
    for ch in nfd:
        if unicodedata.category(ch) == "Mn":
            if ch in {_ACUTE_COMBINING, "\u0300"}:
                accent_base = base_idx
                accent_kind = "A"
            elif ch == _CIRCUMFLEX_COMBINING:
                accent_base = base_idx
                accent_kind = "C"
        else:
            base_idx += 1
    if accent_base < 0:
        return "UNACC"
    # Map base_idx to syllable index from the right (0=ultima, 1=penult, 2=antepenult)
    n_syl = len(nuclei)
    for s_idx, (s_beg, s_end) in enumerate(nuclei):
        if s_beg <= accent_base < s_end:
            from_end = (n_syl - 1) - s_idx
            return f"{accent_kind}{min(from_end, 2)}"
    return "NONE"


def strip_greek_to_bare(text: str) -> str:
    """Strip polytonic diacritics, lowercase, and normalize word-final sigma."""
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").lower()
    words: list[str] = []
    for raw in stripped.split():
        if len(raw) > 1 and raw.endswith("σ"):
            raw = raw[:-1] + "ς"
        if raw:
            words.append(raw)
    return " ".join(words)


def _normalize_polytonic_acute(word: str) -> str:
    """Normalize a Greek surface token to acute-only NFC (matching MorphGNT)."""
    cleaned = word.replace("(", "").replace(")", "").strip()
    cleaned = cleaned.strip('[]<>«»"“”„†‡*.,;;·:!?—-')
    nfd = unicodedata.normalize("NFD", cleaned).replace("\u0300", "\u0301")
    nfc = unicodedata.normalize("NFC", nfd)
    if len(nfc) > 1 and nfc.endswith("σ"):
        nfc = nfc[:-1] + "ς"
    return nfc


def _bare_key(word: str) -> str:
    nfd = unicodedata.normalize("NFD", word)
    bare = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").lower()
    if len(bare) > 1 and bare.endswith("σ"):
        bare = bare[:-1] + "ς"
    return bare


def _split_syllable_nuclei(bare: str) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` indices of vowel/diphthong nuclei."""
    nuclei: list[tuple[int, int]] = []
    i = 0
    n = len(bare)
    while i < n:
        if bare[i] in _GREEK_VOWELS:
            if i + 1 < n and bare[i : i + 2] in _GREEK_DIPHTHONGS:
                nuclei.append((i, i + 2))
                i += 2
            else:
                nuclei.append((i, i + 1))
                i += 1
        else:
            i += 1
    return nuclei


def _recessive_accent_fallback(bare: str) -> str:
    """Deterministic morphological/recessive polytonic fallback for unseen forms."""
    if not bare:
        return ""
    clean = bare.lower()
    if len(clean) > 1 and clean.endswith("σ"):
        clean = clean[:-1] + "ς"
    nuclei = _split_syllable_nuclei(clean)
    if not nuclei:
        return clean

    chars = list(clean)
    # 1. Breathing on initial vowel / diphthong / rho
    if clean[0] == "ρ":
        chars[0] = chars[0] + _ROUGH_COMBINING
    elif nuclei[0][0] == 0:
        b_idx = nuclei[0][1] - 1
        cmp = clean.replace("ς", "σ")
        breathing = (
            _ROUGH_COMBINING
            if any(cmp.startswith(pref) for pref in _ROUGH_INITIAL_PREFIXES)
            else _SMOOTH_COMBINING
        )
        chars[b_idx] = chars[b_idx] + breathing

    # 2. Universal Koine verbal suffix accents, else recessive acute
    acc_mark = _ACUTE_COMBINING
    if len(nuclei) >= 2 and clean.endswith(("θηναι", "φηναι", "χηναι")):
        acc_nuc = nuclei[-2]
        acc_mark = _CIRCUMFLEX_COMBINING
    elif len(nuclei) >= 3 and clean.endswith(
        (
            "μενοσ",
            "μενος",
            "μενη",
            "μενον",
            "μενοι",
            "μεναι",
            "μενα",
            "μενου",
            "μενης",
            "μενω",
            "μενων",
            "μενους",
            "μενας",
        )
    ):
        # Perfect passive participles carry penult acute (-μένος / -μένα)
        acc_nuc = nuclei[-2]
    elif len(nuclei) <= 2:
        acc_nuc = nuclei[0]
    else:
        ultima = clean[nuclei[-1][0] : nuclei[-1][1]]
        long_ultima = ultima in {"η", "ω", "αυ", "ει", "ευ", "ηυ", "ου", "υι"} or (
            ultima in {"αι", "οι"} and not clean.endswith(("αι", "οι"))
        )
        acc_nuc = nuclei[-2] if long_ultima else nuclei[-3]
    a_idx = acc_nuc[1] - 1
    chars[a_idx] = chars[a_idx] + acc_mark
    return unicodedata.normalize("NFC", "".join(chars))


class KoineAccentRestorer:
    """Deterministic offline unigram + bigram + enclitic + recessive restorer."""

    def __init__(
        self,
        *,
        unigrams: dict[str, str],
        prev_bigrams: dict[str, str],
        next_bigrams: dict[str, str] | None = None,
        enclitic_rules: dict[str, str] | None = None,
    ) -> None:
        self._uni = unigrams
        self._prev_bi = prev_bigrams
        self._next_bi = next_bigrams or {}
        self._enclitic = enclitic_rules or {}

    @property
    def unigram_size(self) -> int:
        return len(self._uni)

    @property
    def bigram_size(self) -> int:
        return len(self._prev_bi) + len(self._next_bi) + len(self._enclitic)

    def restore_word(
        self,
        bare_word: str,
        *,
        prev_bare: str = "<BOS>",
        next_bare: str = "<EOS>",
        prev_accent: str = "BOS",
    ) -> str:
        """Restore polytonic accents & capitalization for a single bare word."""
        key = _bare_key(bare_word)
        if not key:
            return bare_word
        prev_k = f"{prev_bare} {key}"
        if prev_k in self._prev_bi:
            return self._prev_bi[prev_k]
        next_k = f"{key} {next_bare}"
        if next_k in self._next_bi:
            return self._next_bi[next_k]
        enc_k = f"{prev_accent} {key}"
        if enc_k in self._enclitic:
            return self._enclitic[enc_k]
        if key in self._uni:
            return self._uni[key]
        return _recessive_accent_fallback(key)

    def restore_sentence(self, text: str) -> str:
        """Deterministically restore polytonic accents for a space-separated string."""
        raw_words = text.split()
        if not raw_words:
            return ""
        bare_words = [_bare_key(w) for w in raw_words]
        out: list[str] = []
        n = len(raw_words)
        prev_acc = "BOS"
        for idx, raw in enumerate(raw_words):
            prev_b = bare_words[idx - 1] if idx > 0 and bare_words[idx - 1] else "<BOS>"
            next_b = (
                bare_words[idx + 1] if idx + 1 < n and bare_words[idx + 1] else "<EOS>"
            )
            restored = self.restore_word(
                raw, prev_bare=prev_b, next_bare=next_b, prev_accent=prev_acc
            )
            out.append(restored)
            prev_acc = _accent_class(restored)
        return " ".join(out)

    @classmethod
    def build(
        cls,
        *,
        morphgnt_dir: Path,
        exclude_unit_ids: Collection[str],
        swete_dir: Path | None = None,
        ccl_lexicon: CclLexicon | None = None,
        downloads_dir: Path | None = None,
        include_patristics: bool = True,
        min_bigram_count: int = 3,
        min_bigram_purity: float = 0.75,
    ) -> KoineAccentRestorer:
        """Train leak-free restorer on MorphGNT TRAIN + Swete + CCL + patristics."""
        excluded = set(exclude_unit_ids)
        nt_uni_counts: dict[str, Counter[str]] = defaultdict(Counter)
        aux_uni_counts: dict[str, Counter[str]] = defaultdict(Counter)
        prev_bi_counts: dict[str, Counter[str]] = defaultdict(Counter)
        next_bi_counts: dict[str, Counter[str]] = defaultdict(Counter)
        enclitic_counts: dict[str, Counter[str]] = defaultdict(Counter)

        def _ingest_nt_tokens(words: list[str]) -> None:
            surfs = [_normalize_polytonic_acute(w) for w in words]
            bares = [_bare_key(s) for s in surfs]
            n = len(surfs)
            prev_acc = "BOS"
            for i, (surf, key) in enumerate(zip(surfs, bares, strict=True)):
                if not key:
                    continue
                nt_uni_counts[key][surf] += 1
                prev_b = bares[i - 1] if i > 0 and bares[i - 1] else "<BOS>"
                next_b = bares[i + 1] if i + 1 < n and bares[i + 1] else "<EOS>"
                prev_bi_counts[f"{prev_b} {key}"][surf] += 1
                next_bi_counts[f"{key} {next_b}"][surf] += 1
                enclitic_counts[f"{prev_acc} {key}"][surf] += 1
                prev_acc = _accent_class(surf)

        def _ingest_aux_tokens(words: Iterable[str]) -> None:
            for raw in words:
                surf = _normalize_polytonic_acute(raw)
                key = _bare_key(surf)
                if key:
                    aux_uni_counts[key][surf] += 1

        # 1. MorphGNT TRAIN (canonical Gospel accentuation & proper-noun priority)
        mg_units = load_morphgnt_nt(morphgnt_dir)
        for uid, unit in sorted(mg_units.items()):
            if uid in excluded:
                continue
            assert_clean_training_unit(
                uid, source_id=unit.source_id, excluded_ids=excluded
            )
            toks = [
                (t.normalized or t.surface)
                for t in unit.tokens
                if t.morph.pos != "X-" and (t.normalized or t.surface)
            ]
            _ingest_nt_tokens(toks)

        # 2. Swete LXX (fallback for non-NT vocabulary)
        if swete_dir is not None and swete_dir.is_dir():
            sw_data = swete_dir / "data" if (swete_dir / "data").is_dir() else swete_dir
            for txt_path in sorted(sw_data.glob("*.txt")):
                parsed = parse_swete_file(txt_path.read_text(encoding="utf-8"))
                for _k, words in sorted(parsed.items()):
                    _ingest_aux_tokens(words)

        # 3. Pure CCL Greek loanword lemmas
        if ccl_lexicon is not None:
            for entries in ccl_lexicon._loanwords.values():
                for e in entries:
                    _ingest_aux_tokens((e.greek_lemma,))

        # 4. Pinned Patristic & 1st-century Koine prose
        if include_patristics and downloads_dir is not None:
            for _uid, sent in load_all_patristic_koine_sentences(downloads_dir):
                _ingest_aux_tokens(sent.split())

        # Unigram resolution: MorphGNT TRAIN wins when attested; else auxiliary
        unigrams: dict[str, str] = {}
        all_keys = set(nt_uni_counts) | set(aux_uni_counts)
        for key in all_keys:
            cnt = nt_uni_counts[key] if key in nt_uni_counts else aux_uni_counts[key]
            winner = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            unigrams[key] = winner

        # High-precision MorphGNT-TRAIN context overrides for ambiguous bare keys
        prev_bigrams: dict[str, str] = {}
        for bi_key, cnt in prev_bi_counts.items():
            total = sum(cnt.values())
            if total < min_bigram_count:
                continue
            bare = bi_key.split(" ", 1)[1]
            if len(nt_uni_counts[bare]) <= 1:
                continue
            winner, w_cnt = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if winner != unigrams.get(bare) and (w_cnt / total) >= min_bigram_purity:
                prev_bigrams[bi_key] = winner

        next_bigrams: dict[str, str] = {}
        for bi_key, cnt in next_bi_counts.items():
            total = sum(cnt.values())
            if total < min_bigram_count:
                continue
            bare = bi_key.split(" ", 1)[0]
            if len(nt_uni_counts[bare]) <= 1:
                continue
            winner, w_cnt = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if winner != unigrams.get(bare) and (w_cnt / total) >= min_bigram_purity:
                next_bigrams[bi_key] = winner

        # Empirical enclitic accent rules keyed by (prev_accent_class, bare)
        enclitic_rules: dict[str, str] = {}
        for enc_k, cnt in enclitic_counts.items():
            total = sum(cnt.values())
            if total < 5:
                continue
            bare = enc_k.split(" ", 1)[1]
            if len(nt_uni_counts[bare]) <= 1:
                continue
            winner, w_cnt = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if winner != unigrams.get(bare) and (w_cnt / total) >= 0.70:
                enclitic_rules[enc_k] = winner

        return cls(
            unigrams=unigrams,
            prev_bigrams=prev_bigrams,
            next_bigrams=next_bigrams,
            enclitic_rules=enclitic_rules,
        )

    def save_shards(self, out_dir: Path, *, max_bytes: int = 600_000) -> list[Path]:
        """Persist accent tables in <1024 KiB JSON shards."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("accent-*.json"):
            old.unlink()
        files: list[Path] = []

        def _write_dict(prefix: str, mapping: dict[str, str]) -> None:
            shard: dict[str, str] = {}
            shard_idx = 0
            est = 2
            for k, v in sorted(mapping.items()):
                row_bytes = len(k.encode("utf-8")) + len(v.encode("utf-8")) + 6
                if est + row_bytes > max_bytes and shard:
                    p = out_dir / f"{prefix}-{shard_idx}.json"
                    p.write_text(
                        json.dumps(shard, ensure_ascii=False, separators=(",", ":"))
                        + "\n",
                        encoding="utf-8",
                    )
                    files.append(p)
                    shard = {}
                    shard_idx += 1
                    est = 2
                shard[k] = v
                est += row_bytes
            if shard or shard_idx == 0:
                p = out_dir / f"{prefix}-{shard_idx}.json"
                p.write_text(
                    json.dumps(shard, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                files.append(p)

        _write_dict("accent-uni", self._uni)
        _write_dict("accent-bi-prev", self._prev_bi)
        _write_dict("accent-bi-next", self._next_bi)
        _write_dict("accent-enclitic", self._enclitic)
        return files

    @classmethod
    def load_shards(cls, model_dir: Path) -> KoineAccentRestorer:
        """Load sharded accent restorer tables from ``model_dir``."""
        uni: dict[str, str] = {}
        prev_bi: dict[str, str] = {}
        next_bi: dict[str, str] = {}
        enclitic: dict[str, str] = {}
        for p in sorted(model_dir.glob("accent-uni-*.json")):
            uni.update(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(model_dir.glob("accent-bi-prev-*.json")):
            prev_bi.update(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(model_dir.glob("accent-bi-next-*.json")):
            next_bi.update(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(model_dir.glob("accent-enclitic-*.json")):
            enclitic.update(json.loads(p.read_text(encoding="utf-8")))
        return cls(
            unigrams=uni,
            prev_bigrams=prev_bi,
            next_bigrams=next_bi,
            enclitic_rules=enclitic,
        )
