# SPDX-License-Identifier: Apache-2.0
"""Smoothed 3-gram Koine Greek lemma+morph and surface language model."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from zetodos_thomas.ingest.holdouts import (
    assert_clean_training_unit,
    clean_nt_train_ids,
    load_nt_split,
)
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt
from zetodos_thomas.ingest.ot_swete import load_ot_swete_pairs
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.schemas.linguistic import GreekMorphFeatures

_CLAUSE_STOP_LEMMAS = frozenset({"καί", "ἀλλά", "δέ", "ὅτι", "ἵνα", "ἐάν", "ὅταν"})


def morph_lm_key(lemma: str, morph: GreekMorphFeatures) -> str:
    """Canonical compact LM token key ``lemma|pos|case_or_mood|num``."""
    slot3 = morph.case or morph.mood or "-"
    slot4 = morph.number or morph.tense or "-"
    return f"{lemma}|{morph.pos}|{slot3}|{slot4}"


class KoineNgramLM:
    """Interpolated 3-gram Koine LM over MorphGNT train + Swete LXX train."""

    def __init__(
        self,
        *,
        unigram_logp: dict[str, float],
        bigram_logp: dict[tuple[str, str], float],
        trigram_logp: dict[tuple[str, str, str], float],
        surface_bi_logp: dict[tuple[str, str], float],
        verb_arg_pmi: dict[tuple[str, str], float] | None = None,
        default_unigram: float = -10.5,
    ) -> None:
        self._uni = unigram_logp
        self._bi = bigram_logp
        self._tri = trigram_logp
        self._surf_bi = surface_bi_logp
        self._colloc = verb_arg_pmi or {}
        self._default_uni = default_unigram

    def score_collocation(self, verb_lemma: str, arg_lemma: str) -> float:
        """Return smoothed pointwise mutual information for ``(verb, arg)``."""
        return self._colloc.get((verb_lemma, arg_lemma), 0.0)

    @classmethod
    def train(
        cls,
        *,
        downloads_dir: Path,
        split_path: Path,
        inflector: GreekInflector,
        nt_weight: float = 3.0,
    ) -> KoineNgramLM:
        """Train LM strictly on clean NT TRAIN split + Swete LXX."""
        split = load_nt_split(split_path)
        train_ids = clean_nt_train_ids(split, exclude_synoptic_thomas=True)
        morphgnt = load_morphgnt_nt(downloads_dir / "morphgnt-sblgnt")

        uni_c: dict[str, float] = defaultdict(float)
        bi_c: dict[tuple[str, str], float] = defaultdict(float)
        tri_c: dict[tuple[str, str, str], float] = defaultdict(float)
        bi_prev: dict[str, float] = defaultdict(float)
        tri_prev: dict[tuple[str, str], float] = defaultdict(float)
        surf_bi_c: dict[tuple[str, str], float] = defaultdict(float)
        surf_prev: dict[str, float] = defaultdict(float)
        v_arg_c: dict[tuple[str, str], float] = defaultdict(float)
        v_cnt: dict[str, float] = defaultdict(float)
        a_cnt: dict[str, float] = defaultdict(float)
        tot_va = 0.0

        def _feed_colloc(seq_items: list[tuple[str, str, str]], weight: float) -> None:
            nonlocal tot_va
            seq_len = len(seq_items)
            for idx_v, (v_lem, v_pos, _) in enumerate(seq_items):
                if v_pos != "V-" or not v_lem:
                    continue
                for off in range(1, 6):
                    if idx_v + off >= seq_len:
                        break
                    a_lem, a_pos, a_surf = seq_items[idx_v + off]
                    if (
                        a_pos == "X-"
                        or a_surf in {".", ",", "·", ";", ";"}
                        or (a_pos == "C-" and a_lem in _CLAUSE_STOP_LEMMAS)
                    ):
                        break
                    if a_pos in {"N-", "A-"} and a_lem:
                        v_arg_c[(v_lem, a_lem)] += weight
                        v_cnt[v_lem] += weight
                        a_cnt[a_lem] += weight
                        tot_va += weight

        def _feed_sequence(keys: list[str], surfs: list[str], weight: float) -> None:
            padded = ["<s>", "<s>", *keys, "</s>"]
            spadded = ["<s>", *surfs, "</s>"]
            for i in range(2, len(padded)):
                k0, k1, k2 = padded[i - 2], padded[i - 1], padded[i]
                uni_c[k2] += weight
                bi_c[(k1, k2)] += weight
                bi_prev[k1] += weight
                tri_c[(k0, k1, k2)] += weight
                tri_prev[(k0, k1)] += weight
            for i in range(1, len(spadded)):
                s1, s2 = spadded[i - 1], spadded[i]
                surf_bi_c[(s1, s2)] += weight
                surf_prev[s1] += weight

        for uid in sorted(train_ids):
            u = morphgnt.get(uid)
            if u is None:
                continue
            assert_clean_training_unit(u.unit_id, source_id=u.source_id)
            keys = [morph_lm_key(t.lemma, t.morph) for t in u.tokens]
            surfs = [t.normalized for t in u.tokens]
            _feed_sequence(keys, surfs, nt_weight)
            _feed_colloc(
                [(t.lemma, t.morph.pos, t.normalized) for t in u.tokens],
                nt_weight,
            )

        ot_pairs = load_ot_swete_pairs(
            tt_zip=downloads_dir / "sahidic-ot" / "sahidic.ot_TT.zip",
            swete_dir=downloads_dir / "lxx-swete",
        )
        for _c, g_u in ot_pairs:
            assert_clean_training_unit(g_u.unit_id, source_id=g_u.source_id)
            keys: list[str] = []
            surfs: list[str] = []
            colloc_seq: list[tuple[str, str, str]] = []
            for tok in g_u.tokens:
                parses = inflector.analyze_surface(tok.surface)
                lem, mf, _ = parses[0] if parses else (tok.normalized, tok.morph, -5.0)
                keys.append(morph_lm_key(lem, mf))
                surfs.append(tok.normalized)
                colloc_seq.append((lem, mf.pos, tok.normalized))
            _feed_sequence(keys, surfs, 1.0)
            _feed_colloc(colloc_seq, 1.0)

        tot_u = max(sum(uni_c.values()), 1.0)
        vocab = max(len(uni_c), 1)
        unigram_logp = {
            k: round(math.log((c + 0.5) / (tot_u + 0.5 * vocab)), 4)
            for k, c in uni_c.items()
            if c >= 2.0
        }
        # Prune rare n-grams (count < 3 -> saves >85% storage, fits <1024 KiB cleanly)
        bigram_logp = {
            (k1, k2): round(math.log((c + 0.1) / (bi_prev[k1] + 50.0)), 4)
            for (k1, k2), c in bi_c.items()
            if c >= 3.0
        }
        trigram_logp = {
            (k0, k1, k2): round(math.log((c + 0.05) / (tri_prev[(k0, k1)] + 15.0)), 4)
            for (k0, k1, k2), c in tri_c.items()
            if c >= 4.0
        }
        surface_bi_logp = {
            (s1, s2): round(math.log((c + 0.1) / (surf_prev[s1] + 40.0)), 4)
            for (s1, s2), c in surf_bi_c.items()
            if c >= 4.0
        }
        verb_arg_pmi: dict[tuple[str, str], float] = {}
        va_denom_scale = max(tot_va, 1.0)
        for (v_lem, a_lem), c in v_arg_c.items():
            if c < 4.0:
                continue
            pmi = math.log(
                ((c + 0.5) * va_denom_scale)
                / ((v_cnt[v_lem] + 5.0) * (a_cnt[a_lem] + 5.0))
            )
            if pmi >= 0.35:
                verb_arg_pmi[(v_lem, a_lem)] = round(min(pmi, 3.0), 4)

        return cls(
            unigram_logp=unigram_logp,
            bigram_logp=bigram_logp,
            trigram_logp=trigram_logp,
            surface_bi_logp=surface_bi_logp,
            verb_arg_pmi=verb_arg_pmi,
        )

    def score_step(
        self,
        *,
        k0: str,
        k1: str,
        k2: str,
        s1: str,
        s2: str,
    ) -> float:
        """Interpolated trigram + bigram + unigram + surface-bigram log score."""
        p_uni = self._uni.get(k2, self._default_uni)
        p_bi = self._bi.get((k1, k2), p_uni - 1.2)
        p_tri = self._tri.get((k0, k1, k2), p_bi - 0.8)
        s_bi = self._surf_bi.get((s1, s2), -8.5)
        return 0.45 * p_tri + 0.35 * p_bi + 0.20 * p_uni + 0.25 * s_bi

    def save_shards(self, out_dir: Path, *, max_bytes: int = 680_000) -> list[Path]:
        """Persist compact LM shards, every file strictly < 1024 KiB."""
        out_dir.mkdir(parents=True, exist_ok=True)
        legacy_uni = out_dir / "koine-lm-uni.json"
        if legacy_uni.is_file():
            legacy_uni.unlink()
        for old in out_dir.glob("koine-lm-*.json"):
            old.unlink()
        files: list[Path] = []

        uni_chunk: dict[str, float] = {}
        uni_idx = 0
        uni_est = 2
        for k, v in sorted(self._uni.items()):
            sv = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            esz = len(k.encode("utf-8")) + len(sv.encode("utf-8")) + 6
            if uni_est + esz > max_bytes and uni_chunk:
                p = out_dir / f"koine-lm-uni-{uni_idx}.json"
                p.write_text(
                    json.dumps(uni_chunk, ensure_ascii=False, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                files.append(p)
                uni_chunk = {}
                uni_idx += 1
                uni_est = 2
            uni_chunk[k] = v
            uni_est += esz
        if uni_chunk or uni_idx == 0:
            p = out_dir / f"koine-lm-uni-{uni_idx}.json"
            p.write_text(
                json.dumps(uni_chunk, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            files.append(p)

        def _write_table_shards(prefix: str, items: list[list[object]]) -> None:
            shard: list[list[object]] = []
            idx = 0
            est = 2
            for row in items:
                srow = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                rsz = len(srow.encode("utf-8")) + 2
                if est + rsz > max_bytes and shard:
                    p = out_dir / f"{prefix}-{idx}.json"
                    p.write_text(
                        json.dumps(shard, ensure_ascii=False, separators=(",", ":"))
                        + "\n",
                        encoding="utf-8",
                    )
                    files.append(p)
                    shard = []
                    idx += 1
                    est = 2
                shard.append(row)
                est += rsz
            if shard or idx == 0:
                p = out_dir / f"{prefix}-{idx}.json"
                p.write_text(
                    json.dumps(shard, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                files.append(p)

        _write_table_shards(
            "koine-lm-bi",
            [[k1, k2, v] for (k1, k2), v in sorted(self._bi.items())],
        )
        _write_table_shards(
            "koine-lm-tri",
            [[k0, k1, k2, v] for (k0, k1, k2), v in sorted(self._tri.items())],
        )
        _write_table_shards(
            "koine-lm-sbi",
            [[s1, s2, v] for (s1, s2), v in sorted(self._surf_bi.items())],
        )
        _write_table_shards(
            "koine-lm-colloc",
            [[v, a, pmi] for (v, a), pmi in sorted(self._colloc.items())],
        )
        return files

    @classmethod
    def load_shards(cls, out_dir: Path) -> KoineNgramLM:
        """Load LM shards offline."""
        uni: dict[str, float] = {}
        uni_path = out_dir / "koine-lm-uni.json"
        if uni_path.is_file():
            uni.update(
                {
                    k: float(v)
                    for k, v in json.loads(uni_path.read_text(encoding="utf-8")).items()
                }
            )
        for p in sorted(out_dir.glob("koine-lm-uni-*.json")):
            uni.update(
                {
                    k: float(v)
                    for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                }
            )
        bi: dict[tuple[str, str], float] = {}
        for p in sorted(out_dir.glob("koine-lm-bi-*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                bi[(str(r[0]), str(r[1]))] = float(r[2])
        tri: dict[tuple[str, str, str], float] = {}
        for p in sorted(out_dir.glob("koine-lm-tri-*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                tri[(str(r[0]), str(r[1]), str(r[2]))] = float(r[3])
        sbi: dict[tuple[str, str], float] = {}
        for p in sorted(out_dir.glob("koine-lm-sbi-*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                sbi[(str(r[0]), str(r[1]))] = float(r[2])
        colloc: dict[tuple[str, str], float] = {}
        for p in sorted(out_dir.glob("koine-lm-colloc-*.json")):
            for r in json.loads(p.read_text(encoding="utf-8")):
                colloc[(str(r[0]), str(r[1]))] = float(r[2])
        return cls(
            unigram_logp=uni,
            bigram_logp=bi,
            trigram_logp=tri,
            surface_bi_logp=sbi,
            verb_arg_pmi=colloc,
        )
