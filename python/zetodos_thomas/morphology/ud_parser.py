# SPDX-License-Identifier: Apache-2.0
"""Lightweight supervised Sahidic Coptic UD dependency parser.

Trained strictly on ``sahidic.ot_TT.zip`` gold dependency trees (holding out
evaluation books such as Ruth/Jonah, and never reading Thomas or gated NT
corpora). Uses factorized root/head/label statistics and greedy cycle-breaking
so every decoded sequence forms a single-root dependency tree.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit

_CLAUSE_BREAK_POS = frozenset(
    {"V", "VBD", "VIMP", "VSTAT", "EXIST", "COP", "CONJ", "PUNCT"}
)
_ROOT_PRIOR = 0.08
_ARC_PRIOR = 0.05
_PRIOR_STRENGTH = 6.0


def _dir_bucket(ms: Sequence[CopticMorpheme], i: int, h: int) -> str:
    gi = ms[i].group_seq
    gh = ms[h].group_seq
    if gi == gh:
        return "G0+" if h > i else "G0-"
    diff = gh - gi
    if diff == 1:
        return "R1"
    if diff > 1:
        return "R2+"
    if diff == -1:
        return "L1"
    return "L2+"


def _left_marker(ms: Sequence[CopticMorpheme], i: int) -> str:
    for k in range(i - 1, max(-1, i - 4), -1):
        mk = ms[k]
        if mk.pos == "PREP":
            return f"p:{mk.lemma}"
        if mk.pos == "PTC" and mk.lemma == "ⲛϭⲓ":
            return "s:ⲛϭⲓ"
        if mk.pos in {"CREL", "CCIRC"}:
            return f"c:{mk.pos}"
        if mk.pos in _CLAUSE_BREAK_POS:
            break
    return "-"


def _root_key(ms: Sequence[CopticMorpheme], i: int) -> str:
    c_pos = ms[i].pos
    prev_pos = ms[i - 1].pos if i > 0 else "<s>"
    seen_pred = any(ms[k].pos in {"V", "VBD", "VIMP", "EXIST", "COP"} for k in range(i))
    subord = any(
        ms[k].pos in {"CREL", "CCIRC", "CONJ", "ACOND", "APREC", "ALIM"}
        for k in range(max(0, i - 4), i)
    )
    return f"{c_pos}|{prev_pos}|{'1' if seen_pred else '0'}|{'1' if subord else '0'}"


class CopticDependencyParser:
    """Factorized Sahidic Coptic dependency parser trained on OT gold trees."""

    def __init__(
        self,
        *,
        root_logp: dict[str, float],
        root_backoff: dict[str, float],
        arc_logp: dict[str, float],
        arc_backoff: dict[str, float],
        rel_map: dict[str, str],
        rel_backoff: dict[str, str],
    ) -> None:
        self._root_logp = root_logp
        self._root_backoff = root_backoff
        self._arc_logp = arc_logp
        self._arc_backoff = arc_backoff
        self._rel_map = rel_map
        self._rel_backoff = rel_backoff

    @classmethod
    def train(
        cls,
        ot_by_book: Mapping[str, Mapping[tuple[int, int], CopticUnit]],
        *,
        exclude_books: set[str] | None = None,
    ) -> CopticDependencyParser:
        """Train root, arc, and dep_rel statistics strictly from ``ot_by_book``."""
        excluded = set(exclude_books or ())
        root_true: Counter[str] = Counter()
        root_opp: Counter[str] = Counter()
        root_pos_true: Counter[str] = Counter()
        root_pos_opp: Counter[str] = Counter()

        arc_true: Counter[tuple[str, str, str, str, str]] = Counter()
        arc_opp: Counter[tuple[str, str, str, str, str]] = Counter()
        arc_bo_true: Counter[tuple[str, str, str, str]] = Counter()
        arc_bo_opp: Counter[tuple[str, str, str, str]] = Counter()

        rel_counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        rel_bo_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

        for book, verses in sorted(ot_by_book.items()):
            if book in excluded:
                continue
            for (_ch, _vs), unit in sorted(verses.items()):
                ms = unit.morphemes
                n = len(ms)
                if n == 0:
                    continue
                markers = [_left_marker(ms, i) for i in range(n)]
                for i in range(n):
                    m = ms[i]
                    rk = _root_key(ms, i)
                    root_opp[rk] += 1
                    root_pos_opp[m.pos] += 1
                    if m.head_seq is None:
                        root_true[rk] += 1
                        root_pos_true[m.pos] += 1
                        continue

                    gold_h = m.head_seq
                    if gold_h < 0 or gold_h >= n or gold_h == i:
                        continue
                    c_pos = m.pos
                    lm = markers[i]
                    gi = m.group_seq

                    # Sweep leftward from i - 1 down to 0
                    seen_left: set[str] = set()
                    for h in range(i - 1, -1, -1):
                        h_pos = ms[h].pos
                        if h_pos != "PUNCT" or c_pos == "PUNCT":
                            gh = ms[h].group_seq
                            db = (
                                "G0-"
                                if gi == gh
                                else ("L1" if gh - gi == -1 else "L2+")
                            )
                            interv = "1" if h_pos in seen_left else "0"
                            akey = (c_pos, lm, h_pos, db, interv)
                            bkey = (c_pos, h_pos, db, interv)
                            arc_opp[akey] += 1
                            arc_bo_opp[bkey] += 1
                            if h == gold_h:
                                arc_true[akey] += 1
                                arc_bo_true[bkey] += 1
                                if m.dep_rel:
                                    rel_counts[(c_pos, lm, h_pos, db)][m.dep_rel] += 1
                                    rel_bo_counts[(c_pos, h_pos, "-")][m.dep_rel] += 1
                        seen_left.add(h_pos)

                    # Sweep rightward from i + 1 up to n - 1
                    seen_right: set[str] = set()
                    for h in range(i + 1, n):
                        h_pos = ms[h].pos
                        if h_pos != "PUNCT" or c_pos == "PUNCT":
                            gh = ms[h].group_seq
                            db = (
                                "G0+" if gi == gh else ("R1" if gh - gi == 1 else "R2+")
                            )
                            interv = "1" if h_pos in seen_right else "0"
                            akey = (c_pos, lm, h_pos, db, interv)
                            bkey = (c_pos, h_pos, db, interv)
                            arc_opp[akey] += 1
                            arc_bo_opp[bkey] += 1
                            if h == gold_h:
                                arc_true[akey] += 1
                                arc_bo_true[bkey] += 1
                                if m.dep_rel:
                                    rel_counts[(c_pos, lm, h_pos, db)][m.dep_rel] += 1
                                    rel_bo_counts[(c_pos, h_pos, "+")][m.dep_rel] += 1
                        seen_right.add(h_pos)

        root_logp = {
            k: round(
                math.log(
                    (root_true.get(k, 0) + _PRIOR_STRENGTH * _ROOT_PRIOR)
                    / (opp + _PRIOR_STRENGTH)
                ),
                4,
            )
            for k, opp in root_opp.items()
            if opp >= 3
        }
        root_backoff = {
            k: round(
                math.log(
                    (root_pos_true.get(k, 0) + _PRIOR_STRENGTH * _ROOT_PRIOR)
                    / (opp + _PRIOR_STRENGTH)
                ),
                4,
            )
            for k, opp in root_pos_opp.items()
        }
        arc_logp = {
            "|".join(k): round(
                math.log(
                    (arc_true.get(k, 0) + _PRIOR_STRENGTH * _ARC_PRIOR)
                    / (opp + _PRIOR_STRENGTH)
                ),
                4,
            )
            for k, opp in arc_opp.items()
            if opp >= 4 and arc_true.get(k, 0) >= 1
        }
        arc_backoff = {
            "|".join(k): round(
                math.log(
                    (arc_bo_true.get(k, 0) + _PRIOR_STRENGTH * _ARC_PRIOR)
                    / (opp + _PRIOR_STRENGTH)
                ),
                4,
            )
            for k, opp in arc_bo_opp.items()
            if opp >= 3
        }
        rel_map = {
            "|".join(k): cnt.most_common(1)[0][0]
            for k, cnt in rel_counts.items()
            if sum(cnt.values()) >= 2
        }
        rel_backoff = {
            "|".join(k): cnt.most_common(1)[0][0] for k, cnt in rel_bo_counts.items()
        }
        return cls(
            root_logp=root_logp,
            root_backoff=root_backoff,
            arc_logp=arc_logp,
            arc_backoff=arc_backoff,
            rel_map=rel_map,
            rel_backoff=rel_backoff,
        )

    def _score_root(self, ms: Sequence[CopticMorpheme], i: int) -> float:
        rk = _root_key(ms, i)
        if rk in self._root_logp:
            return self._root_logp[rk]
        return self._root_backoff.get(ms[i].pos, -6.0)

    def _score_arc(
        self, ms: Sequence[CopticMorpheme], i: int, h: int, lm: str, interv: str
    ) -> float:
        c_pos = ms[i].pos
        h_pos = ms[h].pos
        db = _dir_bucket(ms, i, h)
        akey = f"{c_pos}|{lm}|{h_pos}|{db}|{interv}"
        sc = self._arc_logp.get(akey)
        if sc is None:
            sc = self._arc_backoff.get(f"{c_pos}|{h_pos}|{db}|{interv}", -6.5)
        dist = abs(ms[h].group_seq - ms[i].group_seq)
        return sc - 0.04 * dist

    def _predict_rel(
        self, ms: Sequence[CopticMorpheme], i: int, h: int, lm: str
    ) -> str:
        c_pos = ms[i].pos
        h_pos = ms[h].pos
        db = _dir_bucket(ms, i, h)
        rel = self._rel_map.get(f"{c_pos}|{lm}|{h_pos}|{db}")
        if rel is not None:
            return rel
        direction = "+" if h > i else "-"
        return self._rel_backoff.get(f"{c_pos}|{h_pos}|{direction}", "dep")

    def annotate_in_place(self, morphemes: list[CopticMorpheme]) -> None:
        """Populate ``head_seq`` and ``dep_rel`` with a single-root valid UD tree."""
        n = len(morphemes)
        if n == 0:
            return
        non_punct = [i for i in range(n) if morphemes[i].pos != "PUNCT"]
        root_cands = non_punct or list(range(n))
        root_idx = max(
            root_cands, key=lambda idx: (self._score_root(morphemes, idx), -idx)
        )
        markers = [_left_marker(morphemes, i) for i in range(n)]

        ranked_cands: dict[int, list[tuple[int, float]]] = {}
        parent: list[int | None] = [None] * n
        for i in range(n):
            if i == root_idx:
                continue
            lm = markers[i]
            cands: list[tuple[int, float]] = []
            seen_left: set[str] = set()
            for h in range(i - 1, -1, -1):
                h_pos = morphemes[h].pos
                if h_pos != "PUNCT" or morphemes[i].pos == "PUNCT":
                    interv = "1" if h_pos in seen_left else "0"
                    cands.append((h, self._score_arc(morphemes, i, h, lm, interv)))
                seen_left.add(h_pos)
            seen_right: set[str] = set()
            for h in range(i + 1, n):
                h_pos = morphemes[h].pos
                if h_pos != "PUNCT" or morphemes[i].pos == "PUNCT":
                    interv = "1" if h_pos in seen_right else "0"
                    cands.append((h, self._score_arc(morphemes, i, h, lm, interv)))
                seen_right.add(h_pos)
            if not cands:
                cands = [(root_idx, -10.0)]
            cands.sort(key=lambda item: (-item[1], abs(item[0] - i), item[0]))
            ranked_cands[i] = cands
            parent[i] = cands[0][0]

        # Break any cycles greedily so every node reaches root_idx
        while True:
            reaches_root = {root_idx}
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if i in reaches_root:
                        continue
                    p = parent[i]
                    if p is not None and p in reaches_root:
                        reaches_root.add(i)
                        changed = True
            if len(reaches_root) == n:
                break
            best_fix: tuple[float, int, int] | None = None
            for i in range(n):
                if i in reaches_root:
                    continue
                top_sc = ranked_cands[i][0][1]
                chosen_h = root_idx
                chosen_sc = -1e9
                for ch, csc in ranked_cands[i]:
                    if ch in reaches_root:
                        chosen_h = ch
                        chosen_sc = csc
                        break
                loss = top_sc - chosen_sc
                if best_fix is None or loss < best_fix[0]:
                    best_fix = (loss, i, chosen_h)
            if best_fix is None:
                break
            parent[best_fix[1]] = best_fix[2]

        for i, m in enumerate(morphemes):
            if i == root_idx:
                m.head_seq = None
                m.dep_rel = "root"
            else:
                p_val = parent[i]
                h = p_val if p_val is not None else root_idx
                m.head_seq = h
                m.dep_rel = self._predict_rel(morphemes, i, h, markers[i])

    def to_dict(self) -> dict[str, object]:
        """Serialize parser tables to a compact JSON-ready dictionary."""
        return {
            "root_logp": dict(sorted(self._root_logp.items())),
            "root_backoff": dict(sorted(self._root_backoff.items())),
            "arc_logp": dict(sorted(self._arc_logp.items())),
            "arc_backoff": dict(sorted(self._arc_backoff.items())),
            "rel_map": dict(sorted(self._rel_map.items())),
            "rel_backoff": dict(sorted(self._rel_backoff.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CopticDependencyParser:
        """Load parser tables from ``to_dict()`` dictionary."""

        def _fmap(key: str) -> dict[str, float]:
            raw = data.get(key)
            if not isinstance(raw, Mapping):
                return {}
            return {
                str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))
            }

        def _smap(key: str) -> dict[str, str]:
            raw = data.get(key)
            if not isinstance(raw, Mapping):
                return {}
            return {str(k): str(v) for k, v in raw.items()}

        return cls(
            root_logp=_fmap("root_logp"),
            root_backoff=_fmap("root_backoff"),
            arc_logp=_fmap("arc_logp"),
            arc_backoff=_fmap("arc_backoff"),
            rel_map=_smap("rel_map"),
            rel_backoff=_smap("rel_backoff"),
        )

    def save_shards(self, out_dir: Path, *, max_bytes: int = 680_000) -> list[Path]:
        """Save dependency parser tables as ``coptic-ud-*.json`` (< 1024 KiB each)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("coptic-ud-*.json"):
            old.unlink()

        files: list[Path] = []
        raw_dict = self.to_dict()
        payload = json.dumps(raw_dict, ensure_ascii=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) <= max_bytes:
            p = out_dir / "coptic-ud-0.json"
            p.write_text(payload + "\n", encoding="utf-8")
            return [p]

        # Fall back to chunked sharding of arc_logp if tables grow larger
        arc_items = sorted(self._arc_logp.items())
        meta_obj = {k: v for k, v in raw_dict.items() if k != "arc_logp"}
        meta_p = out_dir / "coptic-ud-0.json"
        chunk: dict[str, float] = {}
        est = len(
            json.dumps(meta_obj, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        idx = 0
        for k, v in arc_items:
            row_sz = len(k.encode("utf-8")) + 12
            if est + row_sz > max_bytes and chunk:
                p = out_dir / f"coptic-ud-{idx}.json"
                out_obj = dict(meta_obj) if idx == 0 else {}
                out_obj["arc_logp"] = chunk
                p.write_text(
                    json.dumps(out_obj, ensure_ascii=False, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                files.append(p)
                chunk = {}
                idx += 1
                est = 16
            chunk[k] = v
            est += row_sz
        p = out_dir / f"coptic-ud-{idx}.json"
        out_obj = dict(meta_obj) if idx == 0 else {}
        out_obj["arc_logp"] = chunk
        p.write_text(
            json.dumps(out_obj, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        files.append(p)
        _ = meta_p
        return files

    @classmethod
    def load_shards(cls, out_dir: Path) -> CopticDependencyParser | None:
        """Load ``coptic-ud-*.json`` shards if present in ``out_dir``."""
        paths = sorted(out_dir.glob("coptic-ud-*.json"))
        if not paths:
            return None
        merged: dict[str, object] = {}
        arc_all: dict[str, float] = {}
        for p in paths:
            part = json.loads(p.read_text(encoding="utf-8"))
            for k, v in part.items():
                if k == "arc_logp" and isinstance(v, dict):
                    for ak, av in v.items():
                        arc_all[str(ak)] = float(av)
                else:
                    merged[str(k)] = v
        merged["arc_logp"] = arc_all
        return cls.from_dict(merged)
