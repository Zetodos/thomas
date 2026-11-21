# SPDX-License-Identifier: Apache-2.0
"""Sahidic Coptic bound-group segmenter, lemmatizer, and POS tagger.

Trained strictly on ``sahidic.ot_TT.zip`` (CC-BY-SA-4.0) and ``CCL v1.2``
(CC-BY-SA-4.0) - excludes Gospel of Thomas and all gated/non-default corpora.
Uses empirical bound-group distributions, CCL/OT stem & construct tables,
a data-driven intra-group character-lattice DP segmenter, and smoothed trigram
POS Viterbi disambiguation.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.ingest.ot_swete import load_sahidic_ot_chapters
from zetodos_thomas.morphology.ud_parser import CopticDependencyParser
from zetodos_thomas.schemas.ids import token_id
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit
from zetodos_thomas.schemas.source import TextUnit, UnitKind

_TEI_NS = "{http://www.tei-c.org/ns/1.0}"
_PUNCT_EDGE_RE = re.compile(r"^([·.,;;:!?«»\"'()\[\]\-]+)|([·.,;;:!?«»\"'()\[\]\-]+)$")
_SPACE_RE = re.compile(r"\s+")
_GEMINATE_RE = re.compile(r"(.)\1+")

MorphemeSpec = tuple[str, str, str]  # (norm, lemma, pos)

_CLOSED_PROCLITIC_POS = {
    "PREP",
    "ART",
    "PPOS",
    "PDEM",
    "PPERS",
    "APST",
    "FUT",
    "CCIRC",
    "CREL",
    "CFOC",
    "CPRET",
    "ACONJ",
    "ACOND",
    "ACAUS",
    "APREC",
    "ALIM",
    "AJUS",
    "AAOR",
    "ANEGPST",
    "ANEGAOR",
    "ANEGJUS",
    "ANY",
    "NEG",
    "CONJ",
    "PTC",
    "COP",
    "VBD",
    "AOPT_PPERS",
    "ACONJ_PPERS",
    "AFUTCONJ",
    "EXIST",
    "PINT",
}

# Standard CopSahHorner nomina sacra expansions when overline/context signals abbrev.
_NOMINA_SACRA: dict[str, list[MorphemeSpec]] = {
    "ⲓⲥ": [("ⲓⲥ", "ⲓⲏⲥⲟⲩⲥ", "NPROP")],
    "ⲭⲥ": [("ⲭⲥ", "ⲭⲣⲓⲥⲧⲟⲥ", "NPROP")],
    "ⲡⲛⲁ": [("ⲡⲛⲁ", "ⲡⲛⲉⲩⲙⲁ", "N")],
    "ⲕⲥ": [("ⲕⲥ", "ϫⲟⲉⲓⲥ", "N")],
    "ϫⲥ": [("ϫⲥ", "ϫⲟⲉⲓⲥ", "N")],
    "ⲓⲏⲗ": [("ⲓⲏⲗ", "ⲓⲥⲣⲁⲏⲗ", "NPROP")],
    "ⲓⲗⲏⲙ": [("ⲓⲗⲏⲙ", "ϩⲓⲉⲣⲟⲩⲥⲁⲗⲏⲙ", "NPROP")],
    "ⲑⲓⲗⲏⲙ": [("ⲧ", "ⲡ", "ART"), ("ϩⲓⲗⲏⲙ", "ϩⲓⲉⲣⲟⲩⲥⲁⲗⲏⲙ", "NPROP")],
    "ⲑⲓⲉⲗⲏⲙ": [("ⲧ", "ⲡ", "ART"), ("ϩⲓⲉⲗⲏⲙ", "ϩⲓⲉⲣⲟⲩⲥⲁⲗⲏⲙ", "NPROP")],
    "ⲡⲉⲭⲥ": [("ⲡⲉ", "ⲡ", "ART"), ("ⲭⲥ", "ⲭⲣⲓⲥⲧⲟⲥ", "NPROP")],
    "ⲡⲭⲥ": [("ⲡ", "ⲡ", "ART"), ("ⲭⲥ", "ⲭⲣⲓⲥⲧⲟⲥ", "NPROP")],
    "ⲙⲡⲉⲭⲥ": [("ⲙ", "ⲛ", "PREP"), ("ⲡⲉ", "ⲡ", "ART"), ("ⲭⲥ", "ⲭⲣⲓⲥⲧⲟⲥ", "NPROP")],
    "ⲙⲡⲭⲥ": [("ⲙ", "ⲛ", "PREP"), ("ⲡ", "ⲡ", "ART"), ("ⲭⲥ", "ⲭⲣⲓⲥⲧⲟⲥ", "NPROP")],
    "ⲛⲓⲥ": [("ⲛ", "ⲛ", "PREP"), ("ⲓⲥ", "ⲓⲏⲥⲟⲩⲥ", "NPROP")],
    "ⲙⲡⲛⲁ": [("ⲙ", "ⲛ", "PREP"), ("ⲡⲛⲁ", "ⲡⲛⲉⲩⲙⲁ", "N")],
    "ⲡⲉⲡⲛⲁ": [("ⲡⲉ", "ⲡ", "ART"), ("ⲡⲛⲁ", "ⲡⲛⲉⲩⲙⲁ", "N")],
    "ⲡⲡⲛⲁ": [("ⲡ", "ⲡ", "ART"), ("ⲡⲛⲁ", "ⲡⲛⲉⲩⲙⲁ", "N")],
}

_CCL_POS_MAP: dict[str, str] = {
    "Subst.": "N",
    "Vb.": "V",
    "trans. Vb.": "V",
    "intrans. Vb.": "V",
    "unpers. Vb.": "V",
    "Qual.": "VSTAT",
    "Präp.": "PREP",
    "Adv.": "ADV",
    "Konj.": "CONJ",
    "Interj.": "PTC",
    "Part.": "PTC",
    "Pron.": "PPERI",
    "Dem.": "PDEM",
    "Zahlwort": "NUM",
    "Zahlzeichen": "NUM",
    "Adj.": "N",
    "Eigenn.": "NPROP",
}

_GREEK_LOAN_ENDINGS = (
    "ⲟⲥ",
    "ⲟⲛ",
    "ⲏⲥ",
    "ⲁⲥ",
    "ⲓⲁ",
    "ⲉⲓⲁ",
    "ⲱⲛ",
    "ⲓⲥ",
    "ⲙⲁ",
    "ⲟⲩ",
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _strip_edge_punct(raw: str) -> tuple[str, str, str]:
    """Split leading and trailing punctuation around a Coptic bound group."""
    left = ""
    right = ""
    core = raw
    while core:
        m = _PUNCT_EDGE_RE.search(core)
        if m is None:
            break
        if m.group(1):
            left += m.group(1)
            core = core[len(m.group(1)) :]
        elif m.group(2):
            right = m.group(2) + right
            core = core[: -len(m.group(2))]
    return left, core, right


def _extract_ccl_stems(ccl_xml: Path) -> dict[str, list[tuple[str, str, float]]]:
    """Extract normalized Sahidic/general stems & forms -> (lemma, pos, log_w)."""
    stems: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    tree = ET.parse(ccl_xml)
    root = tree.getroot()
    for entry in root.findall(f".//{_TEI_NS}entry"):
        pos_el = entry.find(f"{_TEI_NS}gramGrp/{_TEI_NS}pos")
        raw_pos = pos_el.text.strip() if pos_el is not None and pos_el.text else ""
        pos = _CCL_POS_MAP.get(raw_pos, "")
        if not pos:
            is_foreign = entry.attrib.get("type") == "foreign" or (
                entry.find(f"{_TEI_NS}etym") is not None
            )
            pos = "N" if is_foreign else ""
        if not pos:
            continue

        lemma_form = ""
        orths: list[tuple[str, str, bool]] = []
        for form in entry.findall(f"{_TEI_NS}form"):
            change = form.attrib.get("change", "")
            if "deprecated" in change:
                continue
            orth_el = form.find(f"{_TEI_NS}orth")
            if orth_el is None or not orth_el.text:
                continue
            raw_orth = _nfc(orth_el.text.strip())
            if "(" in raw_orth or " " in raw_orth or not raw_orth:
                continue
            norm_orth = normalize_coptic(raw_orth)
            if not norm_orth:
                continue
            usg_el = form.find(f"{_TEI_NS}usg[@type='geo']")
            geo = usg_el.text.strip() if usg_el is not None and usg_el.text else ""
            if form.attrib.get("type") == "lemma" and not lemma_form:
                lemma_form = norm_orth
            if geo and "S" not in geo and "?" not in geo:
                continue
            is_inflected = form.attrib.get("type") == "inflected"
            orths.append((norm_orth, geo, is_inflected))

        if not lemma_form and orths:
            lemma_form = orths[0][0]
        if not lemma_form:
            continue
        for surf, geo, is_infl in orths:
            weight = 1.0 if ("S" in geo or not geo) else 0.75
            if is_infl:
                weight *= 0.88
            key = (lemma_form, pos)
            if weight > stems[surf].get(key, 0.0):
                stems[surf][key] = weight

    out: dict[str, list[tuple[str, str, float]]] = {}
    for surf, pairs in stems.items():
        ranked = sorted(pairs.items(), key=lambda item: -item[1])
        out[surf] = [(lemma, pos, math.log(w) - 1.0) for (lemma, pos), w in ranked]
    return out


def _apply_crasis_sandhi(rest: str) -> list[tuple[MorphemeSpec, str, float]]:
    """Expand Greek-Coptic aspirated crasis (ⲑ=ⲧ+ϩ, ⲫ=ⲡ+ϩ)."""
    variants: list[tuple[MorphemeSpec, str, float]] = []
    if rest.startswith("ⲑ") and len(rest) >= 2:
        variants.append((("ⲧ", "ⲡ", "ART"), "ϩ" + rest[1:], 0.85))
    if rest.startswith("ⲫ") and len(rest) >= 2:
        variants.append((("ⲡ", "ⲡ", "ART"), "ϩ" + rest[1:], 0.9))
    return variants


@dataclass
class _TrainedCopticTables:
    group_lex: dict[str, list[tuple[tuple[MorphemeSpec, ...], float]]]
    stem_lex: dict[str, list[tuple[str, str, float]]]
    proclitic_lex: dict[str, list[tuple[str, str, str, float]]]
    suffix_lex: dict[str, list[tuple[str, str, str, float]]]
    intra_bigram: dict[tuple[str, str], float]
    bigram_pos: dict[tuple[str, str], float]
    trigram_pos: dict[tuple[str, str, str], float]
    unigram_pos: dict[str, float]
    morph_pos_logp: dict[tuple[str, str], float]


class CopticAnalyzer:
    """Segment, lemmatize, and POS-tag Sahidic Coptic (Horner NT or raw text)."""

    def __init__(
        self,
        tables: _TrainedCopticTables,
        ud_parser: CopticDependencyParser | None = None,
    ) -> None:
        self._tables = tables
        self._ud_parser = ud_parser
        self._max_proclitic_len = max((len(k) for k in tables.proclitic_lex), default=8)
        self._max_suffix_len = max((len(k) for k in tables.suffix_lex), default=6)

    @classmethod
    def train(
        cls,
        *,
        ot_zip: Path,
        ccl_xml: Path | None = None,
        exclude_books: set[str] | None = None,
    ) -> CopticAnalyzer:
        """Train analyzer strictly from sahidic.ot_TT.zip + optional CCL v1.2."""
        excluded = set(exclude_books or ())
        ot_by_book = load_sahidic_ot_chapters(ot_zip)

        group_counts: dict[str, Counter[tuple[MorphemeSpec, ...]]] = defaultdict(
            Counter
        )
        stem_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        proclitic_counts: dict[str, Counter[MorphemeSpec]] = defaultdict(Counter)
        suffix_counts: dict[str, Counter[MorphemeSpec]] = defaultdict(Counter)
        intra_bi_counts: Counter[tuple[str, str]] = Counter()
        intra_prev_totals: Counter[str] = Counter()
        uni_pos: Counter[str] = Counter()
        bi_pos: Counter[tuple[str, str]] = Counter()
        tri_pos: Counter[tuple[str, str, str]] = Counter()
        morph_pos: Counter[tuple[str, str]] = Counter()

        for book, verses in sorted(ot_by_book.items()):
            if book in excluded:
                continue
            for (_ch, _vs), unit in sorted(verses.items()):
                by_g: dict[int, list[CopticMorpheme]] = defaultdict(list)
                pos_seq: list[str] = ["<s>", "<s>"]
                for m in unit.morphemes:
                    by_g[m.group_seq].append(m)
                    stem_counts[m.norm][(m.lemma, m.pos)] += 1
                    uni_pos[m.pos] += 1
                    morph_pos[(m.norm, m.pos)] += 1
                    pos_seq.append(m.pos)
                pos_seq.append("</s>")
                for i in range(2, len(pos_seq)):
                    bi_pos[(pos_seq[i - 1], pos_seq[i])] += 1
                    tri_pos[(pos_seq[i - 2], pos_seq[i - 1], pos_seq[i])] += 1

                raw_orig_groups = unit.original_text.split()
                for g_idx, grp_morphs in sorted(by_g.items()):
                    sig = tuple((m.norm, m.lemma, m.pos) for m in grp_morphs)
                    ng = normalize_coptic("".join(m.norm for m in grp_morphs))
                    if ng:
                        group_counts[ng][sig] += 1
                    if 0 <= g_idx < len(raw_orig_groups):
                        raw_ng = normalize_coptic(raw_orig_groups[g_idx])
                        if raw_ng and raw_ng != ng:
                            group_counts[raw_ng][sig] += 1

                    # Intra-group transitions & proclitic/suffix distributions
                    g_pos = ["<g>", *[m.pos for m in grp_morphs], "</g>"]
                    for j in range(1, len(g_pos)):
                        intra_bi_counts[(g_pos[j - 1], g_pos[j])] += 1
                        intra_prev_totals[g_pos[j - 1]] += 1

                    if len(grp_morphs) >= 2:
                        for m in grp_morphs[:-1]:
                            if m.pos in _CLOSED_PROCLITIC_POS:
                                proclitic_counts[m.norm][(m.norm, m.lemma, m.pos)] += 1
                        last_m = grp_morphs[-1]
                        if last_m.pos == "PPERO":
                            suffix_counts[last_m.norm][
                                (last_m.norm, last_m.lemma, last_m.pos)
                            ] += 1

        group_lex: dict[str, list[tuple[tuple[MorphemeSpec, ...], float]]] = {}
        for ng, cnt in group_counts.items():
            total = sum(cnt.values())
            group_lex[ng] = [
                (sig, math.log((c + 0.25) / (total + 0.5)))
                for sig, c in cnt.most_common(6)
            ]

        proclitic_lex: dict[str, list[tuple[str, str, str, float]]] = {}
        for surf, cnt in proclitic_counts.items():
            surf_tot = max(sum(cnt.values()), 1)
            entries = [
                (
                    n,
                    lemma,
                    pos,
                    -0.35
                    + 0.25 * math.log((c + 0.5) / (surf_tot + 1.0))
                    + 0.08 * min(math.log1p(c), 4.0)
                    - (0.2 if len(surf) == 1 else 0.0),
                )
                for (n, lemma, pos), c in cnt.most_common(4)
                if c >= 3
            ]
            if entries:
                proclitic_lex[surf] = entries

        # Surface contraction ⲩ = ⲟⲩ (ART after ⲉ/ⲁ/ϣⲁ/ϩⲁ/ⲛⲥⲁ/ⲛⲁ or PPERS ⲛⲧⲟⲟⲩ)
        u_list = proclitic_lex.get("ⲩ", [])
        if not any(pos == "ART" for _, _, pos, _ in u_list):
            u_list.append(("ⲟⲩ", "ⲟⲩ", "ART", -0.65))
        proclitic_lex["ⲩ"] = u_list

        suffix_lex: dict[str, list[tuple[str, str, str, float]]] = {}
        for surf, cnt in suffix_counts.items():
            suf_tot = max(sum(cnt.values()), 1)
            entries = [
                (
                    n,
                    lemma,
                    pos,
                    -0.35 + 0.25 * math.log((c + 0.5) / (suf_tot + 1.0)),
                )
                for (n, lemma, pos), c in cnt.most_common(2)
                if c >= 3
            ]
            if entries:
                suffix_lex[surf] = entries

        intra_denom = max(len(uni_pos) + 2, 1)
        intra_bigram: dict[tuple[str, str], float] = {
            (p1, p2): 0.25
            * math.log((c + 0.2) / (intra_prev_totals[p1] + 0.2 * intra_denom))
            for (p1, p2), c in intra_bi_counts.items()
        }

        merged_stems: dict[str, list[tuple[str, str, float]]] = {}
        ccl_stems = _extract_ccl_stems(ccl_xml) if ccl_xml and ccl_xml.is_file() else {}
        all_norms = set(stem_counts) | set(ccl_stems)
        for norm_tok in all_norms:
            combined: dict[tuple[str, str], float] = {}
            ot_cnt = stem_counts.get(norm_tok)
            if ot_cnt:
                tot = sum(ot_cnt.values())
                for (lemma, pos), c in ot_cnt.most_common(5):
                    bonus = min(1.3, 0.25 * math.log1p(c))
                    combined[(lemma, pos)] = math.log((c + 0.1) / (tot + 0.5)) + bonus
            for lemma, pos, log_w in ccl_stems.get(norm_tok, ()):
                if (lemma, pos) not in combined:
                    combined[(lemma, pos)] = log_w - 0.15
            merged_stems[norm_tok] = [
                (lemma, pos, score)
                for (lemma, pos), score in sorted(
                    combined.items(), key=lambda x: -x[1]
                )[:6]
            ]

        total_toks = max(sum(uni_pos.values()), 1)
        vocab_pos = max(len(uni_pos), 1)
        unigram_pos = {
            pos: math.log((c + 1.0) / (total_toks + vocab_pos))
            for pos, c in uni_pos.items()
        }
        unigram_pos["</s>"] = -2.0

        bi_totals: Counter[str] = Counter()
        for (p1, _p2), c in bi_pos.items():
            bi_totals[p1] += c
        bigram_pos = {
            (p1, p2): math.log((c + 0.2) / (bi_totals[p1] + 0.2 * vocab_pos))
            for (p1, p2), c in bi_pos.items()
        }

        tri_totals: Counter[tuple[str, str]] = Counter()
        for (p0, p1, _p2), c in tri_pos.items():
            tri_totals[(p0, p1)] += c
        trigram_pos = {
            (p0, p1, p2): math.log(
                (c + 0.15) / (tri_totals[(p0, p1)] + 0.15 * vocab_pos)
            )
            for (p0, p1, p2), c in tri_pos.items()
        }

        pos_tok_totals: Counter[str] = Counter()
        for (_n, pos), c in morph_pos.items():
            pos_tok_totals[pos] += c
        morph_pos_logp = {
            (n, pos): math.log((c + 0.1) / (pos_tok_totals[pos] + 10.0))
            for (n, pos), c in morph_pos.items()
        }

        ud_parser = CopticDependencyParser.train(ot_by_book, exclude_books=excluded)

        return cls(
            _TrainedCopticTables(
                group_lex=group_lex,
                stem_lex=merged_stems,
                proclitic_lex=proclitic_lex,
                suffix_lex=suffix_lex,
                intra_bigram=intra_bigram,
                bigram_pos=bigram_pos,
                trigram_pos=trigram_pos,
                unigram_pos=unigram_pos,
                morph_pos_logp=morph_pos_logp,
            ),
            ud_parser=ud_parser,
        )

    def save_json(self, path: Path) -> None:
        """Persist analyzer tables as compact JSON."""
        payload = {
            "group_lex": {
                k: [[[list(m) for m in sig], round(lp, 4)] for sig, lp in v]
                for k, v in self._tables.group_lex.items()
            },
            "stem_lex": {
                k: [[lemma, pos, round(lp, 4)] for lemma, pos, lp in v]
                for k, v in self._tables.stem_lex.items()
            },
            "proclitic_lex": {
                k: [[n, lemma, pos, round(lp, 4)] for n, lemma, pos, lp in v]
                for k, v in self._tables.proclitic_lex.items()
            },
            "suffix_lex": {
                k: [[n, lemma, pos, round(lp, 4)] for n, lemma, pos, lp in v]
                for k, v in self._tables.suffix_lex.items()
            },
            "intra_bigram": [
                [p1, p2, round(v, 4)]
                for (p1, p2), v in self._tables.intra_bigram.items()
            ],
            "unigram_pos": {
                k: round(v, 4) for k, v in self._tables.unigram_pos.items()
            },
            "bigram_pos": [
                [p1, p2, round(v, 4)] for (p1, p2), v in self._tables.bigram_pos.items()
            ],
            "trigram_pos": [
                [p0, p1, p2, round(v, 4)]
                for (p0, p1, p2), v in self._tables.trigram_pos.items()
            ],
            "morph_pos_logp": [
                [n, pos, round(v, 4)]
                for (n, pos), v in self._tables.morph_pos_logp.items()
            ],
            "ud_parser": (
                self._ud_parser.to_dict() if self._ud_parser is not None else None
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> CopticAnalyzer:
        """Load pre-trained analyzer tables from compact JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        group_lex: dict[str, list[tuple[tuple[MorphemeSpec, ...], float]]] = {}
        for k, cands in data["group_lex"].items():
            group_lex[k] = [
                (tuple((m[0], m[1], m[2]) for m in sig), float(lp)) for sig, lp in cands
            ]
        stem_lex = {
            k: [(c[0], c[1], float(c[2])) for c in cands]
            for k, cands in data["stem_lex"].items()
        }
        proclitic_lex = {
            k: [(c[0], c[1], c[2], float(c[3])) for c in cands]
            for k, cands in data["proclitic_lex"].items()
        }
        suffix_lex = {
            k: [(c[0], c[1], c[2], float(c[3])) for c in cands]
            for k, cands in data["suffix_lex"].items()
        }
        ud_raw = data.get("ud_parser")
        ud_parser = (
            CopticDependencyParser.from_dict(ud_raw)
            if isinstance(ud_raw, dict)
            else None
        )
        return cls(
            _TrainedCopticTables(
                group_lex=group_lex,
                stem_lex=stem_lex,
                proclitic_lex=proclitic_lex,
                suffix_lex=suffix_lex,
                intra_bigram={(p1, p2): float(v) for p1, p2, v in data["intra_bigram"]},
                unigram_pos={k: float(v) for k, v in data["unigram_pos"].items()},
                bigram_pos={(p1, p2): float(v) for p1, p2, v in data["bigram_pos"]},
                trigram_pos={
                    (p0, p1, p2): float(v) for p0, p1, p2, v in data["trigram_pos"]
                },
                morph_pos_logp={
                    (n, pos): float(v) for n, pos, v in data["morph_pos_logp"]
                },
            ),
            ud_parser=ud_parser,
        )

    def save_shards(self, out_dir: Path, *, max_bytes: int = 680_000) -> list[Path]:
        """Persist analyzer tables sharded cleanly so each file < 1024 KiB."""
        out_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []

        def _shard_dict(prefix: str, items: list[tuple[str, object]]) -> None:
            for old in out_dir.glob(f"{prefix}-*.json"):
                old.unlink()
            chunk: dict[str, object] = {}
            idx = 0
            est = 2
            for k, val in items:
                sval = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
                esz = len(k.encode("utf-8")) + len(sval.encode("utf-8")) + 6
                if est + esz > max_bytes and chunk:
                    p = out_dir / f"{prefix}-{idx}.json"
                    p.write_text(
                        json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    files.append(p)
                    chunk = {}
                    idx += 1
                    est = 2
                chunk[k] = val
                est += esz
            if chunk or idx == 0:
                p = out_dir / f"{prefix}-{idx}.json"
                p.write_text(
                    json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                files.append(p)

        _shard_dict(
            "coptic-groups",
            [
                (k, [[[list(m) for m in sig], round(lp, 4)] for sig, lp in v[:3]])
                for k, v in sorted(self._tables.group_lex.items())
            ],
        )
        _shard_dict(
            "coptic-stems",
            [
                (k, [[lem, pos, round(lp, 4)] for lem, pos, lp in v[:4]])
                for k, v in sorted(self._tables.stem_lex.items())
            ],
        )
        meta_p = out_dir / "coptic-meta.json"
        meta = {
            "proclitic_lex": {
                k: [[n, lemma, pos, round(lp, 4)] for n, lemma, pos, lp in v]
                for k, v in self._tables.proclitic_lex.items()
            },
            "suffix_lex": {
                k: [[n, lemma, pos, round(lp, 4)] for n, lemma, pos, lp in v]
                for k, v in self._tables.suffix_lex.items()
            },
            "intra_bigram": [
                [p1, p2, round(v, 4)]
                for (p1, p2), v in self._tables.intra_bigram.items()
            ],
            "unigram_pos": {
                k: round(v, 4) for k, v in self._tables.unigram_pos.items()
            },
            "bigram_pos": [
                [p1, p2, round(v, 4)] for (p1, p2), v in self._tables.bigram_pos.items()
            ],
            "trigram_pos": [
                [p0, p1, p2, round(v, 4)]
                for (p0, p1, p2), v in self._tables.trigram_pos.items()
            ],
            "morph_pos_logp": [
                [n, pos, round(v, 4)]
                for (n, pos), v in self._tables.morph_pos_logp.items()
            ],
        }
        meta_p.write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        files.append(meta_p)
        if self._ud_parser is not None:
            files.extend(self._ud_parser.save_shards(out_dir, max_bytes=max_bytes))
        else:
            for old in out_dir.glob("coptic-ud-*.json"):
                old.unlink()
        return files

    @classmethod
    def load_shards(cls, out_dir: Path) -> CopticAnalyzer:
        """Load sharded Coptic analyzer tables cleanly."""
        group_lex: dict[str, list[tuple[tuple[MorphemeSpec, ...], float]]] = {}
        for p in sorted(out_dir.glob("coptic-groups-*.json")):
            for k, cands in json.loads(p.read_text(encoding="utf-8")).items():
                group_lex[k] = [
                    (tuple((m[0], m[1], m[2]) for m in sig), float(lp))
                    for sig, lp in cands
                ]
        stem_lex: dict[str, list[tuple[str, str, float]]] = {}
        for p in sorted(out_dir.glob("coptic-stems-*.json")):
            for k, cands in json.loads(p.read_text(encoding="utf-8")).items():
                stem_lex[k] = [(c[0], c[1], float(c[2])) for c in cands]
        data = json.loads((out_dir / "coptic-meta.json").read_text(encoding="utf-8"))
        ud_parser = CopticDependencyParser.load_shards(out_dir)
        return cls(
            _TrainedCopticTables(
                group_lex=group_lex,
                stem_lex=stem_lex,
                proclitic_lex={
                    k: [(c[0], c[1], c[2], float(c[3])) for c in cands]
                    for k, cands in data["proclitic_lex"].items()
                },
                suffix_lex={
                    k: [(c[0], c[1], c[2], float(c[3])) for c in cands]
                    for k, cands in data["suffix_lex"].items()
                },
                intra_bigram={(p1, p2): float(v) for p1, p2, v in data["intra_bigram"]},
                unigram_pos={k: float(v) for k, v in data["unigram_pos"].items()},
                bigram_pos={(p1, p2): float(v) for p1, p2, v in data["bigram_pos"]},
                trigram_pos={
                    (p0, p1, p2): float(v) for p0, p1, p2, v in data["trigram_pos"]
                },
                morph_pos_logp={
                    (n, pos): float(v) for n, pos, v in data["morph_pos_logp"]
                },
            ),
            ud_parser=ud_parser,
        )

    def _intra(self, p1: str, p2: str) -> float:
        return self._tables.intra_bigram.get((p1, p2), -6.0)

    def _stem_hits(self, seg: str) -> list[tuple[str, str, float]]:
        hits = self._tables.stem_lex.get(seg)
        if hits:
            return hits
        if len(seg) >= 5:
            simplified = _GEMINATE_RE.sub(r"\1", seg)
            if simplified != seg:
                alt = self._tables.stem_lex.get(simplified)
                if alt:
                    return [(lemma, pos, sc - 0.5) for lemma, pos, sc in alt]
        return []

    def _terminal_candidates(
        self,
        rest: str,
        prev_pos: str,
        start_idx: int,
        doc_bare_stems: set[str],
    ) -> list[tuple[tuple[MorphemeSpec, ...], float]]:
        """Score stem/suffix completions from ``rest`` (char offset ``start_idx``)."""
        ends: list[tuple[tuple[MorphemeSpec, ...], float]] = []

        # 1. Exact stem (or geminate-relaxed stem)
        for lemma, pos, stem_lp in self._stem_hits(rest)[:3]:
            sc = stem_lp + self._intra(prev_pos, pos) + self._intra(pos, "</g>")
            ends.append((((rest, lemma, pos),), sc))

        # 2. Stem + pronominal suffix (suffix_lex at end)
        max_s = min(self._max_suffix_len, len(rest) - 2)
        for k in range(max_s, 0, -1):
            suf = rest[-k:]
            stem = rest[:-k]
            suf_hits = self._tables.suffix_lex.get(suf)
            if not suf_hits:
                continue
            for s_norm, s_lemma, s_pos, s_lp in suf_hits:
                end_tr = self._intra(s_pos, "</g>")
                for lemma, pos, stem_lp in self._stem_hits(stem)[:2]:
                    if pos in {"V", "PREP", "N", "PDEM", "PPOS", "IMOD", "NUM"}:
                        sc = (
                            stem_lp
                            + s_lp
                            + self._intra(prev_pos, pos)
                            + self._intra(pos, s_pos)
                            + end_tr
                        )
                        ends.append(
                            (((stem, lemma, pos), (s_norm, s_lemma, s_pos)), sc)
                        )
                # Direct prepositional/modal pronominal base from proclitic_lex
                for p_n, p_lem, p_pos, p_lp in self._tables.proclitic_lex.get(stem, ()):
                    if p_pos in {"PREP", "IMOD", "PPOS"}:
                        sc = (
                            p_lp
                            + s_lp
                            + self._intra(prev_pos, p_pos)
                            + self._intra(p_pos, s_pos)
                            + end_tr
                            - 0.2
                        )
                        ends.append(
                            (((p_n, p_lem, p_pos), (s_norm, s_lemma, s_pos)), sc)
                        )

        # 3. Crasis ⲑ->ⲧ+ϩ, ⲫ->ⲡ+ϩ
        for (a_n, a_l, a_p), uncrased, c_cost in _apply_crasis_sandhi(rest):
            for lemma, pos, stem_lp in self._stem_hits(uncrased)[:2]:
                sc = (
                    -c_cost
                    + stem_lp
                    + self._intra(prev_pos, a_p)
                    + self._intra(a_p, pos)
                    + self._intra(pos, "</g>")
                )
                ends.append((((a_n, a_l, a_p), (uncrased, lemma, pos)), sc))

        # 4. Ordinal ⲙⲉϩ + known NUM or light verb ⲣ + known/Greek-loan stem
        if rest.startswith("ⲙⲉϩ") and len(rest) >= 5:
            inner = rest[3:]
            if any(p == "NUM" for _, p, _ in self._stem_hits(inner)):
                sc = -1.1 + self._intra(prev_pos, "NUM") + self._intra("NUM", "</g>")
                ends.append((((rest, rest, "NUM"),), sc))
        if rest.startswith("ⲣ") and len(rest) >= 3:
            inner = rest[1:]
            if self._stem_hits(inner) or any(
                inner.endswith(e) for e in _GREEK_LOAN_ENDINGS
            ):
                sc = -1.2 + self._intra(prev_pos, "V") + self._intra("V", "</g>")
                ends.append((((rest, rest, "V"),), sc))

        # 5. Chapter-verified bare proper noun (never self-match at i==0!)
        if rest in doc_bare_stems and (start_idx > 0 or prev_pos == "<g>"):
            guess_p = (
                "N" if any(rest.endswith(e) for e in _GREEK_LOAN_ENDINGS) else "NPROP"
            )
            sc = (
                (-0.9 if start_idx > 0 else -1.8)
                + self._intra(prev_pos, guess_p)
                + self._intra(guess_p, "</g>")
            )
            ends.append((((rest, rest, guess_p),), sc))

        # 6. Length-proportional OOV fallback so multi-char proclitics strip cleanly
        if len(rest) >= 3 or start_idx == 0:
            guess_pos = (
                "N"
                if any(rest.endswith(e) for e in _GREEK_LOAN_ENDINGS)
                else "V"
                if prev_pos in {"PPERS", "FUT", "ACAUS", "ACONJ", "APST"}
                else "NPROP"
            )
            single_char_pen = 2.0 if (start_idx == 1 and prev_pos != "<g>") else 0.0
            oov_sc = (
                -2.8
                - 1.6 * len(rest)
                - single_char_pen
                + self._intra(prev_pos, guess_pos)
                + self._intra(guess_pos, "</g>")
            )
            ends.append((((rest, rest, guess_pos),), oov_sc))

        return ends

    def _lattice_candidates(
        self,
        norm_group: str,
        *,
        doc_bare_stems: set[str],
    ) -> list[tuple[tuple[MorphemeSpec, ...], float]]:
        """Intra-group character-lattice DP over Sahidic proclitics, stems, suffixes."""
        L = len(norm_group)
        if L == 0:
            return []

        # beam per char offset i: (pos_tag) -> top-K paths (score, spec_tuple)
        states: list[dict[str, list[tuple[float, tuple[MorphemeSpec, ...]]]]] = [
            defaultdict(list) for _ in range(L + 1)
        ]
        states[0]["<g>"] = [(0.0, ())]
        completed: dict[tuple[MorphemeSpec, ...], float] = {}

        for i in range(L):
            bucket = states[i]
            if not bucket:
                continue
            rest = norm_group[i:]
            for prev_pos, paths in bucket.items():
                # Keep top 4 partial paths per (i, prev_pos)
                paths_sorted = sorted(paths, key=lambda x: -x[0])[:4]

                # Terminal arcs i -> L
                ends = self._terminal_candidates(rest, prev_pos, i, doc_bare_stems)
                for p_score, p_seq in paths_sorted:
                    if len(p_seq) >= 7:
                        continue
                    for end_seq, end_sc in ends:
                        full = p_seq + end_seq
                        tot = p_score + end_sc
                        prior = completed.get(full)
                        if prior is None or tot > prior:
                            completed[full] = tot

                # Non-final proclitic transitions i -> j (0 < i < L - 1)
                max_k = min(self._max_proclitic_len, L - i - 1)
                for k in range(max_k, 0, -1):
                    piece = norm_group[i : i + k]
                    proc_hits = self._tables.proclitic_lex.get(piece)
                    if not proc_hits:
                        continue
                    nxt = norm_group[i + k :]
                    for p_norm, p_lemma, p_pos, p_lp in proc_hits:
                        if p_norm == "ⲙ" and not nxt.startswith(
                            ("ⲡ", "ⲫ", "ⲙ", "ⲃ", "ⲯ")
                        ):
                            continue
                        if piece == "ϩⲙ" and not nxt.startswith(("ⲡ", "ⲫ", "ⲙ")):
                            continue
                        if piece == "ⲩ" and not (
                            i > 0 and norm_group[i - 1] in {"ⲉ", "ⲁ", "ⲏ"}
                        ):
                            continue
                        if piece == "ⲅ" and prev_pos not in {
                            "CONJ",
                            "CREL",
                            "CCIRC",
                            "ACONJ",
                        }:
                            continue
                        if p_pos == "PPERS" and prev_pos in {"ART", "PPOS", "PDEM"}:
                            continue
                        tr = self._intra(prev_pos, p_pos)
                        for p_score, p_seq in paths_sorted:
                            if len(p_seq) >= 6:
                                continue
                            sc = p_score + p_lp + tr
                            states[i + k][p_pos].append(
                                (sc, (*p_seq, (p_norm, p_lemma, p_pos)))
                            )

        return sorted(completed.items(), key=lambda x: (-x[1], x[0]))[:8]

    def _candidates_for_group(
        self,
        raw_group: str,
        *,
        doc_bare_stems: set[str],
    ) -> list[tuple[tuple[MorphemeSpec, ...], float]]:
        _left, core, _right = _strip_edge_punct(raw_group)
        if not core:
            return [(((raw_group, raw_group, "PUNCT"),), 0.0)]
        norm = normalize_coptic(core)
        if not norm:
            return [(((raw_group, raw_group, "PUNCT"),), 0.0)]

        if (
            "\u0305" in raw_group or norm in {"ⲓⲥ", "ⲭⲥ", "ⲡⲛⲁ", "ⲓⲏⲗ", "ⲓⲗⲏⲙ"}
        ) and norm in _NOMINA_SACRA:
            return [(tuple(_NOMINA_SACRA[norm]), 0.0)]

        seen = self._tables.group_lex.get(norm)
        lat = self._lattice_candidates(norm, doc_bare_stems=doc_bare_stems)
        merged: dict[tuple[MorphemeSpec, ...], float] = {}
        if seen:
            for sig, lp in seen:
                merged[sig] = lp + 2.2
        for sig, lp in lat:
            prev = merged.get(sig)
            if prev is None or lp > prev:
                merged[sig] = lp
        return sorted(merged.items(), key=lambda item: -item[1])[:8]

    def _collect_bare_doc_stems(self, all_raw_groups: Iterable[str]) -> set[str]:
        """Identify genuine bare OOV proper nouns verified by p + stem pairs."""
        norms: set[str] = set()
        for raw_grp in all_raw_groups:
            _l, core, _r = _strip_edge_punct(raw_grp)
            norm = normalize_coptic(core)
            if norm:
                norms.add(norm)

        verified: set[str] = set()
        for cand in norms:
            if (
                len(cand) < 3
                or cand in self._tables.group_lex
                or cand in self._tables.stem_lex
            ):
                continue
            # Check if cand also appears prefixed by an observed proclitic
            for full in norms:
                if full == cand or not full.endswith(cand):
                    continue
                prefix = full[: -len(cand)]
                if prefix in self._tables.proclitic_lex:
                    verified.add(cand)
                    break
        return verified

    def _transition_score(self, p0: str, p1: str, p2: str) -> float:
        tri = self._tables.trigram_pos.get((p0, p1, p2))
        if tri is not None:
            return tri
        bi = self._tables.bigram_pos.get((p1, p2))
        if bi is not None:
            return bi - 0.5
        return self._tables.unigram_pos.get(p2, -6.5) - 1.2

    def analyze_groups(
        self,
        raw_groups: Iterable[str],
        *,
        unit_id: str = "coptic.0",
        source_id: str = "copsahhorner",
        doc_bare_stems: set[str] | None = None,
    ) -> list[CopticMorpheme]:
        """Segment and POS-tag a sequence of Coptic bound groups via Viterbi."""
        raw_list = list(raw_groups)
        bare_stems = (
            doc_bare_stems
            if doc_bare_stems is not None
            else self._collect_bare_doc_stems(raw_list)
        )
        groups_with_cands: list[
            tuple[int, str, list[tuple[tuple[MorphemeSpec, ...], float]]]
        ] = []
        for g_idx, raw_grp in enumerate(raw_list):
            left, core, right = _strip_edge_punct(raw_grp)
            if left:
                groups_with_cands.append(
                    (g_idx, left, [(((left, left, "PUNCT"),), 0.0)])
                )
            if core:
                groups_with_cands.append(
                    (
                        g_idx,
                        core,
                        self._candidates_for_group(raw_grp, doc_bare_stems=bare_stems),
                    )
                )
            if right:
                groups_with_cands.append(
                    (g_idx, right, [(((right, right, "PUNCT"),), 0.0)])
                )

        if not groups_with_cands:
            return []

        State = tuple[str, str]
        beams: dict[State, tuple[float, list[tuple[int, str, MorphemeSpec]]]] = {
            ("<s>", "<s>"): (0.0, [])
        }
        for g_idx, orig_slice, cands in groups_with_cands:
            next_beams: dict[
                State, tuple[float, list[tuple[int, str, MorphemeSpec]]]
            ] = {}
            for (p0, p1), (cur_score, hist) in beams.items():
                for sig, seg_score in cands:
                    sc = cur_score + seg_score
                    s0, s1 = p0, p1
                    ctx_sc = 0.0
                    for norm_m, _lemma_m, pos_m in sig:
                        if pos_m == "PUNCT":
                            ctx_sc += self._transition_score(s0, s1, "PUNCT")
                        else:
                            emit = self._tables.morph_pos_logp.get(
                                (norm_m, pos_m), -6.5
                            )
                            ctx_sc += self._transition_score(s0, s1, pos_m) + 0.4 * emit
                        s0, s1 = s1, pos_m
                    sc += 0.45 * (ctx_sc / max(len(sig), 1))
                    state = (s0, s1)
                    prev_best = next_beams.get(state)
                    if prev_best is None or sc > prev_best[0]:
                        extra = [
                            (g_idx, orig_slice if i == 0 else spec[0], spec)
                            for i, spec in enumerate(sig)
                        ]
                        next_beams[state] = (sc, hist + extra)
            ordered = sorted(
                next_beams.items(),
                key=lambda item: (-item[1][0], item[0]),
            )[:12]
            beams = dict(ordered)

        best_path: list[tuple[int, str, MorphemeSpec]] = []
        best_score = -1e18
        for (p0, p1), (sc, hist) in sorted(beams.items()):
            tot = sc + 0.25 * self._transition_score(p0, p1, "</s>")
            if tot > best_score:
                best_score = tot
                best_path = hist

        morphemes: list[CopticMorpheme] = []
        for seq, (g_idx, orig_val, (norm_val, lemma_val, pos_val)) in enumerate(
            best_path
        ):
            morphemes.append(
                CopticMorpheme(
                    id=token_id(source_id, unit_id, seq),
                    unit_id=unit_id,
                    seq=seq,
                    group_seq=g_idx,
                    orig=orig_val,
                    norm=norm_val,
                    lemma=lemma_val,
                    pos=pos_val,
                )
            )
        if self._ud_parser is not None and morphemes:
            self._ud_parser.annotate_in_place(morphemes)
        return morphemes

    def analyze_unit(
        self,
        unit: TextUnit,
        *,
        doc_bare_stems: set[str] | None = None,
    ) -> CopticUnit:
        """Analyze a raw ``TextUnit`` (e.g. CopSahHorner NT) into ``CopticUnit``."""
        raw_groups = [g for g in _SPACE_RE.split(unit.original_text.strip()) if g]
        morphemes = self.analyze_groups(
            raw_groups,
            unit_id=unit.unit_id,
            source_id=unit.source_id,
            doc_bare_stems=doc_bare_stems,
        )
        return CopticUnit(
            unit_id=unit.unit_id,
            source_id=unit.source_id,
            book=unit.book,
            chapter=unit.chapter,
            verse=unit.verse,
            kind=UnitKind(unit.kind),
            original_text=unit.original_text,
            normalized_text=" ".join(m.norm for m in morphemes),
            morphemes=morphemes,
        )

    def analyze_units(self, units: Iterable[TextUnit]) -> list[CopticUnit]:
        """Analyze related units (e.g. a book) sharing bare proper nouns."""
        unit_list = list(units)
        by_book: dict[str, list[TextUnit]] = defaultdict(list)
        for u in unit_list:
            by_book[u.book].append(u)
        results: dict[str, CopticUnit] = {}
        for bk_units in by_book.values():
            all_groups = [
                g
                for u in bk_units
                for g in _SPACE_RE.split(u.original_text.strip())
                if g
            ]
            bare_stems = self._collect_bare_doc_stems(all_groups)
            for u in bk_units:
                results[u.unit_id] = self.analyze_unit(u, doc_bare_stems=bare_stems)
        return [results[u.unit_id] for u in unit_list]
