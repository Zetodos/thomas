# SPDX-License-Identifier: Apache-2.0
"""Bridge native Egyptian Coptic CCL glosses to Strong's Greek lemmas via English.

Strictly uses only pinned, open-licensed lexicons:
- ``Comprehensive_Coptic_Lexicon-v1.2-2020.xml`` (CC-BY-SA-4.0)
- ``morphgnt/strongs-dictionary-xml/strongsgreek.xml`` (CC0-1.0)

Produces high-precision ``native_lex_pairs`` (for rare native Sahidic vocabulary
reinforcement in ``model-v3``) and ``grc_eng_pairs`` (to anchor the shared English
pivot space alongside Sahidic Apophthegmata ``cop -> eng`` pairs).
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.lexicon.ccl_lexicon import _GRAMGRP_POS_MAP, _TEI_NS

_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_WORD_RE = re.compile(r"[a-z]{3,}")
_CLEAN_GREEK_RE = re.compile(r"[()\[\]<>·.,;;:!?\"'«»]+")
# Lexicographic meta-glosses expressing uncertainty, editorial status, or generic
# taxonomic placeholders ("unknown", "meaning uncertain", "a certain ...", "month")
# that would otherwise collide with Strong's lemmas like ἄγνωστος / δεῖνα / τετράμηνον.
_META_GLOSS_RE = re.compile(
    r"\b("
    r"unknown|uncertain|unidentified|unclear|obscure|doubtful|unexplained|"
    r"fragmentary|corrupt|untranslated|dubious|"
    r"meaning\s+(?:unknown|uncertain|obscure|doubtful|unclear)|"
    r"a\s+certain|some\s+kind\s+of|kind\s+of|sort\s+of|type\s+of|species\s+of|"
    r"name\s+of|designation\s+of|epithet\s+of|title\s+of|"
    r"measure\s+of|weight\s+of|coin\s+of|vessel\s+for|"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth)\s+month|month\s+of"
    r")\b",
    re.IGNORECASE,
)

_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "onto",
        "upon",
        "over",
        "under",
        "that",
        "this",
        "these",
        "those",
        "one",
        "ones",
        "who",
        "whom",
        "which",
        "what",
        "when",
        "where",
        "while",
        "also",
        "only",
        "very",
        "more",
        "most",
        "much",
        "many",
        "some",
        "any",
        "all",
        "not",
        "nor",
        "but",
        "out",
        "off",
        "about",
        "after",
        "before",
        "between",
        "through",
        "during",
        "without",
        "within",
        "being",
        "been",
        "have",
        "has",
        "had",
        "make",
        "made",
        "give",
        "take",
        "come",
        "used",
        "properly",
        "figuratively",
        "literally",
        "especially",
        "generally",
        "usually",
        "often",
        "sometimes",
        "i.e",
        "etc",
        "something",
        "someone",
        "person",
        "man",
        "men",
        "thing",
        "things",
        "state",
        "act",
        "action",
        "manner",
        "kind",
        "sort",
        "form",
        "part",
        "place",
        "time",
        "way",
        "sense",
        "word",
        "name",
        "greek",
        "hebrew",
        "coptic",
        "egyptian",
        "compare",
        "contraction",
        "derivative",
        "diminutive",
        "feminine",
        "masculine",
        "neuter",
        "plural",
        "singular",
        "passive",
        "active",
        "middle",
        "reflexive",
        "intransitive",
        "transitive",
        "causative",
        "perhaps",
        "probably",
        "hence",
        "thus",
        "like",
        "same",
        "other",
        "another",
        "such",
        "self",
        "own",
        "him",
        "her",
        "his",
        "its",
        "their",
        "them",
        "they",
        "you",
        "your",
        "our",
        "unknown",
        "uncertain",
        "unidentified",
        "unclear",
        "certain",
        "meaning",
        "obscure",
        "doubtful",
        "unexplained",
        "dubious",
        "fragmentary",
        "corrupt",
        "variant",
        "designation",
        "epithet",
        "species",
        "month",
        "months",
        "measure",
        "vessel",
        "bear",
        "bears",
        "bearer",
        "bearing",
    }
)


def _is_meta_gloss(text: str) -> bool:
    """Return True if an English dictionary gloss is a lexicographic placeholder."""
    return bool(_META_GLOSS_RE.search(text))


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _extract_en_content_words(text: str) -> set[str]:
    """Extract normalized lowercase English content stems from a dictionary gloss."""
    out: set[str] = set()
    # Strip parenthetical notes like "(lit. ...)" before tokenizing
    cleaned = re.sub(r"\([^)]*\)", " ", text.lower())
    for tok in _WORD_RE.findall(cleaned):
        if tok in _EN_STOPWORDS:
            continue
        # Light inflectional folding for English dictionary glosses
        if len(tok) > 5 and tok.endswith("ing"):
            out.add(tok[:-3])
        if len(tok) > 5 and tok.endswith("ed"):
            out.add(tok[:-2])
        if len(tok) > 4 and tok.endswith("es") and not tok.endswith(("ses", "ies")):
            out.add(tok[:-2])
        elif len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            out.add(tok[:-1])
        out.add(tok)
    return out


def _infer_strongs_pos(greek_lemma: str, def_text: str) -> str:
    """Heuristically infer coarse Greek POS (N-/V-/A-/D-) for a Strong's entry."""
    low_def = def_text.strip().lower()
    if low_def.startswith("to ") or " to " in low_def[:30]:
        return "V-"
    if greek_lemma.endswith(("ω", "ῶ", "ομαι", "μαι", "νυμι", "μι")):
        return "V-"
    if greek_lemma.endswith("ως") or low_def.startswith(("adverb", "in a ")):
        return "D-"
    if low_def.startswith(("a ", "an ", "the ")):
        return "N-"
    if greek_lemma.endswith(("ος", "ός", "ής", "ης", "ων", "ών", "υς", "ύς")):
        return (
            "A-" if any(k in low_def[:24] for k in ("full of", "pertaining")) else "N-"
        )
    return "N-"


@dataclass(frozen=True)
class NativeGlossMatch:
    """One bridged native Sahidic Coptic -> Koine Greek lexical candidate."""

    coptic_lemma: str
    coptic_pos: str
    greek_lemma: str
    greek_pos: str
    score: float
    shared_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlossBridgeResult:
    """Output of CCL native gloss x Strong's Greek dictionary bridging."""

    native_pairs: list[NativeGlossMatch]
    grc_eng_pairs: list[tuple[str, str]]
    native_coptic_count: int
    strongs_entry_count: int

    def to_hint_map(
        self,
        *,
        min_score: float = 0.40,
        max_per_lemma: int = 1,
    ) -> dict[str, list[tuple[str, str, float]]]:
        """Return ``coptic_lemma -> [(greek_lemma, greek_pos, score), ...]``."""
        out: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for m in self.native_pairs:
            if m.score < min_score:
                continue
            bucket = out[m.coptic_lemma]
            if len(bucket) < max_per_lemma:
                bucket.append((m.greek_lemma, m.greek_pos, round(m.score, 4)))
        return dict(out)

    def save_json(self, path: Path) -> None:
        payload = {
            "native_coptic_count": self.native_coptic_count,
            "strongs_entry_count": self.strongs_entry_count,
            "hints": self.to_hint_map(min_score=0.35, max_per_lemma=2),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def _parse_ccl_native_entries(
    ccl_xml: Path,
) -> list[tuple[str, str, set[str]]]:
    """Extract Sahidic native Egyptian ``(coptic_lemma, coarse_pos, en_words)``."""
    entries: list[tuple[str, str, set[str]]] = []
    seen: set[tuple[str, str]] = set()
    for _ev, elem in ET.iterparse(ccl_xml, events=("end",)):
        if elem.tag != f"{_TEI_NS}entry":
            continue
        # Skip Greek loanwords (already handled cleanly by CclLexicon!)
        if elem.find(f".//{_TEI_NS}etym") is not None:
            elem.clear()
            continue

        # Require Sahidic dialect ("S") lemma form
        sah_lemmas: list[str] = []
        for form in elem.findall(f"{_TEI_NS}form"):
            if form.attrib.get("type") != "lemma":
                continue
            geos = {
                (u.text or "").strip()
                for u in form.findall(f"{_TEI_NS}usg")
                if u.attrib.get("type") == "geo"
            }
            if geos and "S" not in geos:
                continue
            for orth in form.findall(f"{_TEI_NS}orth"):
                raw_o = (orth.text or "").strip()
                if (
                    not raw_o
                    or raw_o.startswith("*")
                    or "-" in raw_o
                    or " " in raw_o
                    or "⸗" in raw_o
                ):
                    continue
                norm_o = normalize_coptic(raw_o).replace("\u2cff", "")
                if len(norm_o) >= 2 and not norm_o.startswith("*"):
                    sah_lemmas.append(norm_o)
        if not sah_lemmas:
            elem.clear()
            continue

        # Coarse POS from <gramGrp><pos>
        c_pos = "N"
        pos_el = elem.find(f".//{_TEI_NS}gramGrp/{_TEI_NS}pos")
        if pos_el is not None and pos_el.text:
            mapped = _GRAMGRP_POS_MAP.get(pos_el.text.strip(), "")
            if mapped == "V-":
                c_pos = "V"
            elif mapped == "D-":
                c_pos = "ADV"
            elif mapped == "A-":
                c_pos = "A"
            elif mapped in {"N-", "NPROP"}:
                c_pos = "N"
            elif mapped:
                # Skip closed-class function words (PREP, CONJ, etc.)
                elem.clear()
                continue

        en_words: set[str] = set()
        for quote in elem.findall(f".//{_TEI_NS}quote"):
            if quote.attrib.get(_XML_LANG) == "en" and quote.text:
                if _is_meta_gloss(quote.text):
                    continue
                en_words.update(_extract_en_content_words(quote.text))
        if en_words:
            lem = sah_lemmas[0]
            key = (lem, c_pos)
            if key not in seen:
                seen.add(key)
                entries.append((lem, c_pos, en_words))
        elem.clear()
    return entries


def _parse_strongs_entries(
    strongs_xml: Path,
) -> tuple[list[tuple[str, str, set[str]]], list[tuple[str, str]]]:
    """Parse Strong's Greek XML into ``(greek_lemma, pos, en_words)`` + pivot pairs."""
    tree = ET.parse(strongs_xml)
    root = tree.getroot()
    entries: list[tuple[str, str, set[str]]] = []
    grc_eng_pairs: list[tuple[str, str]] = []
    seen_lemmas: set[str] = set()

    for entry in root.iter("entry"):
        grk_el = entry.find("greek")
        if grk_el is None:
            continue
        raw_g = (grk_el.attrib.get("unicode") or "").strip()
        raw_g = _CLEAN_GREEK_RE.sub("", _nfc(raw_g)).strip()
        if not raw_g or " " in raw_g or len(raw_g) < 2:
            continue
        if raw_g in seen_lemmas:
            continue

        def_parts: list[str] = []
        sdef = entry.find("strongs_def")
        if sdef is not None:
            def_parts.append("".join(sdef.itertext()).strip())
        kdef = entry.find("kjv_def")
        if kdef is not None:
            def_parts.append("".join(kdef.itertext()).strip())
        full_def = " ".join(p for p in def_parts if p).strip()
        if not full_def:
            continue
        # Clean concise English gloss for pivot training
        short_en = re.sub(
            r"\s+", " ", ("".join(sdef.itertext()) if sdef is not None else full_def)
        ).strip()
        short_en = re.sub(r"^[:;\-\s]+|[:;\-\s]+$", "", short_en)
        en_words = _extract_en_content_words(full_def)
        if not en_words:
            continue

        g_pos = _infer_strongs_pos(raw_g, full_def)
        seen_lemmas.add(raw_g)
        entries.append((raw_g, g_pos, en_words))
        if 3 <= len(short_en) <= 120:
            grc_eng_pairs.append((raw_g, short_en))
    return entries, grc_eng_pairs


def _pos_compatible(c_pos: str, g_pos: str) -> bool:
    if c_pos == "V":
        return g_pos == "V-"
    if c_pos == "ADV":
        return g_pos in {"D-", "A-"}
    if c_pos == "A":
        return g_pos in {"A-", "N-"}
    return g_pos in {"N-", "A-"}


def build_ccl_strongs_bridge(
    *,
    ccl_xml: Path,
    strongs_xml: Path,
    min_score: float = 0.38,
    max_matches_per_coptic: int = 2,
    max_coptic_per_greek: int = 4,
) -> GlossBridgeResult:
    """Bridge native Sahidic CCL lemmas to Strong's Greek via IDF gloss overlap."""
    c_entries = _parse_ccl_native_entries(ccl_xml)
    g_entries, grc_eng_pairs = _parse_strongs_entries(strongs_xml)

    # Compute document frequency across both lexicons so rare specific terms dominate
    df: Counter[str] = Counter()
    for _, _, ws in c_entries:
        df.update(ws)
    for _, _, ws in g_entries:
        df.update(ws)
    total_docs = max(len(c_entries) + len(g_entries), 1)
    idf: dict[str, float] = {
        w: math.log((total_docs + 1.0) / (cnt + 1.0)) + 1.0 for w, cnt in df.items()
    }

    # Inverted index from English content stem -> Strong's entry indices
    g_by_term: dict[str, list[int]] = defaultdict(list)
    g_norms: list[float] = []
    for idx, (_gl, _gp, ws) in enumerate(g_entries):
        g_norms.append(math.sqrt(sum(idf[w] * idf[w] for w in ws)) or 1.0)
        for w in ws:
            # Skip ultra-generic words appearing in > 200 Strong's entries
            if df[w] <= 250:
                g_by_term[w].append(idx)

    # First pass: best Greek candidates for each Coptic entry (IDF cosine similarity)
    c_to_best: dict[int, list[tuple[int, float, tuple[str, ...]]]] = {}
    g_top_c: dict[int, float] = defaultdict(float)

    for c_idx, (_cl, c_pos, c_ws) in enumerate(c_entries):
        c_norm = math.sqrt(sum(idf[w] * idf[w] for w in c_ws)) or 1.0
        cand_scores: dict[int, float] = defaultdict(float)
        cand_terms: dict[int, list[str]] = defaultdict(list)
        for w in c_ws:
            wt = idf[w] * idf[w]
            for g_idx in g_by_term.get(w, ()):
                cand_scores[g_idx] += wt
                cand_terms[g_idx].append(w)

        scored: list[tuple[int, float, tuple[str, ...]]] = []
        for g_idx, dot in cand_scores.items():
            _, g_pos, _ = g_entries[g_idx]
            if not _pos_compatible(c_pos, g_pos):
                continue
            cos = dot / (c_norm * g_norms[g_idx])
            if cos >= min_score:
                terms_sorted = tuple(sorted(set(cand_terms[g_idx])))
                scored.append((g_idx, cos, terms_sorted))
                if cos > g_top_c[g_idx]:
                    g_top_c[g_idx] = cos
        if scored:
            scored.sort(key=lambda x: (-x[1], g_entries[x[0]][0]))
            c_to_best[c_idx] = scored[: max_matches_per_coptic + 2]

    # Second pass: reciprocal affinity check (>= 85% of Greek lemma's best match)
    matches: list[NativeGlossMatch] = []
    for c_idx, cands in sorted(c_to_best.items()):
        c_lem, c_pos, _ = c_entries[c_idx]
        kept = 0
        for g_idx, cos, terms in cands:
            if cos < 0.85 * g_top_c[g_idx]:
                continue
            g_lem, g_pos, _ = g_entries[g_idx]
            matches.append(
                NativeGlossMatch(
                    coptic_lemma=c_lem,
                    coptic_pos=c_pos,
                    greek_lemma=g_lem,
                    greek_pos=g_pos,
                    score=round(cos, 4),
                    shared_terms=terms,
                )
            )
            kept += 1
            if kept >= max_matches_per_coptic:
                break

    # Third pass: cap many-to-one fan-in per Greek target so generic/noisy Greek
    # lemmas cannot accumulate dozens of unrelated Sahidic lemmas.
    if max_coptic_per_greek > 0:
        by_greek: dict[str, list[NativeGlossMatch]] = defaultdict(list)
        for m in matches:
            by_greek[m.greek_lemma].append(m)
        filtered: list[NativeGlossMatch] = []
        for g_lem in sorted(by_greek):
            bucket = by_greek[g_lem]
            bucket.sort(key=lambda m: (-m.score, m.coptic_lemma, m.coptic_pos))
            filtered.extend(bucket[:max_coptic_per_greek])
        filtered.sort(key=lambda m: (m.coptic_lemma, -m.score, m.greek_lemma))
        matches = filtered

    return GlossBridgeResult(
        native_pairs=matches,
        grc_eng_pairs=grc_eng_pairs,
        native_coptic_count=len(c_entries),
        strongs_entry_count=len(g_entries),
    )
