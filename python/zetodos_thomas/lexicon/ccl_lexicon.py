# SPDX-License-Identifier: Apache-2.0
"""Ingest Comprehensive Coptic Lexicon (CCL v1.2 + DDGLC) Greek loanword tables."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.ingest.normalize import normalize_coptic

_TEI_NS = "{http://www.tei-c.org/ns/1.0}"
_CLEAN_GREEK_RE = re.compile(r"[()\[\]<>·.,;;:!?\"'«»]+")

_DDGLC_POS_MAP: dict[str, str] = {
    "noun": "N-",
    "substantive": "N-",
    "adjective": "A-",
    "verb": "V-",
    "adverb": "D-",
    "conjunction": "C-",
    "preposition": "P-",
    "particle": "C-",
    "pronoun": "RP",
    "numeral": "A-",
}

_GRAMGRP_POS_MAP: dict[str, str] = {
    "Subst.": "N-",
    "Vb.": "V-",
    "trans. Vb.": "V-",
    "intrans. Vb.": "V-",
    "Adj.": "A-",
    "Adv.": "D-",
    "Konj.": "C-",
    "Präp.": "P-",
    "Part.": "C-",
    "Interj.": "I-",
    "Eigenn.": "NPROP",
}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean_greek_lemma(raw: str) -> str:
    txt = _nfc(raw.strip())
    # DDGLC sometimes appends slash/gloss variants like "λόγος / λεγ-" or "(οῦ)"
    txt = txt.split("/")[0].split(",")[0].strip()
    txt = _CLEAN_GREEK_RE.sub("", txt).strip()
    return _nfc(txt)


def _itacism_coptic_key(text: str) -> str:
    """Fold minor Sahidic/Bohairic vowel itacisms for loanword fuzzy match."""
    return (
        text.replace("ⲉⲓ", "ⲓ")
        .replace("ⲏ", "ⲓ")
        .replace("ⲩ", "ⲓ")
        .replace("ⲟⲩ", "ⲱ")
        .replace("ⲑ", "ⲧ")
        .replace("ⲫ", "ⲡ")
        .replace("ⲭ", "ⲕ")
    )


@dataclass(frozen=True)
class LoanwordEntry:
    """One Coptic form -> Greek loanword lemma entry from CCL/DDGLC."""

    coptic_lemma: str
    greek_lemma: str
    greek_pos: str
    gender: str | None = None
    source: str = "L"


class CclLexicon:
    """Index of CCL v1.2 + DDGLC Greek loanwords parsed strictly from pinned TEI."""

    def __init__(
        self,
        *,
        loanwords: dict[str, list[LoanwordEntry]],
        itacism_index: dict[str, list[LoanwordEntry]],
        greek_genders: dict[str, str],
        source_sha256: str = "",
    ) -> None:
        self._loanwords = loanwords
        self._itacism = itacism_index
        self._greek_genders = greek_genders
        self.source_sha256 = source_sha256

    @classmethod
    def from_tei_xml(
        cls, xml_path: Path, *, expected_sha256: str | None = None
    ) -> CclLexicon:
        """Parse ``Comprehensive_Coptic_Lexicon-v1.2-2020.xml`` cleanly (NFC)."""
        if not xml_path.is_file():
            msg = f"missing CCL XML: {xml_path}"
            raise FileNotFoundError(msg)

        digest = hashlib.sha256(xml_path.read_bytes()).hexdigest()
        if expected_sha256 is not None:
            exp_clean = expected_sha256.removeprefix("sha256:").strip().lower()
            if digest != exp_clean:
                msg = (
                    f"CCL XML sha256 mismatch for {xml_path}: "
                    f"got {digest}, expected {exp_clean}"
                )
                raise ValueError(msg)

        loanwords: dict[str, list[LoanwordEntry]] = defaultdict(list)
        itacism_idx: dict[str, list[LoanwordEntry]] = defaultdict(list)
        greek_genders: dict[str, str] = {}

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for entry in root.findall(f".//{_TEI_NS}entry"):
            etym = entry.find(f"{_TEI_NS}etym")
            if etym is None:
                continue

            raw_gr_lemma = ""
            raw_gr_pos = ""
            for ref in etym.findall(f"{_TEI_NS}ref"):
                rtype = ref.attrib.get("type", "")
                if rtype == "greek_lemma::grl_lemma" and ref.text:
                    raw_gr_lemma = ref.text
                elif rtype == "greek_lemma::grl_pos" and ref.text:
                    raw_gr_pos = ref.text.strip().lower()
                elif (
                    ref.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                    == "grc"
                    and ref.text
                ):
                    raw_gr_lemma = raw_gr_lemma or ref.text

            gr_lemma = _clean_greek_lemma(raw_gr_lemma)
            if not gr_lemma or " " in gr_lemma:
                continue

            gen_el = entry.find(f"{_TEI_NS}gramGrp/{_TEI_NS}gen")
            raw_gen = (
                gen_el.text.strip().lower()
                if gen_el is not None and gen_el.text
                else ""
            )
            gender: str | None = (
                "M"
                if raw_gen.startswith("m")
                else "F"
                if raw_gen.startswith("f")
                else "N"
                if raw_gen.startswith("n")
                else None
            )
            if gender and gr_lemma not in greek_genders:
                greek_genders[gr_lemma] = gender

            pos_el = entry.find(f"{_TEI_NS}gramGrp/{_TEI_NS}pos")
            gram_pos = pos_el.text.strip() if pos_el is not None and pos_el.text else ""
            gr_pos = _DDGLC_POS_MAP.get(
                raw_gr_pos, _GRAMGRP_POS_MAP.get(gram_pos, "N-")
            )

            coptic_lemma = ""
            orth_variants: list[str] = []
            for form in entry.findall(f"{_TEI_NS}form"):
                if "deprecated" in form.attrib.get("change", ""):
                    continue
                orth = form.find(f"{_TEI_NS}orth")
                if orth is None or not orth.text:
                    continue
                norm = normalize_coptic(_nfc(orth.text.strip()))
                if not norm or "(" in norm or " " in norm:
                    continue
                # Skip 1-2 char scribal abbreviations so native stems stay unshadowed
                if len(norm) <= 2 and len(gr_lemma) >= 4:
                    continue
                if form.attrib.get("type") == "lemma" and not coptic_lemma:
                    coptic_lemma = norm
                if norm not in orth_variants:
                    orth_variants.append(norm)
            if not coptic_lemma and orth_variants:
                coptic_lemma = orth_variants[0]
            if not coptic_lemma:
                continue

            item = LoanwordEntry(
                coptic_lemma=coptic_lemma,
                greek_lemma=gr_lemma,
                greek_pos=gr_pos,
                gender=gender,
            )
            for surf in orth_variants:
                if item not in loanwords[surf]:
                    loanwords[surf].append(item)
                ita = _itacism_coptic_key(surf)
                if item not in itacism_idx[ita]:
                    itacism_idx[ita].append(item)

        return cls(
            loanwords=dict(loanwords),
            itacism_index=dict(itacism_idx),
            greek_genders=greek_genders,
            source_sha256=digest,
        )

    def resolve(
        self,
        *,
        norm: str,
        lemma: str,
        pos: str | None = None,
        lang: str | None = None,
    ) -> list[LoanwordEntry]:
        """Resolve a Coptic surface/lemma (light-verb compounds & crasis)."""
        candidates: list[str] = []
        for raw in (normalize_coptic(lemma), normalize_coptic(norm)):
            if raw and raw not in candidates:
                candidates.append(raw)

        # Unpack light-verb compounds ⲣ-/ϯ-/ϫⲓ- + Greek stem
        expanded: list[tuple[str, str | None]] = [(c, pos) for c in candidates]
        for c in candidates:
            if c.startswith(("ⲣ", "ϯ")) and len(c) >= 4:
                inner = c[1:]
                expanded.append((inner, "V-"))
            if c.startswith("ϫⲓ") and len(c) >= 5:
                expanded.append((c[2:], "V-"))
            if c.startswith("ⲑ") and len(c) >= 4:
                expanded.append(("ϩ" + c[1:], None))
            if c.startswith("ⲫ") and len(c) >= 4:
                expanded.append(("ϩ" + c[1:], None))
            # Coptic feminine Greek -ⲁ ending or imperative/compound -ⲉ <-> -ω verb stem
            if (pos == "V" or c.startswith(("ⲣ", "ϯ"))) and len(c) >= 4:
                stem = c[1:] if c.startswith(("ⲣ", "ϯ")) else c
                for suffix in ("ⲉ", "ⲉⲓ", "ⲁ", "ⲱ"):
                    if stem.endswith(suffix):
                        expanded.append((stem, "V-"))
                        break

        seen_gl: set[str] = set()
        hits: list[LoanwordEntry] = []

        def _append_list(entries: list[LoanwordEntry], force_pos: str | None) -> None:
            for e in entries:
                adj_pos = (
                    "V-"
                    if (force_pos == "V-" and e.greek_pos in {"V-", "A-", "N-"})
                    else e.greek_pos
                )
                # Prefer verb-derived Greek lemmas when ⲣ-+verb matches both N and V
                item = (
                    LoanwordEntry(
                        coptic_lemma=e.coptic_lemma,
                        greek_lemma=e.greek_lemma,
                        greek_pos=adj_pos,
                        gender=e.gender,
                        source=e.source,
                    )
                    if adj_pos != e.greek_pos
                    else e
                )
                if item.greek_lemma not in seen_gl:
                    seen_gl.add(item.greek_lemma)
                    hits.append(item)

        for key, f_pos in expanded:
            if key in self._loanwords:
                _append_list(self._loanwords[key], f_pos)

        if not hits and lang == "Greek":
            for key, f_pos in expanded:
                ita = _itacism_coptic_key(key)
                if ita in self._itacism:
                    _append_list(self._itacism[ita], f_pos)

        return hits

    def greek_gender(self, greek_lemma: str) -> str | None:
        """Return DDGLC/CCL recorded grammatical gender if known."""
        return self._greek_genders.get(_nfc(greek_lemma))

    def save_json(self, path: Path) -> None:
        """Persist compact loanword lookup (< 1024 KiB)."""
        payload = {
            "source_sha256": self.source_sha256,
            "loanwords": {
                k: [
                    [e.coptic_lemma, e.greek_lemma, e.greek_pos, e.gender]
                    for e in entries[:3]
                ]
                for k, entries in sorted(self._loanwords.items())
            },
            "greek_genders": dict(sorted(self._greek_genders.items())),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> CclLexicon:
        """Load compact persisted CCL/DDGLC loanword table."""
        data = json.loads(path.read_text(encoding="utf-8"))
        loanwords: dict[str, list[LoanwordEntry]] = {}
        itacism_idx: dict[str, list[LoanwordEntry]] = defaultdict(list)
        for k, rows in data["loanwords"].items():
            entries = [
                LoanwordEntry(
                    coptic_lemma=r[0],
                    greek_lemma=r[1],
                    greek_pos=r[2],
                    gender=r[3],
                )
                for r in rows
            ]
            loanwords[k] = entries
            ita = _itacism_coptic_key(k)
            for e in entries:
                if e not in itacism_idx[ita]:
                    itacism_idx[ita].append(e)
        return cls(
            loanwords=loanwords,
            itacism_index=dict(itacism_idx),
            greek_genders=data.get("greek_genders", {}),
            source_sha256=str(data.get("source_sha256", "")),
        )
