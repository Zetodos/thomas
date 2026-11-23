# SPDX-License-Identifier: Apache-2.0
"""Rule- + UD-guided Coptic clause and phrase chunker (uniform on NT & Thomas)."""

from __future__ import annotations

from dataclasses import dataclass, field

from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit

_CLAUSE_SPLIT_CONJ = frozenset(
    {
        "ϫⲉ",
        "ⲁⲩⲱ",
        "ⲁⲗⲗⲁ",
        "ϩⲟⲧⲁⲛ",
        "ⲉϣⲱⲡⲉ",
        "ⲉϣϫⲉ",
        "ϩⲓⲛⲁ",
        "ϫⲉⲕⲁⲥ",
        "ⲉⲡⲉⲓ",
        "ⲟⲩⲇⲉ",
        "ⲟⲩⲧⲉ",
        "ⲏ",
        "ⲙⲏⲡⲱⲥ",
        "ϩⲱⲥ",
    }
)
_POSTPOSITIVE_LEMMAS = frozenset({"ⲇⲉ", "ⲅⲁⲣ", "ⲙⲉⲛ", "ⲟⲩⲛ", "ϭⲉ"})
_VERBAL_PREFIX_POS = frozenset(
    {
        "APST",
        "ANEGPST",
        "ANEGAOR",
        "ANEGJUS",
        "ANEGOPT",
        "AAOR",
        "APREC",
        "ALIM",
        "AJUS",
        "AFUTCONJ",
        "AOPT_PPERS",
        "ACONJ",
        "ACONJ_PPERS",
        "ACOND",
        "ACOND_PPERS",
        "ACAUS",
        "CPRET",
        "FUT",
        "NEG",
        "PPERS",
        "V",
        "VBD",
        "VIMP",
        "VSTAT",
    }
)


@dataclass
class CopticChunk:
    """One syntactic chunk inside a Coptic clause."""

    kind: str  # NP | PP | REL_PART | PRED_V | EXIST | COP | CONJ | ADV | PUNCT | OTHER
    morphemes: list[CopticMorpheme] = field(default_factory=list)
    sub_kind: str = ""  # e.g. gen_marked, obj_marked, dat_marked, subj_marked


@dataclass
class CopticClause:
    """One clause containing ordered chunks plus any Wackernagel postpositive."""

    chunks: list[CopticChunk] = field(default_factory=list)
    postpositive: CopticMorpheme | None = None
    morphemes: list[CopticMorpheme] = field(default_factory=list)


def _split_clauses(morphemes: list[CopticMorpheme]) -> list[list[CopticMorpheme]]:
    """Split morpheme stream into clauses uniformly on Dilley Thomas & Horner NT."""
    if not morphemes:
        return []
    clauses: list[list[CopticMorpheme]] = []
    cur: list[CopticMorpheme] = []
    seen_pred = False
    n = len(morphemes)

    for i, m in enumerate(morphemes):
        if m.pos == "PUNCT":
            if cur:
                cur.append(m)
                clauses.append(cur)
                cur = []
                seen_pred = False
            else:
                clauses.append([m])
            continue

        start_new = False
        if m.pos == "CONJ" and m.lemma in _CLAUSE_SPLIT_CONJ and cur:
            # ⲁⲩⲱ starts new coord clause once a predicate has been seen
            if m.lemma != "ⲁⲩⲱ" or seen_pred:
                start_new = True
        elif seen_pred and m.pos in {
            "APST",
            "ANEGPST",
            "ANEGAOR",
            "ANEGJUS",
            "ANEGOPT",
            "AAOR",
            "APREC",
            "ALIM",
            "AJUS",
            "ACONJ",
            "ACONJ_PPERS",
            "ACOND_PPERS",
            "AFUTCONJ",
            "CPRET",
            "VBD",
        }:
            # Do not break inside a relative clause (CREL/CONJ right before APST)
            # or inside modal `ⲙⲛ ϭⲟⲙ ⲛⲧⲉ ...`
            prev = morphemes[i - 1]
            is_mn_bom_aconj = (
                m.pos in {"ACONJ", "ACONJ_PPERS"}
                and i >= 2
                and morphemes[i - 2].lemma == "ⲙⲛ"
                and prev.lemma == "ϭⲟⲙ"
            )
            if (
                prev.pos not in {"CREL", "CCIRC", "CFOC", "CONJ", "NEG"}
                and not is_mn_bom_aconj
            ):
                start_new = True
        elif (
            seen_pred
            and m.pos == "EXIST"
            and (i == 0 or morphemes[i - 1].pos not in {"CREL", "CONJ"})
        ):
            start_new = True
        elif (
            seen_pred
            and m.lemma == "ⲙⲛ"
            and (i == 0 or morphemes[i - 1].pos not in {"CREL", "CONJ"})
            and i + 2 < n
            and (
                (
                    morphemes[i + 1].lemma == "ϭⲟⲙ"
                    and morphemes[i + 2].pos
                    in {"ACONJ", "ACONJ_PPERS", "PREP", "ACAUS"}
                )
                or (
                    morphemes[i + 1].pos == "N"
                    and morphemes[i + 2].pos in {"V", "VSTAT", "CCIRC", "CREL"}
                )
            )
        ):
            start_new = True
        elif (
            seen_pred
            and m.pos == "PPERS"
            and (
                i == 0
                or morphemes[i - 1].pos
                not in {
                    "APST",
                    "FUT",
                    "NEG",
                    "CREL",
                    "CCIRC",
                    "CFOC",
                    "CPRET",
                    "ACONJ",
                    "ACOND",
                    "ACOND_PPERS",
                    "ACAUS",
                    "APREC",
                    "ALIM",
                    "AJUS",
                    "AAOR",
                    "ANEGPST",
                    "ANEGAOR",
                    "ANEGJUS",
                    "ANEGOPT",
                    "ANY",
                    "CONJ",
                    "VBD",
                    "V",
                    "VSTAT",
                    "VIMP",
                }
            )
        ):
            # Nominal predicate followed by tripartite/bipartite verbal/locative clause
            nxt = morphemes[i + 1] if i + 1 < n else None
            if nxt is not None and nxt.pos in {"FUT", "V", "VSTAT", "PREP", "ADV"}:
                start_new = True
        elif (
            seen_pred
            and i > 0
            and m.head_seq is None
            and m.dep_rel == "root"
            and m.pos in {"V", "VIMP"}
            and morphemes[i - 1].pos
            not in (
                _VERBAL_PREFIX_POS
                | {"CONJ", "CREL", "CCIRC", "CFOC", "NEG", "PREP", "ACAUS"}
            )
            and not (
                morphemes[i - 1].head_seq == m.seq
                and morphemes[i - 1].dep_rel in {"nsubj", "dislocated"}
            )
        ):
            start_new = True
        elif (
            seen_pred
            and i > 0
            and m.pos == "PPERI"
            and (
                (m.head_seq is None and m.dep_rel == "root")
                or (
                    m.dep_rel in {"nsubj", "dislocated"}
                    and m.head_seq is not None
                    and m.head_seq > m.seq
                )
            )
            and morphemes[i - 1].pos
            not in {
                "CONJ",
                "PREP",
                "PTC",
                "CREL",
                "CCIRC",
                "CFOC",
                "EXIST",
                "COP",
                "VBD",
            }
        ):
            start_new = True

        if start_new and cur:
            clauses.append(cur)
            cur = []
            seen_pred = False

        cur.append(m)
        if m.pos in {"V", "VSTAT", "VBD", "VIMP", "EXIST", "COP"}:
            seen_pred = True

    if cur:
        clauses.append(cur)
    return clauses


def _chunk_clause_morphemes(clause_ms: list[CopticMorpheme]) -> CopticClause:
    """Group clause morphemes into structured NP, PP, REL_PART, PRED_V chunks."""
    postpos: CopticMorpheme | None = None
    filtered: list[CopticMorpheme] = []
    for m in clause_ms:
        if (
            postpos is None
            and m.lemma in _POSTPOSITIVE_LEMMAS
            and m.pos in {"PTC", "CONJ", "ADV"}
        ):
            postpos = m
        else:
            filtered.append(m)

    chunks: list[CopticChunk] = []
    i = 0
    n = len(filtered)
    while i < n:
        m = filtered[i]

        # 1. Punctuation
        if m.pos == "PUNCT":
            chunks.append(CopticChunk(kind="PUNCT", morphemes=[m]))
            i += 1
            continue

        # 2. Leading clause conjunction / particle
        if m.pos == "CONJ" or (m.pos == "PTC" and m.lemma != "ⲛϭⲓ"):
            chunks.append(CopticChunk(kind="CONJ", morphemes=[m]))
            i += 1
            continue

        # 3. Articular relative participle: ART? + CREL + verb chain
        if (
            m.pos == "ART" and i + 1 < n and filtered[i + 1].pos == "CREL"
        ) or m.pos == "CREL":
            j = i + 1 if m.pos == "CREL" else i + 2
            while j < n and filtered[j].pos in _VERBAL_PREFIX_POS:
                j += 1
                if filtered[j - 1].pos in {"V", "VSTAT"}:
                    break
            chunks.append(CopticChunk(kind="REL_PART", morphemes=filtered[i:j]))
            i = j
            continue

        # 4. Existential ⲟⲩⲛ / ⲙⲛ (+ optional pronominal ⲛⲧⲁ=)
        if m.pos == "EXIST":
            j = i + 1
            if (
                j < n
                and filtered[j].pos == "PREP"
                and filtered[j].lemma in {"ⲛⲧⲉ", "ⲛ"}
            ):
                if j + 1 < n and filtered[j + 1].pos in {"PPERO", "N", "ART"}:
                    j += 1
            chunks.append(CopticChunk(kind="EXIST", morphemes=filtered[i:j]))
            i = j
            continue

        # 5. Copula ⲡⲉ / ⲧⲉ / ⲛⲉ
        if m.pos == "COP":
            chunks.append(CopticChunk(kind="COP", morphemes=[m]))
            i += 1
            continue

        # 6. Prepositional phrase (or PTC ⲛϭⲓ subject marker!) + NP / PPERO
        if m.pos == "PREP" or (m.pos == "PTC" and m.lemma == "ⲛϭⲓ"):
            marker = m
            sub_k = "subj_marked" if marker.lemma == "ⲛϭⲓ" else "pp"
            j = i + 1
            # Infinitive complement ⲉ + (ⲧⲣⲉ)? + V
            if marker.lemma == "ⲉ" and j < n and filtered[j].pos in {"ACAUS", "V"}:
                chunks.append(CopticChunk(kind="CONJ", morphemes=[marker]))
                i = j
                continue
            # Pronoun object: PREP + PPERO
            if j < n and filtered[j].pos in {"PPERO", "PPERI", "PDEM", "PINT"}:
                j += 1
                chunks.append(
                    CopticChunk(kind="PP", morphemes=filtered[i:j], sub_kind=sub_k)
                )
                i = j
                continue
            # NP after marker: ART|PPOS|PDEM|NUM? + N|NPROP|PINT|NUM
            if j < n and filtered[j].pos in {"ART", "PPOS", "PDEM", "NUM"}:
                j += 1
            if j < n and filtered[j].pos in {"N", "NPROP", "NUM", "PINT", "PDEM"}:
                j += 1
                # Keep atomic NP for clean case marking
                chunks.append(
                    CopticChunk(kind="PP", morphemes=filtered[i:j], sub_kind=sub_k)
                )
                i = j
                continue
            chunks.append(CopticChunk(kind="PP", morphemes=[marker], sub_kind=sub_k))
            i += 1
            continue

        # 7. Verbal predication chain + bound PPERO object
        if m.pos in _VERBAL_PREFIX_POS and not (m.pos == "NEG" and m.lemma == "ⲁⲛ"):
            # Look ahead to confirm a verb head exists before an ART/PREP/NOUN breaks it
            j = i
            has_verb = False
            while j < n:
                pj = filtered[j].pos
                if pj in {"V", "VBD", "VIMP", "VSTAT"}:
                    has_verb = True
                    j += 1
                    # Bound pronominal object directly attached to V (PPERO)
                    if j < n and filtered[j].pos == "PPERO":
                        j += 1
                    break
                if pj in _VERBAL_PREFIX_POS:
                    j += 1
                    continue
                break
            if has_verb:
                chunks.append(CopticChunk(kind="PRED_V", morphemes=filtered[i:j]))
                i = j
                continue

        # 8. Standalone NP: [ART|PPOS|PDEM|NUM] + N|NPROP|NUM?
        if m.pos in {"ART", "PPOS", "PDEM", "NUM", "N", "NPROP"}:
            j = i + 1
            if (
                m.pos in {"ART", "PPOS", "PDEM", "NUM"}
                and j < n
                and filtered[j].pos
                in {
                    "N",
                    "NPROP",
                    "NUM",
                }
            ):
                j += 1
            chunks.append(CopticChunk(kind="NP", morphemes=filtered[i:j]))
            i = j
            continue

        # 9. Negation trailing ⲁⲛ, independent/interrogative pronouns, adverbs, PPERO
        chunks.append(
            CopticChunk(
                kind="ADV" if m.pos in {"ADV", "IMOD", "NEG"} else "OTHER",
                morphemes=[m],
            )
        )
        i += 1

    return CopticClause(chunks=chunks, postpositive=postpos, morphemes=clause_ms)


def chunk_unit(unit: CopticUnit) -> list[CopticClause]:
    """Chunk a ``CopticUnit`` into grammatical clauses and NP/PP/VP spans."""
    raw_clauses = _split_clauses(unit.morphemes)
    return [_chunk_clause_morphemes(c) for c in raw_clauses]
