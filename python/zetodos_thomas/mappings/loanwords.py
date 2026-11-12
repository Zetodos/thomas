# SPDX-License-Identifier: Apache-2.0
"""Minimal CCL/DDGLC verified loanword subset (JAK-38 / JAK-19).

Entries are independently verified (manually spot-checked against CCL TEI
v1.2) and must never be mixed into statistical training observations.
License: CC BY-SA 4.0 on adapted lexical exports.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zetodos_thomas.schemas.mappings import (
    EvidenceClass,
    SpotCheckRecord,
    VerifiedLoanword,
    VerifiedLoanwordTable,
)

# Curated from Comprehensive Coptic Lexicon v1.2 (Burns/Feder/John/Kupreyev;
# DDGLC Greek loanwords). Spot-checked against TEI entry fields.
# DOI: 10.17169/refubium-27566; KELLIA xml pin below.
_CCL_ATTRIBUTION = (
    "Comprehensive Coptic Lexicon v1.2 — Burns, Dylan Michael; Feder, Frank; "
    "John, Katrin; Kupreyev, Maxim; BBAW Thesaurus Linguae Aegyptiae; "
    "DDGLC (Freie Universität Berlin). CC BY-SA 4.0. "
    "DOI https://doi.org/10.17169/refubium-27566"
)

_SPOT = SpotCheckRecord(
    checked_by="cursor-agent-jak-38",
    checked_at_utc="2026-11-12T00:00:00Z",
    method=(
        "Manual inspection of CCL TEI v1.2 entry: lemma orth, "
        "etym/ref greek_lemma::grl_ID + grl_lemma, DDGLC sense bibl."
    ),
    result="pass",
    notes="Lexical correspondence only; inflection not entailed.",
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _undiacriticize_greek(text: str) -> str:
    """Strip combining marks for Vaticanus-style undiacriticized matching."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def default_verified_loanwords_path(repo_root: Path) -> Path:
    """Committed verified loanword subset path."""
    return (
        repo_root
        / "data"
        / "sources"
        / "mappings"
        / "verified-loanwords-ccl-ddglc-v1.json"
    )


def _entry(
    *,
    loanword_id: str,
    coptic_lemma: str,
    coptic_forms: list[str],
    greek_lemma: str,
    greek_pos: str,
    greek_meaning: str,
    tla_entry_id: str,
    ddglc_greek_lemma_id: str,
    ccl_form_ids: list[str],
    pilot_priority_notes: str = "",
    illustrative_nt_span_ids: list[str] | None = None,
    notes: str = "",
) -> VerifiedLoanword:
    greek = _nfc(greek_lemma)
    return VerifiedLoanword(
        evidence_class=EvidenceClass.INDEPENDENTLY_VERIFIED,
        loanword_id=loanword_id,
        coptic_lemma=coptic_lemma,
        coptic_forms=coptic_forms,
        greek_lemma=greek,
        greek_lemma_normalized=_undiacriticize_greek(greek),
        greek_pos=greek_pos,
        greek_meaning=greek_meaning,
        tla_entry_id=tla_entry_id,
        ddglc_greek_lemma_id=ddglc_greek_lemma_id,
        ccl_form_ids=ccl_form_ids,
        license="CC-BY-SA-4.0",
        attribution=_CCL_ATTRIBUTION,
        source_manifest_id="ccl-ddglc-tei",
        spot_check=_SPOT,
        illustrative_nt_span_ids=list(illustrative_nt_span_ids or []),
        pilot_priority_notes=pilot_priority_notes,
        notes=notes,
    )


def verified_loanword_entries() -> list[VerifiedLoanword]:
    """Return the curated, spot-checked Phase-2 loanword subset."""
    return [
        _entry(
            loanword_id="verified-makarios",
            coptic_lemma="ⲙⲁⲕⲁⲣⲓⲟⲥ",
            coptic_forms=["ⲙⲁⲕⲁⲣⲓⲟⲥ", "ⲙⲁⲕⲁⲣⲓⲟⲛ"],
            greek_lemma="μακάριος",
            greek_pos="adjective",
            greek_meaning="blessed, happy",
            tla_entry_id="C9689",
            ddglc_greek_lemma_id="1336",
            ccl_form_ids=["CF24876"],
            pilot_priority_notes=(
                "Motivated by Thomas 7 Coptic ⲙⲁⲕⲁⲣⲓⲟⲥ (L clue). "
                "Do not consult P.Oxy. 654 Greek when verifying this row."
            ),
            notes="DDGLC sense bibl present on CCL TEI entry.",
        ),
        _entry(
            loanword_id="verified-euaggelion",
            coptic_lemma="ⲉⲩⲁⲅⲅⲉⲗⲓⲟⲛ",
            coptic_forms=["ⲉⲩⲁⲅⲅⲉⲗⲓⲟⲛ", "ⲉⲩⲁⲅ"],
            greek_lemma="εὐαγγέλιον",
            greek_pos="noun",
            greek_meaning="good news, gospel",
            tla_entry_id="C9059",
            ddglc_greek_lemma_id="809",
            ccl_form_ids=["CF22411"],
            illustrative_nt_span_ids=[
                "v1:copsahhorner/mark.1.1/s/euaggelion",
                "v1:ntvmr-vaticanus-b/mark.1.1/s/euaggeliou",
            ],
            notes="NT gold sample also tags this as loanword (illustrative only).",
        ),
        _entry(
            loanword_id="verified-mathetes",
            coptic_lemma="ⲙⲁⲑⲏⲧⲏⲥ",
            coptic_forms=["ⲙⲁⲑⲏⲧⲏⲥ", "ⲙⲁⲑⲏⲧ"],
            greek_lemma="μαθητής",
            greek_pos="noun",
            greek_meaning="learner, pupil, disciple",
            tla_entry_id="C9683",
            ddglc_greek_lemma_id="1334",
            ccl_form_ids=["CF24841"],
            illustrative_nt_span_ids=[
                "v1:copsahhorner/luke.6.20/s/mathetes",
                "v1:ntvmr-vaticanus-b/luke.6.20/s/mathetas",
            ],
        ),
        _entry(
            loanword_id="verified-aitei",
            coptic_lemma="ⲁⲓⲧⲉⲓ",
            coptic_forms=["ⲁⲓⲧⲉⲓ", "ⲁⲓⲧⲉ"],
            greek_lemma="αἰτέω",
            greek_pos="verb",
            greek_meaning="to ask (s.o. for s.th.)",
            tla_entry_id="C8126",
            ddglc_greek_lemma_id="54",
            ccl_form_ids=["CF18503"],
            illustrative_nt_span_ids=[
                "v1:copsahhorner/matthew.7.7/s/aitei",
                "v1:ntvmr-vaticanus-b/matthew.7.7/s/aiteite",
            ],
        ),
        _entry(
            loanword_id="verified-parabole",
            coptic_lemma="ⲡⲁⲣⲁⲃⲟⲗⲏ",
            coptic_forms=["ⲡⲁⲣⲁⲃⲟⲗⲏ"],
            greek_lemma="παραβολή",
            greek_pos="noun",
            greek_meaning="juxtaposition, comparison; parable",
            tla_entry_id="C10032",
            ddglc_greek_lemma_id="2558",
            ccl_form_ids=["CF26233"],
            illustrative_nt_span_ids=[
                "v1:copsahhorner/matthew.13.31/s/parabole",
                "v1:ntvmr-vaticanus-b/matthew.13.31/s/parabolen",
            ],
        ),
        _entry(
            loanword_id="verified-de",
            coptic_lemma="ⲇⲉ",
            coptic_forms=["ⲇⲉ"],
            greek_lemma="δέ",
            greek_pos="particle",
            greek_meaning="but, and, so, then",
            tla_entry_id="C8719",
            ddglc_greek_lemma_id="458",
            ccl_form_ids=["CF20757"],
        ),
        _entry(
            loanword_id="verified-gar",
            coptic_lemma="ⲅⲁⲣ",
            coptic_forms=["ⲅⲁⲣ"],
            greek_lemma="γάρ",
            greek_pos="particle",
            greek_meaning="for, then, because",
            tla_entry_id="C8657",
            ddglc_greek_lemma_id="415",
            ccl_form_ids=["CF20474"],
        ),
        _entry(
            loanword_id="verified-alla",
            coptic_lemma="ⲁⲗⲗⲁ",
            coptic_forms=["ⲁⲗⲗⲁ"],
            greek_lemma="ἀλλά",
            greek_pos="conjunction",
            greek_meaning="but, yet, rather",
            tla_entry_id="C8179",
            ddglc_greek_lemma_id="95",
            ccl_form_ids=["CF18776"],
        ),
        _entry(
            loanword_id="verified-oude",
            coptic_lemma="ⲟⲩⲇⲉ",
            coptic_forms=["ⲟⲩⲇⲉ"],
            greek_lemma="οὐδέ",
            greek_pos="conjunction",
            greek_meaning="nor, and not, not even",
            tla_entry_id="C9959",
            ddglc_greek_lemma_id="1592",
            ccl_form_ids=["CF25965"],
        ),
        _entry(
            loanword_id="verified-psyche",
            coptic_lemma="ⲯⲩⲭⲏ",
            coptic_forms=["ⲯⲩⲭⲏ"],
            greek_lemma="ψυχή",
            greek_pos="noun",
            greek_meaning="life, spirit, soul, self",
            tla_entry_id="C11031",
            ddglc_greek_lemma_id="2433",
            ccl_form_ids=["CF30301"],
        ),
        _entry(
            loanword_id="verified-kosmos",
            coptic_lemma="ⲕⲟⲥⲙⲟⲥ",
            coptic_forms=["ⲕⲟⲥⲙⲟⲥ"],
            greek_lemma="κόσμος",
            greek_pos="noun",
            greek_meaning="order, ornament, world, universe",
            tla_entry_id="C9517",
            ddglc_greek_lemma_id="1192",
            ccl_form_ids=["CF24208"],
        ),
        _entry(
            loanword_id="verified-pneuma",
            coptic_lemma="ⲡⲛⲉⲩⲙⲁ",
            coptic_forms=["ⲡⲛⲉⲩⲙⲁ"],
            greek_lemma="πνεῦμα",
            greek_pos="noun",
            greek_meaning="blast, wind, breath, spirit",
            tla_entry_id="C10220",
            ddglc_greek_lemma_id="1775",
            ccl_form_ids=["CF27013"],
        ),
        _entry(
            loanword_id="verified-sarx",
            coptic_lemma="ⲥⲁⲣⲝ",
            coptic_forms=["ⲥⲁⲣⲝ"],
            greek_lemma="σάρξ",
            greek_pos="noun",
            greek_meaning="flesh",
            tla_entry_id="C10421",
            ddglc_greek_lemma_id="1937",
            ccl_form_ids=["CF27869"],
        ),
        _entry(
            loanword_id="verified-angelos",
            coptic_lemma="ⲁⲅⲅⲉⲗⲟⲥ",
            coptic_forms=["ⲁⲅⲅⲉⲗⲟⲥ", "ⲁⲅⲅⲉⲗⲟ"],
            greek_lemma="ἄγγελος",
            greek_pos="noun",
            greek_meaning="messenger, angel",
            tla_entry_id="C8062",
            ddglc_greek_lemma_id="11",
            ccl_form_ids=["CF18246"],
        ),
    ]


def verified_loanword_table(
    *,
    provenance: dict[str, Any] | None = None,
) -> VerifiedLoanwordTable:
    """Assemble the verified loanword export."""
    entries = verified_loanword_entries()
    return VerifiedLoanwordTable(
        id="verified-loanwords-ccl-ddglc-v1",
        title=(
            "Minimal independently verified Greek-loanword correspondences "
            "(CCL/DDGLC; JAK-38)"
        ),
        owned_by="JAK-38",
        schema_version="1",
        evidence_class=EvidenceClass.INDEPENDENTLY_VERIFIED,
        license="CC-BY-SA-4.0",
        entries=entries,
        provenance={
            "owned_by": "JAK-38",
            "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "policy": "JAK-19",
            "source_manifest_id": "ccl-ddglc-tei",
            "ccl_release": "Comprehensive Coptic Lexicon v1.2 (2020)",
            "ccl_tei_filename": "Comprehensive_Coptic_Lexicon-v1.2-2020.xml",
            "ccl_tei_source_url": (
                "https://raw.githubusercontent.com/KELLIA/dictionary/master/"
                "xml/Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
            ),
            "ccl_tei_sha256": (
                "ace84fce11b76386a6adb9888b2cb947056954fc9fe9c04aa0676b1f595fd43d"
            ),
            "doi": "10.17169/refubium-27566",
            "license": "CC-BY-SA-4.0",
            "attribution": _CCL_ATTRIBUTION,
            "dioskoros": (
                "Not used; JAK-19 forbids waiting on / scraping DIOSKOROS."
            ),
            **dict(provenance or {}),
        },
        summary={
            "entry_count": len(entries),
            "evidence_class": EvidenceClass.INDEPENDENTLY_VERIFIED.value,
            "license": "CC-BY-SA-4.0",
        },
        notes=[
            "Independently verified lexical evidence — not statistical.",
            "Do not merge into NT-aligner training observations.",
            "CC BY-SA 4.0 ShareAlike applies to this adapted lexical subset.",
            "Greek loanword ≠ proof of reconstructed Greek inflection.",
            "Illustrative NT span ids (when present) are JAK-32 handles from "
            "provisional gold — not Thomas fragment Greek.",
            "Usable by JAK-39 as hard constraints / stronger L-evidence.",
        ],
    )


def write_verified_loanword_table(
    table: VerifiedLoanwordTable,
    path: Path,
    *,
    compact: bool = False,
) -> None:
    """Serialize the verified loanword table (pretty by default for review)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = table.model_dump(mode="json")
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )
    path.write_text(text + "\n", encoding="utf-8")


def load_verified_loanword_table(path: Path) -> VerifiedLoanwordTable:
    """Load a verified loanword JSON artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VerifiedLoanwordTable.model_validate(payload)
