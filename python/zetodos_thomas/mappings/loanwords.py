# SPDX-License-Identifier: Apache-2.0
"""Minimal CCL/DDGLC verified loanword subset (JAK-38 / JAK-19).

Entries are independently verified against the checksum-pinned CCL TEI
(v1.2) — either the full setup-download file or the offline entry subset
fixture — and must never be mixed into statistical training observations.
License: CC BY-SA 4.0 on adapted lexical exports.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zetodos_thomas.mappings.ccl_tei import (
    PINNED_CCL_TEI_SHA256,
    default_ccl_tei_subset_path,
    load_ccl_tei_index,
    resolve_ccl_tei_path,
    sha256_file,
    verify_loanword_against_tei,
)
from zetodos_thomas.schemas.mappings import (
    EvidenceClass,
    SpotCheckRecord,
    VerifiedLoanword,
    VerifiedLoanwordTable,
)

# Curated from Comprehensive Coptic Lexicon v1.2 (Burns/Feder/John/Kupreyev;
# DDGLC Greek loanwords). Verified against TEI entry fields.
# DOI: 10.17169/refubium-27566; KELLIA xml pin below.
_CCL_ATTRIBUTION = (
    "Comprehensive Coptic Lexicon v1.2 — Burns, Dylan Michael; Feder, Frank; "
    "John, Katrin; Kupreyev, Maxim; BBAW Thesaurus Linguae Aegyptiae; "
    "DDGLC (Freie Universität Berlin). CC BY-SA 4.0. "
    "DOI https://doi.org/10.17169/refubium-27566"
)

_SPOT_METHOD = (
    "Reproducible check against checksum-pinned CCL TEI v1.2 entry: "
    "xml:id (TLA C-id), form xml:id(s), orth, greek_lemma::grl_ID, "
    "greek_lemma::grl_lemma. Constructor does not certify pass without TEI match."
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


def _spot_for(
    *,
    tla_entry_id: str,
    ccl_form_ids: list[str],
    tei_orth: str,
    tei_greek_lemma: str,
    tei_greek_lemma_id: str,
    checked_at_utc: str,
) -> SpotCheckRecord:
    return SpotCheckRecord(
        checked_by="jak-38-ccl-tei-verify",
        checked_at_utc=checked_at_utc,
        method=_SPOT_METHOD,
        result="pass",
        notes="Lexical correspondence only; inflection not entailed.",
        tei_entry_xml_id=tla_entry_id,
        tei_form_xml_ids=list(ccl_form_ids),
        tei_orth_attested=tei_orth,
        tei_greek_lemma=tei_greek_lemma,
        tei_greek_lemma_id=tei_greek_lemma_id,
        parent_tei_sha256=PINNED_CCL_TEI_SHA256,
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
    spot: SpotCheckRecord,
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
        spot_check=spot,
        illustrative_nt_span_ids=list(illustrative_nt_span_ids or []),
        illustrative_spans_are_non_training=True,
        pilot_priority_notes=pilot_priority_notes,
        notes=notes,
    )


def _curated_specs() -> list[dict[str, Any]]:
    """Curated Phase-2 subset (fields must match TEI or write fails)."""
    return [
        {
            "loanword_id": "verified-makarios",
            "coptic_lemma": "ⲙⲁⲕⲁⲣⲓⲟⲥ",
            "coptic_forms": ["ⲙⲁⲕⲁⲣⲓⲟⲥ", "ⲙⲁⲕⲁⲣⲓⲟⲛ"],
            "greek_lemma": "μακάριος",
            "greek_pos": "adjective",
            "greek_meaning": "blessed, happy",
            "tla_entry_id": "C9689",
            "ddglc_greek_lemma_id": "1336",
            "ccl_form_ids": ["CF24876"],
            "pilot_priority_notes": (
                "Motivated by Thomas 7 Coptic ⲙⲁⲕⲁⲣⲓⲟⲥ (L clue). "
                "Do not consult P.Oxy. 654 Greek when verifying this row."
            ),
            "notes": "DDGLC sense bibl present on CCL TEI entry.",
            "illustrative_nt_span_ids": [],
        },
        {
            "loanword_id": "verified-euaggelion",
            "coptic_lemma": "ⲉⲩⲁⲅⲅⲉⲗⲓⲟⲛ",
            "coptic_forms": ["ⲉⲩⲁⲅⲅⲉⲗⲓⲟⲛ", "ⲉⲩⲁⲅ"],
            "greek_lemma": "εὐαγγέλιον",
            "greek_pos": "noun",
            "greek_meaning": "good news, gospel",
            "tla_entry_id": "C9059",
            "ddglc_greek_lemma_id": "809",
            "ccl_form_ids": ["CF22411"],
            "illustrative_nt_span_ids": [
                "v1:copsahhorner/mark.1.1/s/euaggelion",
                "v1:ntvmr-vaticanus-b/mark.1.1/s/euaggeliou",
            ],
            "notes": "NT gold sample also tags this as loanword (illustrative only).",
        },
        {
            "loanword_id": "verified-mathetes",
            "coptic_lemma": "ⲙⲁⲑⲏⲧⲏⲥ",
            "coptic_forms": ["ⲙⲁⲑⲏⲧⲏⲥ", "ⲙⲁⲑⲏⲧ"],
            "greek_lemma": "μαθητής",
            "greek_pos": "noun",
            "greek_meaning": "learner, pupil, disciple",
            "tla_entry_id": "C9683",
            "ddglc_greek_lemma_id": "1334",
            "ccl_form_ids": ["CF24841"],
            # luke.6.20 is dev — do not use as illustrative (train-only policy).
            "illustrative_nt_span_ids": [],
        },
        {
            "loanword_id": "verified-aitei",
            "coptic_lemma": "ⲁⲓⲧⲉⲓ",
            "coptic_forms": ["ⲁⲓⲧⲉⲓ", "ⲁⲓⲧⲉ"],
            "greek_lemma": "αἰτέω",
            "greek_pos": "verb",
            "greek_meaning": "to ask (s.o. for s.th.)",
            "tla_entry_id": "C8126",
            "ddglc_greek_lemma_id": "54",
            "ccl_form_ids": ["CF18503"],
            "illustrative_nt_span_ids": [
                "v1:copsahhorner/matthew.7.7/s/aitei",
                "v1:ntvmr-vaticanus-b/matthew.7.7/s/aiteite",
            ],
        },
        {
            "loanword_id": "verified-parabole",
            "coptic_lemma": "ⲡⲁⲣⲁⲃⲟⲗⲏ",
            "coptic_forms": ["ⲡⲁⲣⲁⲃⲟⲗⲏ"],
            "greek_lemma": "παραβολή",
            "greek_pos": "noun",
            "greek_meaning": "juxtaposition, comparison; parable",
            "tla_entry_id": "C10032",
            "ddglc_greek_lemma_id": "2558",
            "ccl_form_ids": ["CF26233"],
            # matthew.13.31 is holdout — omitted from illustrative spans.
            "illustrative_nt_span_ids": [],
        },
        {
            "loanword_id": "verified-de",
            "coptic_lemma": "ⲇⲉ",
            "coptic_forms": ["ⲇⲉ"],
            "greek_lemma": "δέ",
            "greek_pos": "particle",
            "greek_meaning": "but, and, so, then",
            "tla_entry_id": "C8719",
            "ddglc_greek_lemma_id": "458",
            "ccl_form_ids": ["CF20757"],
        },
        {
            "loanword_id": "verified-gar",
            "coptic_lemma": "ⲅⲁⲣ",
            "coptic_forms": ["ⲅⲁⲣ"],
            "greek_lemma": "γάρ",
            "greek_pos": "particle",
            "greek_meaning": "for, then, because",
            "tla_entry_id": "C8657",
            "ddglc_greek_lemma_id": "415",
            "ccl_form_ids": ["CF20474"],
        },
        {
            "loanword_id": "verified-alla",
            "coptic_lemma": "ⲁⲗⲗⲁ",
            "coptic_forms": ["ⲁⲗⲗⲁ"],
            "greek_lemma": "ἀλλά",
            "greek_pos": "conjunction",
            "greek_meaning": "but, yet, rather",
            "tla_entry_id": "C8179",
            "ddglc_greek_lemma_id": "95",
            "ccl_form_ids": ["CF18776"],
        },
        {
            "loanword_id": "verified-oude",
            "coptic_lemma": "ⲟⲩⲇⲉ",
            "coptic_forms": ["ⲟⲩⲇⲉ"],
            "greek_lemma": "οὐδέ",
            "greek_pos": "conjunction",
            "greek_meaning": "nor, and not, not even",
            "tla_entry_id": "C9959",
            "ddglc_greek_lemma_id": "1592",
            "ccl_form_ids": ["CF25965"],
        },
        {
            "loanword_id": "verified-psyche",
            "coptic_lemma": "ⲯⲩⲭⲏ",
            "coptic_forms": ["ⲯⲩⲭⲏ"],
            "greek_lemma": "ψυχή",
            "greek_pos": "noun",
            "greek_meaning": "life, spirit, soul, self",
            "tla_entry_id": "C11031",
            "ddglc_greek_lemma_id": "2433",
            "ccl_form_ids": ["CF30301"],
        },
        {
            "loanword_id": "verified-kosmos",
            "coptic_lemma": "ⲕⲟⲥⲙⲟⲥ",
            "coptic_forms": ["ⲕⲟⲥⲙⲟⲥ"],
            "greek_lemma": "κόσμος",
            "greek_pos": "noun",
            "greek_meaning": "order, ornament, world, universe",
            "tla_entry_id": "C9517",
            "ddglc_greek_lemma_id": "1192",
            "ccl_form_ids": ["CF24208"],
        },
        {
            "loanword_id": "verified-pneuma",
            "coptic_lemma": "ⲡⲛⲉⲩⲙⲁ",
            "coptic_forms": ["ⲡⲛⲉⲩⲙⲁ"],
            "greek_lemma": "πνεῦμα",
            "greek_pos": "noun",
            "greek_meaning": "blast, wind, breath, spirit",
            "tla_entry_id": "C10220",
            "ddglc_greek_lemma_id": "1775",
            "ccl_form_ids": ["CF27013"],
        },
        {
            "loanword_id": "verified-sarx",
            "coptic_lemma": "ⲥⲁⲣⲝ",
            "coptic_forms": ["ⲥⲁⲣⲝ"],
            "greek_lemma": "σάρξ",
            "greek_pos": "noun",
            "greek_meaning": "flesh",
            "tla_entry_id": "C10421",
            "ddglc_greek_lemma_id": "1937",
            "ccl_form_ids": ["CF27869"],
        },
        {
            "loanword_id": "verified-angelos",
            "coptic_lemma": "ⲁⲅⲅⲉⲗⲟⲥ",
            "coptic_forms": ["ⲁⲅⲅⲉⲗⲟⲥ", "ⲁⲅⲅⲉⲗⲟ"],
            "greek_lemma": "ἄγγελος",
            "greek_pos": "noun",
            "greek_meaning": "messenger, angel",
            "tla_entry_id": "C8062",
            "ddglc_greek_lemma_id": "11",
            "ccl_form_ids": ["CF18246"],
        },
    ]


def verified_loanword_entries(
    *,
    repo_root: Path | None = None,
    tei_path: Path | None = None,
) -> list[VerifiedLoanword]:
    """Build curated rows after TEI verification (fails closed on mismatch)."""
    root = repo_root or Path.cwd()
    path = tei_path or resolve_ccl_tei_path(root)
    # Offline subset is extracted from the pinned full TEI; full file must match pin.
    if path.name == "Comprehensive_Coptic_Lexicon-v1.2-2020.xml":
        digest = sha256_file(path)
        if digest != PINNED_CCL_TEI_SHA256:
            msg = (
                f"CCL TEI checksum mismatch: got {digest}, "
                f"expected {PINNED_CCL_TEI_SHA256}"
            )
            raise ValueError(msg)
    index = load_ccl_tei_index(path)
    checked_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries: list[VerifiedLoanword] = []
    for spec in _curated_specs():
        mismatches = verify_loanword_against_tei(
            tla_entry_id=spec["tla_entry_id"],
            coptic_lemma=spec["coptic_lemma"],
            coptic_forms=list(spec["coptic_forms"]),
            greek_lemma=spec["greek_lemma"],
            ddglc_greek_lemma_id=spec["ddglc_greek_lemma_id"],
            ccl_form_ids=list(spec["ccl_form_ids"]),
            tei_index=index,
        )
        if mismatches:
            msg = "CCL TEI verification failed:\n- " + "\n- ".join(mismatches)
            raise ValueError(msg)
        tei_entry = index[spec["tla_entry_id"]]
        spot = _spot_for(
            tla_entry_id=spec["tla_entry_id"],
            ccl_form_ids=list(spec["ccl_form_ids"]),
            tei_orth=spec["coptic_lemma"],
            tei_greek_lemma=tei_entry.greek_lemma or spec["greek_lemma"],
            tei_greek_lemma_id=tei_entry.greek_lemma_id or spec["ddglc_greek_lemma_id"],
            checked_at_utc=checked_at,
        )
        entries.append(
            _entry(
                loanword_id=spec["loanword_id"],
                coptic_lemma=spec["coptic_lemma"],
                coptic_forms=list(spec["coptic_forms"]),
                greek_lemma=spec["greek_lemma"],
                greek_pos=spec["greek_pos"],
                greek_meaning=spec["greek_meaning"],
                tla_entry_id=spec["tla_entry_id"],
                ddglc_greek_lemma_id=spec["ddglc_greek_lemma_id"],
                ccl_form_ids=list(spec["ccl_form_ids"]),
                spot=spot,
                pilot_priority_notes=str(spec.get("pilot_priority_notes", "")),
                illustrative_nt_span_ids=list(
                    spec.get("illustrative_nt_span_ids") or []
                ),
                notes=str(spec.get("notes", "")),
            )
        )
    return entries


def verified_loanword_table(
    *,
    provenance: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    tei_path: Path | None = None,
) -> VerifiedLoanwordTable:
    """Assemble the verified loanword export (TEI-backed)."""
    root = repo_root or Path.cwd()
    path = tei_path or resolve_ccl_tei_path(root)
    entries = verified_loanword_entries(repo_root=root, tei_path=path)
    subset = default_ccl_tei_subset_path(root)
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
            "ccl_tei_sha256": PINNED_CCL_TEI_SHA256,
            "ccl_tei_verification_path": str(path.relative_to(root))
            if path.is_relative_to(root)
            else str(path),
            "ccl_tei_subset_fixture": str(subset.relative_to(root))
            if subset.is_file() and subset.is_relative_to(root)
            else None,
            "doi": "10.17169/refubium-27566",
            "license": "CC-BY-SA-4.0",
            "attribution": _CCL_ATTRIBUTION,
            "dioskoros": ("Not used; JAK-19 forbids waiting on / scraping DIOSKOROS."),
            **dict(provenance or {}),
        },
        summary={
            "entry_count": len(entries),
            "evidence_class": EvidenceClass.INDEPENDENTLY_VERIFIED.value,
            "license": "CC-BY-SA-4.0",
            "tei_verified": True,
        },
        notes=[
            "Independently verified lexical evidence — not statistical.",
            "Do not merge into NT-aligner training observations.",
            "CC BY-SA 4.0 ShareAlike applies to this adapted lexical subset.",
            "Greek loanword ≠ proof of reconstructed Greek inflection.",
            "Illustrative NT span ids (when present) are JAK-32 handles from "
            "train-split NT gold only — not Thomas fragment Greek, and never "
            "decoder-training exemplars (illustrative_spans_are_non_training).",
            "Usable by JAK-39 as hard constraints / stronger L-evidence.",
            "Certification requires TEI field match; spot_check.result=pass "
            "alone is insufficient.",
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
