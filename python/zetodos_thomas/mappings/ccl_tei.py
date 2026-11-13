# SPDX-License-Identifier: Apache-2.0
"""Checksum-pinned CCL/DDGLC TEI helpers for JAK-38 loanword verification."""

from __future__ import annotations

import hashlib
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

PINNED_CCL_TEI_SHA256 = (
    "ace84fce11b76386a6adb9888b2cb947056954fc9fe9c04aa0676b1f595fd43d"
)
# Bytes of the committed offline subset extracted from the SHA-pinned full TEI.
# Independent of the parent-TEI SHA claimed in the XML comment — both must hold.
PINNED_CCL_TEI_SUBSET_SHA256 = (
    "fc1bc945980acc70428ff6bfcd87b864b6f0209d8c04a26aa9baa536f3f40cd6"
)
FULL_TEI_BASENAME = "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
SUBSET_TEI_BASENAME = "ccl-ddglc-tei-verified-subset-v1.xml"
DEFAULT_SUBSET_REL = f"data/sources/mappings/fixtures/{SUBSET_TEI_BASENAME}"


@dataclass(frozen=True)
class CclTeiEntry:
    """Fields read from one CCL TEI ``<entry>``."""

    tla_entry_id: str
    form_ids: frozenset[str]
    orths: frozenset[str]
    greek_lemma_id: str | None
    greek_lemma: str | None
    greek_meaning: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_ccl_tei_subset_path(repo_root: Path) -> Path:
    return repo_root / DEFAULT_SUBSET_REL


def default_ccl_tei_download_path(repo_root: Path) -> Path:
    return repo_root / "data" / "downloads" / "ccl-ddglc" / FULL_TEI_BASENAME


def verify_ccl_tei_path(path: Path) -> str:
    """Fail closed unless ``path`` is the pinned full TEI or pinned subset.

    Returns ``\"full\"`` or ``\"subset\"`` for the pin that matched.
    Alternate basenames and checksum mismatches raise ``ValueError``.
    """
    digest = sha256_file(path)
    if path.name == FULL_TEI_BASENAME:
        if digest != PINNED_CCL_TEI_SHA256:
            msg = (
                f"CCL TEI checksum mismatch: got {digest}, "
                f"expected {PINNED_CCL_TEI_SHA256}"
            )
            raise ValueError(msg)
        return "full"
    if path.name == SUBSET_TEI_BASENAME:
        if digest != PINNED_CCL_TEI_SUBSET_SHA256:
            msg = (
                f"CCL TEI subset checksum mismatch: got {digest}, "
                f"expected {PINNED_CCL_TEI_SUBSET_SHA256}"
            )
            raise ValueError(msg)
        return "subset"
    msg = (
        f"untrusted CCL TEI path {path.name!r}: expected pinned full TEI "
        f"({FULL_TEI_BASENAME}) or offline subset ({SUBSET_TEI_BASENAME})"
    )
    raise ValueError(msg)


def resolve_ccl_tei_path(repo_root: Path, *, prefer_full: bool = False) -> Path:
    """Prefer full downloaded TEI when present; otherwise the offline subset.

    Returned path is always checksum-verified against the matching pin.
    """
    full = default_ccl_tei_download_path(repo_root)
    subset = default_ccl_tei_subset_path(repo_root)
    if prefer_full and full.is_file():
        verify_ccl_tei_path(full)
        return full
    if full.is_file():
        verify_ccl_tei_path(full)
        return full
    if subset.is_file():
        verify_ccl_tei_path(subset)
        return subset
    msg = (
        f"CCL TEI not found: expected download at {full} or offline subset at {subset}"
    )
    raise FileNotFoundError(msg)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _undiacriticize_greek(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def load_ccl_tei_index(tei_path: Path) -> dict[str, CclTeiEntry]:
    """Index TEI entries by TLA ``xml:id`` (C…)."""
    index: dict[str, CclTeiEntry] = {}
    for _event, elem in ET.iterparse(tei_path, events=("end",)):
        if elem.tag.split("}")[-1] != "entry":
            continue
        xid = elem.attrib.get(XML_ID)
        if not xid or not xid.startswith("C"):
            elem.clear()
            continue
        form_ids: set[str] = set()
        orths: set[str] = set()
        for form in elem.findall("tei:form", TEI_NS):
            fid = form.attrib.get(XML_ID)
            orth = form.findtext("tei:orth", default="", namespaces=TEI_NS)
            if fid:
                form_ids.add(fid)
            if orth:
                orths.add(_nfc(orth))
        greek_id = None
        greek_lemma = None
        greek_meaning = None
        for ref in elem.findall(".//tei:ref", TEI_NS):
            rtype = ref.attrib.get("type", "")
            text = _nfc((ref.text or "").strip())
            if rtype == "greek_lemma::grl_ID":
                greek_id = text
            elif rtype == "greek_lemma::grl_lemma":
                greek_lemma = text
            elif rtype == "greek_lemma::grl_meaning":
                greek_meaning = text
        index[xid] = CclTeiEntry(
            tla_entry_id=xid,
            form_ids=frozenset(form_ids),
            orths=frozenset(orths),
            greek_lemma_id=greek_id,
            greek_lemma=greek_lemma,
            greek_meaning=greek_meaning,
        )
        elem.clear()
    return index


def verify_loanword_against_tei(
    *,
    tla_entry_id: str,
    coptic_lemma: str,
    coptic_forms: list[str],
    greek_lemma: str,
    ddglc_greek_lemma_id: str,
    ccl_form_ids: list[str],
    tei_index: dict[str, CclTeiEntry],
) -> list[str]:
    """Return human-readable mismatch messages (empty ⇒ pass)."""
    errors: list[str] = []
    entry = tei_index.get(tla_entry_id)
    if entry is None:
        return [f"TLA entry {tla_entry_id} not present in TEI"]
    if entry.greek_lemma_id != ddglc_greek_lemma_id:
        errors.append(
            f"{tla_entry_id}: DDGLC id {ddglc_greek_lemma_id!r} != "
            f"TEI {entry.greek_lemma_id!r}"
        )
    if entry.greek_lemma is None:
        errors.append(f"{tla_entry_id}: TEI missing greek_lemma::grl_lemma")
    else:
        want = _undiacriticize_greek(_nfc(greek_lemma))
        got = _undiacriticize_greek(_nfc(entry.greek_lemma))
        if want != got:
            errors.append(
                f"{tla_entry_id}: greek lemma {greek_lemma!r} != "
                f"TEI {entry.greek_lemma!r}"
            )
    lemma = _nfc(coptic_lemma)
    if lemma not in entry.orths:
        errors.append(
            f"{tla_entry_id}: coptic lemma {coptic_lemma!r} not among TEI orths "
            f"{sorted(entry.orths)[:8]}"
        )
    for form in coptic_forms:
        if _nfc(form) not in entry.orths:
            errors.append(
                f"{tla_entry_id}: coptic form {form!r} not attested in TEI orths"
            )
    for fid in ccl_form_ids:
        if fid not in entry.form_ids:
            errors.append(f"{tla_entry_id}: form id {fid} not in TEI entry")
    return errors
