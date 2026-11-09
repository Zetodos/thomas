# SPDX-License-Identifier: Apache-2.0
"""Parse DCLP EpiDoc TEI for Greek Thomas fragments into layered tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from zetodos_thomas.schemas.fragment import (
    CorpusRole,
    EvidenceType,
    FragmentLayerKind,
    FragmentLine,
    FragmentToken,
    FragmentTranscription,
)

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}
TEI = f"{{{TEI_NS}}}"

_XML_WS_RE = re.compile(r"[ \t]*\n[ \t]*")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


def _clean_surface(text: str) -> str:
    """Remove TEI pretty-print newlines; keep single spaces between words."""
    cleaned = _XML_WS_RE.sub("", text)
    return _MULTI_SPACE_RE.sub(" ", cleaned)


@dataclass
class _LineBuilder:
    """Accumulate tokens; flush on TEI ``lb`` (handles nested lbs in ``lem``)."""

    side_or_column: str | None
    lines: list[FragmentLine] = field(default_factory=list)
    _n: str | None = None
    _tokens: list[FragmentToken] = field(default_factory=list)
    _seq: int = 0

    def start_line(self, n: str, *, break_no: bool) -> None:
        del break_no  # word-continuation hint; lines still keyed by @n
        if self._n == n:
            # Same @n repeated (DCLP sometimes splits one line) — keep accumulating.
            return
        self.flush()
        self._n = n

    def add(self, token: FragmentToken) -> None:
        if self._n is None:
            # Rare leading content before first lb — ignore pure whitespace.
            if token.kind is FragmentLayerKind.SURVIVING and not token.text.strip():
                return
            self._n = "0"
        if (
            token.kind is FragmentLayerKind.SURVIVING
            and self._tokens
            and self._tokens[-1].kind is FragmentLayerKind.SURVIVING
            and self._tokens[-1].certainty == token.certainty
        ):
            self._tokens[-1].text += token.text
            return
        self._tokens.append(token)

    def flush(self) -> None:
        if self._n is None:
            self._tokens = []
            return
        normalized: list[FragmentToken] = []
        for tok in self._tokens:
            if tok.kind is FragmentLayerKind.SURVIVING:
                text = _clean_surface(tok.text)
                if text == "":
                    continue
                tok = FragmentToken(
                    kind=tok.kind,
                    text=text,
                    reason=tok.reason,
                    quantity=tok.quantity,
                    unit=tok.unit,
                    certainty=tok.certainty,
                )
            elif tok.kind is FragmentLayerKind.RESTORATION:
                text = _clean_surface(tok.text)
                if text == "" and tok.reason != "lost":
                    continue
                tok = FragmentToken(
                    kind=tok.kind,
                    text=text,
                    reason=tok.reason,
                    quantity=tok.quantity,
                    unit=tok.unit,
                    certainty=tok.certainty,
                )
            normalized.append(tok)
        has_content = any(
            (t.kind is FragmentLayerKind.LACUNA)
            or (t.text.strip() != "")
            or (t.kind is FragmentLayerKind.RESTORATION and t.reason == "lost")
            for t in normalized
        )
        if has_content:
            self._seq += 1
            side = self.side_or_column or "main"
            self.lines.append(
                FragmentLine(
                    line_id=f"{side}.{self._n}",
                    n=self._n,
                    side_or_column=self.side_or_column,
                    tokens=normalized,
                )
            )
        self._tokens = []
        self._n = None


def _walk(el: ET.Element, builder: _LineBuilder) -> None:
    tag = _local(el.tag)

    if tag == "lb":
        n = el.get("n") or "?"
        builder.start_line(n, break_no=el.get("break") == "no")
        return

    if tag in {"note", "head", "space", "milestone", "figure", "graphic"}:
        return

    if tag == "gap":
        builder.add(
            FragmentToken(
                kind=FragmentLayerKind.LACUNA,
                text="",
                reason=el.get("reason"),
                quantity=el.get("quantity"),
                unit=el.get("unit"),
            )
        )
        return

    if tag == "supplied":
        # Entire supplied span is scholarly restoration (may nest expan/unclear).
        text = "".join(el.itertext())
        builder.add(
            FragmentToken(
                kind=FragmentLayerKind.RESTORATION,
                text=text,
                reason=el.get("reason") or "lost",
            )
        )
        return

    if tag == "app":
        lem = el.find(f"{TEI}lem")
        if lem is not None:
            _walk_children(lem, builder)
        return

    if tag == "choice":
        orig = el.find(f"{TEI}orig")
        target = orig if orig is not None else el.find(f"{TEI}reg")
        if target is not None:
            _walk_children(target, builder)
        return

    if tag == "subst":
        add = el.find(f"{TEI}add")
        if add is not None:
            _walk_children(add, builder)
        return

    if tag == "ex":
        builder.add(
            FragmentToken(
                kind=FragmentLayerKind.RESTORATION,
                text="".join(el.itertext()),
                reason="abbreviation",
            )
        )
        return

    if tag == "unclear":
        text = "".join(el.itertext())
        builder.add(
            FragmentToken(
                kind=FragmentLayerKind.SURVIVING,
                text=text,
                certainty="unclear",
            )
        )
        return

    if tag in {"rdg", "del"}:
        return

    # Default: emit element text as surviving, recurse into children.
    if el.text:
        builder.add(FragmentToken(kind=FragmentLayerKind.SURVIVING, text=el.text))
    for child in list(el):
        _walk(child, builder)
        if child.tail:
            builder.add(
                FragmentToken(kind=FragmentLayerKind.SURVIVING, text=child.tail)
            )


def _walk_children(el: ET.Element, builder: _LineBuilder) -> None:
    if el.text:
        builder.add(FragmentToken(kind=FragmentLayerKind.SURVIVING, text=el.text))
    for child in list(el):
        _walk(child, builder)
        if child.tail:
            builder.add(
                FragmentToken(kind=FragmentLayerKind.SURVIVING, text=child.tail)
            )


def _side_label(textpart: ET.Element) -> str | None:
    n = textpart.get("n")
    subtype = textpart.get("subtype")
    if n and subtype:
        return f"{subtype}-{n}"
    return n or subtype


def parse_dclp_edition(path: Path) -> list[FragmentLine]:
    """Parse the Greek ``div[@type='edition']`` into layered fragment lines."""
    root = ET.parse(path).getroot()
    edition = root.find('.//tei:div[@type="edition"]', NS)
    if edition is None:
        msg = f"no tei:div[@type='edition'] in {path}"
        raise ValueError(msg)

    textparts = edition.findall('./tei:div[@type="textpart"]', NS)
    containers: list[ET.Element] = textparts if textparts else [edition]
    lines: list[FragmentLine] = []
    for part in containers:
        builder = _LineBuilder(side_or_column=_side_label(part))
        abs_ = part.findall("./tei:ab", NS) or part.findall(".//tei:ab", NS)
        if not abs_ and part is edition:
            _walk_children(edition, builder)
            builder.flush()
        else:
            for ab in abs_:
                _walk_children(ab, builder)
                builder.flush()
        lines.extend(builder.lines)
    return lines


def read_dclp_license(path: Path) -> str | None:
    """Return license URL from TEI availability, if present."""
    root = ET.parse(path).getroot()
    ref = root.find('.//tei:availability//tei:ref[@type="license"]', NS)
    if ref is None:
        return None
    return ref.get("target")


def read_dclp_ids(path: Path) -> dict[str, str]:
    """Extract TM / DCLP / hybrid identifiers from the TEI header."""
    root = ET.parse(path).getroot()
    out: dict[str, str] = {}
    for idno in root.findall(".//tei:idno", NS):
        kind = idno.get("type")
        if kind and idno.text:
            out[kind] = idno.text.strip()
    return out


# Catalog of confirmed Greek Thomas witnesses for this ingest (JAK-31).
DCLP_THOMAS_FRAGMENTS: tuple[dict[str, str], ...] = (
    {
        "source_id": "dclp-poxy-1",
        "transcription_id": "poxy-1-dclp",
        "manuscript_id": "P.Oxy. 1",
        "tm_id": "62838",
        "filename": "poxy-1-tm62838.xml",
        "logia_coverage": "26–30, 77b, 31–33",
        "title": "DCLP transcription of P.Oxy. 1 (Gospel of Thomas)",
    },
    {
        "source_id": "dclp-poxy-654",
        "transcription_id": "poxy-654-dclp",
        "manuscript_id": "P.Oxy. 654",
        "tm_id": "62840",
        "filename": "poxy-654-tm62840.xml",
        "logia_coverage": "incipit, 1–7",
        "title": "DCLP transcription of P.Oxy. 654 (Gospel of Thomas)",
    },
    {
        "source_id": "dclp-poxy-655",
        "transcription_id": "poxy-655-dclp",
        "manuscript_id": "P.Oxy. 655",
        "tm_id": "62839",
        "filename": "poxy-655-tm62839.xml",
        "logia_coverage": "24(?), 36–39",
        "title": "DCLP transcription of P.Oxy. 655 (Gospel of Thomas)",
    },
)

# Independent parallels — do not ingest as Thomas manuscripts.
EXCLUDED_NON_THOMAS_WITNESSES: tuple[str, ...] = (
    "P.Oxy. 5575 (TM 989105) — independent sayings parallel; not a Thomas manuscript",
)

IDP_DATA_TEI_COMMIT = "d9387868bbe3"
IDP_DATA_RAW_BASE = (
    f"https://raw.githubusercontent.com/papyri/idp.data/{IDP_DATA_TEI_COMMIT}/DCLP/62"
)


def build_transcription(
    path: Path,
    *,
    source_id: str,
    transcription_id: str,
    manuscript_id: str,
    tm_id: str,
    logia_coverage: str,
) -> FragmentTranscription:
    """Build a layered transcription from a local DCLP TEI file."""
    license_url = read_dclp_license(path)
    if not license_url or "creativecommons.org/licenses/by/3.0" not in license_url:
        msg = f"{path.name}: missing or unexpected DCLP CC BY 3.0 license ref"
        raise ValueError(msg)
    ids = read_dclp_ids(path)
    if ids.get("TM") != tm_id:
        msg = f"{path.name}: TM id {ids.get('TM')!r} != expected {tm_id!r}"
        raise ValueError(msg)
    lines = parse_dclp_edition(path)
    if not lines:
        msg = f"{path.name}: parsed zero fragment lines"
        raise ValueError(msg)
    return FragmentTranscription(
        id=transcription_id,
        source_id=source_id,
        manuscript_id=manuscript_id,
        tm_id=tm_id,
        dclp_id=ids.get("dclp", tm_id),
        evidence_type=EvidenceType.G,
        corpus_role=CorpusRole.EVALUATION_WITNESS,
        logia_coverage=logia_coverage,
        lines=lines,
        notes=(
            "Layers: surviving (incl. unclear), lacuna (tei:gap), restoration "
            "(tei:supplied + abbreviation expansions in tei:ex). Editorial "
            "apparatus uses lem only; rdg (Attridge/Fitzmyer/etc.) are not "
            "promoted into layers. Corpus role=evaluation-witness — do not feed "
            "into Greek–Sahidic alignment training (JAK-28) without an explicit "
            "holdout decision."
        ),
    )
