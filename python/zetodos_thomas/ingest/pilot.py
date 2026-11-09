# SPDX-License-Identifier: Apache-2.0
"""Build and load the Horner x Vaticanus B* pilot parallel excerpt."""

from __future__ import annotations

import json
from pathlib import Path

from zetodos_thomas.ingest.copsahhorner import load_coptic_verses
from zetodos_thomas.ingest.download import NTVMR_TEIRAW_FILES
from zetodos_thomas.ingest.ntvmr_vaticanus import (
    SOURCE_ID as VATICANUS_SOURCE_ID,
)
from zetodos_thomas.ingest.ntvmr_vaticanus import (
    load_vaticanus_verses,
    teiraw_filename,
)
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus

# Representative Phase 1-2 sample (JAK-24): prologue + gospel opening +
# pericope-adulterae empty slots (documented mismatch / lacuna).
PILOT_RANGES: tuple[tuple[str, int, int, int], ...] = (
    ("john", 1, 1, 18),
    ("mark", 1, 1, 8),
    ("john", 8, 1, 11),
)

DEFAULT_PILOT_ID = "horner-vaticanus-nt-pilot-v1"


def _tei_paths_for_pilot(vaticanus_tei_dir: Path) -> list[Path]:
    return [vaticanus_tei_dir / teiraw_filename(osis) for osis in NTVMR_TEIRAW_FILES]


def build_pilot_corpus(
    *,
    sword_root: Path,
    vaticanus_tei_dir: Path,
    greek_source_id: str = VATICANUS_SOURCE_ID,
    sahidic_source_id: str = "copsahhorner",
) -> PilotCorpus:
    """Extract the pilot ranges from local CopSahHorner + Vaticanus B* TEI."""
    tei_paths = _tei_paths_for_pilot(vaticanus_tei_dir)
    units: list[ParallelUnit] = []
    for book, chapter, verse_from, verse_to in PILOT_RANGES:
        greek_units = {
            u.verse: u
            for u in load_vaticanus_verses(
                tei_paths,
                book,
                chapter,
                verse_from,
                verse_to,
                source_id=greek_source_id,
            )
        }
        sahidic_units = {
            u.verse: u
            for u in load_coptic_verses(
                sword_root,
                book,
                chapter,
                verse_from,
                verse_to,
                source_id=sahidic_source_id,
            )
        }
        for verse in range(verse_from, verse_to + 1):
            greek = greek_units.get(verse)
            sahidic = sahidic_units.get(verse)
            if greek is None:
                msg = f"missing Vaticanus B* verse {book} {chapter}:{verse}"
                raise ValueError(msg)
            if sahidic is None:
                msg = f"missing CopSahHorner verse slot {book} {chapter}:{verse}"
                raise ValueError(msg)
            note_parts: list[str] = []
            if not sahidic.original_text:
                note_parts.append(
                    "Sahidic module has empty text for this NRSV verse slot "
                    "(likely tradition omission or versification gap)."
                )
            if not greek.original_text:
                note_parts.append(
                    "Vaticanus B* (firsthand) has empty text for this verse "
                    "(lacuna / absent passage; not filled from WH or any other "
                    "Greek edition)."
                )
            units.append(
                ParallelUnit(
                    unit_id=f"{book}.{chapter}.{verse}",
                    greek=greek,
                    sahidic=sahidic,
                    notes=" ".join(note_parts),
                )
            )

    return PilotCorpus(
        id=DEFAULT_PILOT_ID,
        title=(
            "Horner Sahidic NT x NTVMR Vaticanus B* Greek NT pilot "
            "(John 1:1-18, Mark 1:1-8, John 8:1-11)"
        ),
        greek_source_id=greek_source_id,
        sahidic_source_id=sahidic_source_id,
        units=units,
        notes=(
            "Public-domain Sahidic (Horner/CrossWire) + CC BY 4.0 INTF/NTVMR "
            "Vaticanus GA 03 firsthand (B*) pilot for lexical/phrase alignment. "
            "Greek = Vaticanus B* only (JAK-25 / updated JAK-15); Westcott-Hort "
            "and SBLGNT are not used. John 8:1-11 is empty on both sides "
            "(PA absent in both traditions). Empty firsthand readings are "
            "omissions, not corrector fill."
        ),
    )


def write_pilot_corpus(corpus: PilotCorpus, path: Path) -> None:
    """Serialize a pilot corpus as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        corpus.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def load_pilot_corpus(path: Path) -> PilotCorpus:
    """Load and validate a committed pilot corpus JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return PilotCorpus.model_validate(data)
