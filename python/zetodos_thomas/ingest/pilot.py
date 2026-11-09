# SPDX-License-Identifier: Apache-2.0
"""Build and load the Horner x Westcott-Hort pilot parallel excerpt."""

from __future__ import annotations

import json
from pathlib import Path

from zetodos_thomas.ingest.copsahhorner import load_coptic_verses
from zetodos_thomas.ingest.westcott_hort import load_wh_verses
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus

# Representative Phase 1-2 sample (JAK-24): prologue + gospel opening +
# pericope-adulterae empty slots (documented mismatch).
PILOT_RANGES: tuple[tuple[str, int, int, int], ...] = (
    ("john", 1, 1, 18),
    ("mark", 1, 1, 8),
    ("john", 8, 1, 11),
)

DEFAULT_PILOT_ID = "horner-wh-nt-pilot-v1"


def build_pilot_corpus(
    *,
    sword_root: Path,
    wh_textonly_dir: Path,
    greek_source_id: str = "westcott-hort",
    sahidic_source_id: str = "copsahhorner",
) -> PilotCorpus:
    """Extract the pilot ranges from local CopSahHorner + WH textonly files."""
    units: list[ParallelUnit] = []
    for book, chapter, verse_from, verse_to in PILOT_RANGES:
        greek_units = {
            u.verse: u
            for u in load_wh_verses(
                wh_textonly_dir,
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
                msg = f"missing WH verse {book} {chapter}:{verse}"
                raise ValueError(msg)
            if sahidic is None:
                msg = f"missing CopSahHorner verse slot {book} {chapter}:{verse}"
                raise ValueError(msg)
            note = ""
            if not sahidic.original_text:
                note = (
                    "Sahidic module has empty text for this NRSV verse slot "
                    "(likely tradition omission or versification gap)."
                )
            elif not greek.original_text:
                note = "Westcott-Hort text is empty for this verse slot."
            units.append(
                ParallelUnit(
                    unit_id=f"{book}.{chapter}.{verse}",
                    greek=greek,
                    sahidic=sahidic,
                    notes=note,
                )
            )

    return PilotCorpus(
        id=DEFAULT_PILOT_ID,
        title=(
            "Horner Sahidic NT x Westcott-Hort Greek NT pilot "
            "(John 1:1-18, Mark 1:1-8, John 8:1-11)"
        ),
        greek_source_id=greek_source_id,
        sahidic_source_id=sahidic_source_id,
        units=units,
        notes=(
            "Public-domain pilot excerpt for lexical/phrase alignment experiments. "
            "Greek = Westcott-Hort (Robinson transcription), not SBLGNT. "
            "Sahidic = CrossWire CopSahHorner v1.5. "
            "John 8:1-11 documents empty Sahidic slots vs Greek tradition."
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
