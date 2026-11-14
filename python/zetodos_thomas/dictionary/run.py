# SPDX-License-Identifier: Apache-2.0
"""Fit parallel-dict-v1, apply it to Dilley Thomas, score fragment witnesses.

Fragment Greek is loaded only after the dictionary and the NT-dev snap
decision are finished. Seeing the score must not change those choices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zetodos_thomas.dictionary.apply import TokenOut, apply_tokens
from zetodos_thomas.dictionary.build import (
    CONTENT_PRECISION_BAR,
    CONTENT_RECALL_BAR,
    LOANWORD_MIN_SUPPORT,
    LOANWORD_SUFFICIENT_RATE,
    DictionaryBuildError,
    ObservedLink,
    ParallelDictionary,
    build_dictionary,
)
from zetodos_thomas.dictionary.diagnose import load_nt_splits, nt_dev_snap_diagnostic
from zetodos_thomas.dictionary.score import score_fragments
from zetodos_thomas.ingest.dilley_thomas import default_corpus_path, load_thomas_corpus
from zetodos_thomas.ingest.fragments import fragments_corpus_path, load_fragment_corpus
from zetodos_thomas.ingest.paths import resolve_pilot_path
from zetodos_thomas.ingest.pilot import load_pilot_corpus

VERSION = "parallel-dict-v1"
LEXICON_REL = Path("data/sources/lexicon/coptic-greek-lexical-candidates-v1.json")


def run_parallel_dictionary(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build, apply, and score. Returns the metrics payload."""
    lexicon_path = repo_root / LEXICON_REL
    links, unit_ids, aligners, provenance = links_from_candidate_table(lexicon_path)
    dictionary = build_dictionary(
        links,
        unit_ids=unit_ids,
        aligners=aligners,
        provenance=provenance,
        version=VERSION,
    )
    corpus = load_pilot_corpus(resolve_pilot_path(repo_root))
    splits = load_nt_splits(repo_root)
    diagnostic = nt_dev_snap_diagnostic(corpus, splits, dictionary)
    dictionary.snap_enabled = bool(diagnostic["snap_enabled"])

    thomas = load_thomas_corpus(default_corpus_path(repo_root))
    decisions: dict[int, list[TokenOut]] = {}
    units: list[dict[str, Any]] = []
    for unit in thomas.units:
        tokens = unit.normalized_text.split()
        outputs = apply_tokens(tokens, dictionary)
        decisions[unit.chapter] = outputs
        units.append(
            {
                "unit_id": unit.unit_id,
                "logion": unit.chapter,
                "tokens": [_token_row(item) for item in outputs],
            }
        )

    # Dictionary is frozen before any fragment text is read.
    fragments = load_fragment_corpus(fragments_corpus_path(repo_root))
    summary = score_fragments(decisions, fragments)
    sufficient, reason = _sufficient(summary)
    metrics: dict[str, Any] = {
        "id": f"{VERSION}-metrics",
        "version": VERSION,
        "uses_thomas_greek_as_training": False,
        "uses_handwritten_glosses": False,
        "uses_verified_loanword_table": False,
        "frozen_before_fragment_score": True,
        "snap_enabled": dictionary.snap_enabled,
        "nt_dev_diagnostic": diagnostic,
        "loanword_stem_match_rate": summary["loanword_stem_match_rate"],
        "loanword_exact_match_rate": summary["loanword_exact_match_rate"],
        "loanword_tokens": summary["loanword_tokens"],
        "content_type_recall": summary["content_type_recall"],
        "content_emission_precision": summary["content_emission_precision"],
        "fragment_type_recall": summary["fragment_type_recall"],
        "abstain_rate": summary["abstain_rate"],
        "approach_sufficient": sufficient,
        "approach_sufficient_reason": reason,
        "iteration_bars": {
            "loanword_stem_match_rate": LOANWORD_SUFFICIENT_RATE,
            "loanword_min_support": LOANWORD_MIN_SUPPORT,
            "content_type_recall": CONTENT_RECALL_BAR,
            "content_emission_precision": CONTENT_PRECISION_BAR,
            "note": (
                "Experiment iteration bars only. Not the Phase 2 feasibility gate "
                "and not tuned on fragment scores."
            ),
        },
        "next_step": _next_step(sufficient),
        "micro": {
            key: value for key, value in summary.items() if key != "by_manuscript"
        },
        "by_manuscript": summary["by_manuscript"],
        "dictionary_entry_count": len(dictionary.emitted_entries()),
        "notes": [
            "Train alignments of Horner Sahidic × Vaticanus B* only.",
            "Both eflomal and fast_align must support an emitted statistical pair.",
            "Inflections of one Greek stem share a posterior; the emitted "
            "string is the most frequent attested surface in that cluster.",
            "Abstain when aligners disagree or the posterior is weak.",
            "Orthographic snap may fire only for forms with no train pair, and "
            "only if the NT dev diagnostic clears its pre-set precision bar.",
            "Fragment Greek is an evaluation witness. Do not train on it.",
            "Loanword audit rows are evaluation-only.",
        ],
    }
    if write:
        dest = output_dir or (repo_root / "data" / "reconstructions" / VERSION)
        _write_outputs(dest, dictionary, units, metrics)
        metrics["output_dir"] = str(dest)
    return metrics


def links_from_candidate_table(
    path: Path,
) -> tuple[list[ObservedLink], list[str], list[str], dict[str, object]]:
    """Read the JAK-37 columnar table. Does not run aligners."""
    if not path.is_file():
        msg = f"missing lexical candidate table: {path}"
        raise DictionaryBuildError(msg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "columnar-v1":
        msg = f"unsupported lexical table format: {payload.get('format')!r}"
        raise DictionaryBuildError(msg)
    aligners = [str(name) for name in payload.get("aligner_names") or []]
    unit_ids = [str(unit_id) for unit_id in payload.get("unit_id_table") or []]
    block = payload["observed_wordforms"]
    fields = list(block["fields"])
    links: list[ObservedLink] = []
    for raw in block["rows"]:
        row = dict(zip(fields, raw, strict=True))
        links.append(
            ObservedLink(
                coptic=str(row["coptic_wordform"]),
                greek=str(row["greek_wordform"]),
                observation_count=int(row["observation_count"]),
                count_eflomal=int(row.get("count_eflomal") or 0),
                count_fast_align=int(row.get("count_fast_align") or 0),
            )
        )
    upstream = payload.get("provenance") or {}
    provenance: dict[str, object] = {
        "source_table_id": payload.get("id"),
        "pilot_corpus_id": upstream.get("pilot_corpus_id"),
        "split_id": upstream.get("split_id"),
        "bitext_checksum_sha256": upstream.get("bitext_checksum_sha256"),
        "coptic_segmentation": upstream.get("coptic_segmentation"),
        "greek_tokenization": upstream.get("greek_tokenization"),
        "greek_source_id": upstream.get("greek_source_id"),
        "sahidic_source_id": upstream.get("sahidic_source_id"),
    }
    return links, unit_ids, aligners, provenance


def _token_row(item: TokenOut) -> dict[str, object]:
    posterior = None if item.posterior is None else round(item.posterior, 6)
    return {
        "c": item.coptic,
        "g": item.greek,
        "ch": item.channel,
        "p": posterior,
        "n": item.observation_count,
        "h": item.span_head,
    }


def _sufficient(summary: dict[str, Any]) -> tuple[bool, str]:
    support = int(summary["loanword_tokens"])
    rate = summary["loanword_stem_match_rate"]
    recall = summary["content_type_recall"]
    precision = summary["content_emission_precision"]
    if support < LOANWORD_MIN_SUPPORT:
        return False, "loanword overlap support is below the pre-set floor"
    if not isinstance(rate, float) or rate < LOANWORD_SUFFICIENT_RATE:
        return False, "loanword stem match is below the pre-set iteration bar"
    if not isinstance(recall, float) or recall < CONTENT_RECALL_BAR:
        return False, "surviving content-type recall is below the pre-set bar"
    if not isinstance(precision, float) or precision < CONTENT_PRECISION_BAR:
        return False, "content emission precision is below the pre-set bar"
    return True, "pre-set loanword, recall, and precision bars met"


def _next_step(sufficient: bool) -> str:
    if sufficient:
        return (
            "Hold this version. Do not retune it on fragment Greek. "
            "Further gains need a new version and more parallel text."
        )
    return (
        "Not sufficient. Next iteration: widen licensed Coptic–Koine parallels "
        "(full Horner × Vaticanus B*, and Sahidic OT × Swete if verse-aligned) "
        "and re-fit both aligners. Do not add handwritten glosses and do not "
        "train on Thomas Greek."
    )


def _write_outputs(
    dest: Path,
    dictionary: ParallelDictionary,
    units: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    dictionary_payload = {
        "id": dictionary.version,
        "snap_enabled": dictionary.snap_enabled,
        "provenance": dictionary.provenance,
        "entry_count": len(dictionary.emitted_entries()),
        "entries": dictionary.emitted_entries(),
        "greek_type_count": len(dictionary.greek_types),
        "greek_types": [
            [greek, count] for greek, count in sorted(dictionary.greek_types.items())
        ],
    }
    reconstruction = {
        "id": f"{dictionary.version}-thomas",
        "coptic_source": "dilley-thomas-v6.2.0",
        "dictionary_id": dictionary.version,
        "snap_enabled": dictionary.snap_enabled,
        "uses_thomas_greek_as_training": False,
        "units": units,
        "notes": (
            "Generated Koine for Dilley Coptic. Abstentions have g=null. "
            "Fragment Greek was not an input."
        ),
    }
    dictionary_path = dest / "dictionary.json"
    _dump(dictionary_path, dictionary_payload)
    if dictionary_path.stat().st_size > 950_000:
        dictionary_payload["greek_types"] = []
        dictionary_payload["greek_types_omitted"] = (
            "Omitted so the artifact stays under the repository size hook. "
            "greek_type_count is retained. Snap was decided before this write."
        )
        _dump(dictionary_path, dictionary_payload)
    _dump(dest / "thomas-reconstruction.json", reconstruction)
    # Metrics include an evaluation audit. Do not feed this file back into training.
    _dump(dest / "metrics.json", metrics)


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
