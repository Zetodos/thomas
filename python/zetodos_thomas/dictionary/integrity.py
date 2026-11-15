# SPDX-License-Identifier: Apache-2.0
"""Digests and frozen-model IO for the parallel dictionary.

These helpers authenticate the bytes that were aligned and the prediction
that was written before any fragment Greek is read. They do not choose glosses.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zetodos_thomas.dictionary.build import (
    DictionaryBuildError,
    ParallelDictionary,
    _Acc,
)
from zetodos_thomas.integrations.aligners.types import AlignerConfig, BitextSentence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_aligner_input(sentences: list[BitextSentence]) -> str:
    """Hash the exact sentence list the aligners receive, in that order."""
    digest = hashlib.sha256()
    for index, sentence in enumerate(sentences):
        source = " ".join(sentence.source_tokens)
        target = " ".join(sentence.target_tokens)
        line = f"{index}\t{sentence.unit_id}\t{source}\t{target}\n"
        digest.update(line.encode())
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return sha256_file(path)


def hash_tei_dirs(directories: list[Path]) -> str:
    """Stable digest of every file under the directories, skipping git metadata."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            files.append(path)
    for path in sorted(files, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def observed_git_commit(repo: Path) -> str:
    """HEAD of a local checkout. Empty when the directory is not a git repo."""
    if not (repo / ".git").exists():
        return ""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def aligner_config_record() -> dict[str, object]:
    config = AlignerConfig()
    return {
        "seed": config.seed,
        "eflomal_model": config.eflomal_model,
        "eflomal_n_samplers": config.eflomal_n_samplers,
        "eflomal_null_prior": config.eflomal_null_prior,
        "symmetrization": config.symmetrization,
        "note": (
            "eflomal's Gibbs sampler reads OS entropy. The seed is recorded "
            "and does not by itself reproduce links."
        ),
    }


def verify_named_files(record: dict[str, str], files: dict[str, Path]) -> None:
    """Re-hash ``files`` and refuse when any digest disagrees with ``record``."""
    problems: list[str] = []
    for key, path in files.items():
        actual = hash_file(path)
        expected = str(record.get(key, ""))
        if actual != expected:
            problems.append(key)
    if problems:
        msg = "training source bytes do not match the recorded digest: " + ", ".join(
            problems
        )
        raise DictionaryBuildError(msg)


def verify_aligner_input(
    record: Mapping[str, object],
    sentences: list[BitextSentence],
) -> None:
    expected = str(record.get("aligner_input_sha256", ""))
    actual = hash_aligner_input(sentences)
    if actual != expected:
        msg = "aligner input bitext does not match the recorded digest"
        raise DictionaryBuildError(msg)


def verify_prediction_pin(directory: Path, pin: dict[str, str]) -> None:
    """Re-read the frozen files and refuse if they changed after the pin."""
    problems: list[str] = []
    for name in ("dictionary.json", "thomas-reconstruction.json"):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} missing")
            continue
        actual = sha256_file(path)
        expected = pin.get(name, "")
        if actual != expected:
            problems.append(name)
    if problems:
        msg = "prediction pin does not match the frozen files: " + ", ".join(problems)
        raise DictionaryBuildError(msg)


def _acc_row(acc: _Acc) -> dict[str, int]:
    return {
        "observation_count": acc.observation_count,
        "count_eflomal": acc.count_eflomal,
        "count_fast_align": acc.count_fast_align,
        "consensus_count": acc.consensus_count,
    }


def _acc_from_row(row: dict[str, Any]) -> _Acc:
    acc = _Acc()
    acc.observation_count = int(row.get("observation_count") or 0)
    acc.count_eflomal = int(row.get("count_eflomal") or 0)
    acc.count_fast_align = int(row.get("count_fast_align") or 0)
    acc.consensus_count = int(row.get("consensus_count") or 0)
    return acc


def export_inference_state(dictionary: ParallelDictionary) -> dict[str, Any]:
    """Full state needed to apply the dictionary again without re-aligning."""
    phrases = [
        {
            "coptic": list(coptic),
            "greek": list(greek),
            "count": count,
            "posterior": posterior,
        }
        for coptic, (greek, count, posterior) in sorted(dictionary.phrases.items())
    ]
    traces = [
        {"tokens": list(coptic), "rows": [dict(row) for row in rows]}
        for coptic, rows in sorted(dictionary.replay_traces.items())
    ]
    return {
        "artifact_role": "complete_frozen_model",
        "version": dictionary.version,
        "snap_enabled": dictionary.snap_enabled,
        "use_context": dictionary.use_context,
        "use_suffix": dictionary.use_suffix,
        "use_phrases": dictionary.use_phrases,
        "use_paradigm": dictionary.use_paradigm,
        "use_loan_spelling": dictionary.use_loan_spelling,
        "provenance": dictionary.provenance,
        "direct": {
            coptic: {greek: _acc_row(acc) for greek, acc in forms.items()}
            for coptic, forms in dictionary.direct.items()
        },
        "stems": {
            coptic: {greek: _acc_row(acc) for greek, acc in forms.items()}
            for coptic, forms in dictionary.stems.items()
        },
        "greek_types": dictionary.greek_types,
        "keys_with_pairs": sorted(dictionary.keys_with_pairs),
        "context_counts": dictionary.context_counts,
        "suffix_given_prev": dictionary.suffix_given_prev,
        "phrases": phrases,
        "replay_traces": traces,
    }


def load_inference_state(payload: dict[str, Any]) -> ParallelDictionary:
    """Rebuild a dictionary from ``export_inference_state``."""
    if payload.get("artifact_role") != "complete_frozen_model":
        msg = "refusing to load a file that is not a complete frozen model"
        raise DictionaryBuildError(msg)
    direct = {
        coptic: {greek: _acc_from_row(row) for greek, row in forms.items()}
        for coptic, forms in (payload.get("direct") or {}).items()
    }
    stems = {
        coptic: {greek: _acc_from_row(row) for greek, row in forms.items()}
        for coptic, forms in (payload.get("stems") or {}).items()
    }
    phrases: dict[tuple[str, ...], tuple[tuple[str, ...], int, float]] = {}
    for row in payload.get("phrases") or []:
        coptic = tuple(str(token) for token in row["coptic"])
        greek = tuple(str(token) for token in row["greek"])
        phrases[coptic] = (greek, int(row["count"]), float(row["posterior"]))
    traces: dict[tuple[str, ...], tuple[dict[str, object], ...]] = {}
    for row in payload.get("replay_traces") or []:
        coptic = tuple(str(token) for token in row["tokens"])
        traces[coptic] = tuple(dict(item) for item in row["rows"])
    return ParallelDictionary(
        version=str(payload.get("version") or ""),
        direct=direct,
        stems=stems,
        greek_types={
            str(k): int(v) for k, v in (payload.get("greek_types") or {}).items()
        },
        keys_with_pairs=set(payload.get("keys_with_pairs") or []),
        snap_enabled=bool(payload.get("snap_enabled")),
        provenance=dict(payload.get("provenance") or {}),
        context_counts=dict(payload.get("context_counts") or {}),
        use_context=bool(payload.get("use_context")),
        suffix_given_prev=dict(payload.get("suffix_given_prev") or {}),
        use_suffix=bool(payload.get("use_suffix")),
        phrases=phrases,
        use_phrases=bool(payload.get("use_phrases")),
        use_paradigm=bool(payload.get("use_paradigm")),
        use_loan_spelling=bool(payload.get("use_loan_spelling")),
        replay_traces=traces,
    )


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def prediction_pin(directory: Path) -> dict[str, str]:
    return {
        "dictionary.json": sha256_file(directory / "dictionary.json"),
        "thomas-reconstruction.json": sha256_file(
            directory / "thomas-reconstruction.json"
        ),
    }
