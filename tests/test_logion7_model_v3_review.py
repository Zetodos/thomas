# SPDX-License-Identifier: Apache-2.0
"""The model-v3 Logion 7 review page copies the committed output."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs" / "logion7-model-v3-review.md"
GENERATED = ROOT / "data" / "generated" / "thomas-generated-v3.json"
COPTIC = ROOT / "data" / "sources" / "thomas" / "dilley-thomas-v6.2.0.json"
EVAL = ROOT / "data" / "generated" / "eval-report-v3.json"
BENCHMARK = ROOT / "docs" / "benchmark-report-v3.md"

FILE_SHA256 = "6c358d57d8d18a7613a616bb69c29d0ef1f8f39592e333bc31e665e9646eb54c"
UNIT_SHA256 = "403d2489e5510c097c4dbd164f7cb4c0c76671b42f1172b0b306c27dedf68e88"
COPTIC_SHA256 = "34b24f4fcec8daa3540fca451d5a8a6d8d5fa02b4c4b803537c8b08b2e9fc501"
OUTPUT_COMMIT = "c28c656c7c890a36ce629067344d0cb0dfa590d2"

_MODEL_DEFAULT = re.compile(r'model:\s*str\s*=\s*typer\.Option\(\s*"v1"')


def _unit_slice(text: str, unit_id: str) -> str:
    """Return the raw JSON object for ``unit_id`` as stored in ``text``."""
    key = f'"unit_id":"{unit_id}"'
    start = text.rfind("{", 0, text.index(key))
    depth = 0
    in_str = False
    esc = False
    for index, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    msg = f"unclosed JSON object for {unit_id}"
    raise AssertionError(msg)


def _token_seq(tok: dict[str, object]) -> int:
    seq = tok["seq"]
    assert isinstance(seq, int)
    return seq


def _greek_cell(tokens: list[dict[str, object]]) -> str:
    if not tokens:
        return "abstain"
    ordered = sorted(tokens, key=_token_seq)
    return "; ".join(f"{tok['surface']} ({tok['seq']})" for tok in ordered)


def _morph_text(morph: dict[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in morph.items())


def test_review_page_copies_committed_logion_7() -> None:
    page = REVIEW.read_text(encoding="utf-8")
    gen_raw = GENERATED.read_bytes()
    cop_raw = COPTIC.read_bytes()
    assert hashlib.sha256(gen_raw).hexdigest() == FILE_SHA256
    assert hashlib.sha256(cop_raw).hexdigest() == COPTIC_SHA256
    unit_json = _unit_slice(gen_raw.decode("utf-8"), "thomas.7")
    assert hashlib.sha256(unit_json.encode("utf-8")).hexdigest() == UNIT_SHA256
    for digest in (FILE_SHA256, UNIT_SHA256, COPTIC_SHA256, OUTPUT_COMMIT):
        assert digest in page

    generated = json.loads(gen_raw)
    assert generated["model_id"] == "model-v3"
    assert generated["id"] == "thomas-generated-v3"
    unit = json.loads(unit_json)
    assert unit["unit_id"] == "thomas.7"
    assert unit["coptic_source_id"] == "dilley-thomas"
    assert "coptic_text" not in unit
    assert "clause_alternatives" not in unit

    corpus = json.loads(cop_raw)
    coptic = next(item for item in corpus["units"] if item["unit_id"] == "thomas.7")
    morphs = coptic["morphemes"]
    assert unit["greek_text"] in page
    assert coptic["original_text"] in page
    assert coptic["normalized_text"] in page
    assert unit["greek_text"] == " ".join(tok["surface"] for tok in unit["tokens"])
    joined = " ".join(morph["norm"] for morph in morphs)
    assert coptic["normalized_text"] == joined
    assert "model-v3 stores one Greek sentence" in page
    assert "explicit abstention" in page

    by_seq: dict[int, list[dict[str, object]]] = {i: [] for i in range(len(morphs))}
    for tok in unit["tokens"]:
        for seq in tok["coptic_seqs"]:
            by_seq[int(seq)].append(tok)

    abstain = 0
    for index, morph in enumerate(morphs):
        cell = _greek_cell(by_seq[index])
        if cell == "abstain":
            abstain += 1
        row = (
            f"| {index} | {morph['norm']} | {morph['lemma']} | "
            f"{morph['pos']} | {cell} |"
        )
        assert row in page, row
    assert abstain == 23
    assert page.count("| abstain |") == 23
    assert len(unit["tokens"]) == 27

    for tok in unit["tokens"]:
        seqs = ", ".join(str(seq) for seq in tok["coptic_seqs"])
        norms = " ".join(morphs[seq]["norm"] for seq in tok["coptic_seqs"])
        row = (
            f"| {tok['seq']} | {tok['surface']} | {tok['lemma']} | "
            f"{_morph_text(tok['morph'])} | {seqs} | {norms} | "
            f"{tok['provenance']} | {json.dumps(tok['confidence'])} |"
        )
        assert row in page, row
        assert "alternatives" not in tok

    report = json.loads(EVAL.read_text(encoding="utf-8"))
    span = next(
        item for item in report["spans"] if item["span_id"] == "thomas.7.poxy654.a"
    )
    for key in ("predicted_greek", "attested_surviving", "attested_editorial"):
        value = span[key]
        assert value
        assert value not in page


def test_cli_default_stays_model_v1_and_model_v2_remains() -> None:
    for rel in (
        "python/zetodos_thomas/cli/reconstruct_cmd.py",
        "python/zetodos_thomas/cli/eval_cmd.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert _MODEL_DEFAULT.search(text), rel
    assert (ROOT / "data/mappings/model-v2/seq2seq-config.json").is_file()
    assert (ROOT / "data/generated/thomas-generated-v2.json").is_file()
    assert (ROOT / "data/mappings/model-v3/seq2seq-config.json").is_file()


def test_benchmark_report_points_at_the_review() -> None:
    text = BENCHMARK.read_text(encoding="utf-8")
    assert "logion7-model-v3-review.md" in text
    assert "**0.6334**" in text
    assert "`thomas.7.poxy654.a`" in text
