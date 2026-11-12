# SPDX-License-Identifier: Apache-2.0
"""JAK-38: statistical constructions, verified loanwords, ablation separation."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.integrations.aligners.types import (
    AlignerName,
    AlignmentLinks,
    BitextSentence,
    DirectionalAlignment,
)
from zetodos_thomas.mappings.ablation import build_ablation_packages
from zetodos_thomas.mappings.loanwords import verified_loanword_table
from zetodos_thomas.mappings.phrases import build_statistical_construction_table
from zetodos_thomas.mappings.validate import (
    validate_ablation_package,
    validate_mappings_dir,
    validate_statistical_construction_table,
    validate_verified_loanword_table,
)
from zetodos_thomas.schemas.mappings import (
    AblationCondition,
    ConstructionFamily,
    EvidenceClass,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = REPO_ROOT / "data" / "sources" / "mappings"


def _sentence(
    unit_id: str,
    src: tuple[str, ...],
    trg: tuple[str, ...],
    *,
    role: str = "train",
) -> BitextSentence:
    return BitextSentence(
        unit_id=unit_id,
        source_tokens=src,
        target_tokens=trg,
        source_token_ids=tuple(f"v2:c/{unit_id}/t/{i}" for i in range(len(src))),
        target_token_ids=tuple(f"v1:g/{unit_id}/t/{i}" for i in range(len(trg))),
        split_role=role,
        source_parent_v1_ids=tuple(f"v1:c/{unit_id}/t/{i}" for i in range(len(src))),
    )


def _links(
    name: AlignerName,
    sentences: list[BitextSentence],
    sym_pairs: dict[str, tuple[tuple[int, int], ...]],
) -> AlignmentLinks:
    forward = [
        DirectionalAlignment(unit_id=s.unit_id, pairs=sym_pairs.get(s.unit_id, ()))
        for s in sentences
    ]
    return AlignmentLinks(
        aligner=name,
        sentences=sentences,
        forward=forward,
        reverse=list(forward),
        symmetrized=list(forward),
        notes=["fixture"],
    )


def test_verbal_false_morphology_verses_not_promoted() -> None:
    """Matt 5:40/5:48 and John 1:23 must not become reusable verbal mappings."""
    from zetodos_thomas.mappings.phrases import VERBAL_FALSE_POSITIVE_UNITS

    sents = [
        _sentence(
            "matthew.5.40",
            ("ⲛⲧⲉ", "ⲕϣⲧⲏⲛ"),
            ("σου", "λαβειν"),
        ),
        _sentence(
            "matthew.5.48",
            ("ⲛⲧⲉ", "ⲗⲓⲟⲥ"),
            ("τελειοι",),
        ),
        _sentence(
            "john.1.23",
            ("ⲛⲧⲉ", "ϩⲓ"),
            ("νατε", "την"),
        ),
        # A second identical false hit still must not promote (min dual units=2
        # is necessary but insufficient without morphology guards).
        _sentence(
            "matthew.5.40b",
            ("ⲛⲧⲉ", "ⲕϣⲧⲏⲛ"),
            ("σου", "λαβειν"),
        ),
    ]
    # Use real unit ids for the blocked set; extra unit uses a non-blocked id
    # with the same peel pattern so dual-unit alone cannot promote it.
    sents[3] = _sentence("mark.4.6", ("ⲛⲧⲉ", "ⲕϣⲧⲏⲛ"), ("σου", "λαβειν"))
    pairs = {
        s.unit_id: ((0, 0), (1, 1) if len(s.target_tokens) > 1 else (0, 0))
        for s in sents
    }
    # Fix mark.4.6 pairs for 2-token greek... actually greek has 2 tokens for first
    pairs["matthew.5.40"] = ((0, 0), (1, 1))
    pairs["matthew.5.48"] = ((0, 0), (1, 0))
    pairs["john.1.23"] = ((0, 0), (1, 1))
    pairs["mark.4.6"] = ((0, 0), (1, 1))
    table = build_statistical_construction_table(
        {
            "eflomal": _links(AlignerName.EFLOMAL, sents, pairs),
            "fast_align": _links(AlignerName.FAST_ALIGN, sents, pairs),
        },
        min_dual_units=2,
    )
    verbals = [c for c in table.constructions if c.family is ConstructionFamily.VERBAL]
    assert not verbals
    for unit in VERBAL_FALSE_POSITIVE_UNITS:
        assert all(unit not in c.source_unit_ids for c in table.constructions)


def test_example_non_train_unit_rejected_by_validate() -> None:
    sents = [
        _sentence("john.1.1", ("ⲡ", "ϣⲁϫⲉ"), ("ο", "λογος")),
        _sentence("john.1.2", ("ⲡ", "ϣⲁϫⲉ"), ("ο", "λογος")),
    ]
    pairs = {
        "john.1.1": ((0, 0), (1, 1)),
        "john.1.2": ((0, 0), (1, 1)),
    }
    table = build_statistical_construction_table(
        {
            "eflomal": _links(AlignerName.EFLOMAL, sents, pairs),
            "fast_align": _links(AlignerName.FAST_ALIGN, sents, pairs),
        },
        min_dual_units=2,
    )
    assert table.constructions
    # Mutate an example onto holdout membership.
    row = table.constructions[0]
    row.examples[0].unit_id = "matthew.13.31"
    row.examples[0].coptic_span.unit_id = "matthew.13.31"
    report = validate_statistical_construction_table(
        table, train_unit_ids={"john.1.1", "john.1.2"}
    )
    assert not report.ok
    assert any(i.code == "example_non_train_unit" for i in report.issues)


def test_loanword_tei_mismatch_fails_validation() -> None:
    loan = verified_loanword_table(repo_root=REPO_ROOT)
    entry = loan.entries[0]
    entry.ddglc_greek_lemma_id = "999999"
    report = validate_verified_loanword_table(
        loan, repo_root=REPO_ROOT, train_unit_ids=set()
    )
    assert not report.ok
    assert any(i.code == "tei_field_mismatch" for i in report.issues)


def test_illustrative_holdout_span_rejected() -> None:
    loan = verified_loanword_table(repo_root=REPO_ROOT)
    entry = next(e for e in loan.entries if e.loanword_id == "verified-euaggelion")
    entry.illustrative_nt_span_ids.append(
        "v1:ntvmr-vaticanus-b/matthew.13.31/s/parabolen"
    )
    from zetodos_thomas.align.splits import SplitRole, load_canonical_splits
    from zetodos_thomas.ingest.paths import (
        resolve_nt_split_path,
        resolve_omission_fixtures_path,
    )

    splits = load_canonical_splits(
        resolve_nt_split_path(REPO_ROOT),
        resolve_omission_fixtures_path(REPO_ROOT),
    )
    train_ids = set(splits.unit_ids_for(SplitRole.TRAIN))
    report = validate_verified_loanword_table(
        loan, repo_root=REPO_ROOT, train_unit_ids=train_ids
    )
    assert not report.ok
    assert any(i.code == "illustrative_non_train_unit" for i in report.issues)


def test_statistical_vs_verified_are_separate_classes() -> None:
    sents = [
        _sentence("john.1.1", ("ⲡ", "ϣⲁϫⲉ", "ⲁⲛⲟⲕ"), ("ο", "λογος", "εγω")),
        _sentence("john.1.2", ("ⲡ", "ϣⲁϫⲉ"), ("ο", "λογος")),
        _sentence("john.1.3", ("ⲁⲛ",), ("ουχ",)),
        _sentence("john.1.4", ("ⲁⲛ",), ("ουχ",)),
    ]
    pairs: dict[str, tuple[tuple[int, int], ...]] = {
        "john.1.1": ((0, 0), (1, 1), (2, 2)),
        "john.1.2": ((0, 0), (1, 1)),
        "john.1.3": ((0, 0),),
        "john.1.4": ((0, 0),),
    }
    table = build_statistical_construction_table(
        {
            "eflomal": _links(AlignerName.EFLOMAL, sents, pairs),
            "fast_align": _links(AlignerName.FAST_ALIGN, sents, pairs),
        },
        provenance={"owned_by": "JAK-38-test"},
        min_dual_units=2,
    )
    assert table.evidence_class is EvidenceClass.STATISTICAL
    assert table.constructions
    assert all(
        c.evidence_class is EvidenceClass.STATISTICAL for c in table.constructions
    )
    families = {c.family for c in table.constructions}
    assert (
        ConstructionFamily.ARTICLE in families or ConstructionFamily.PRONOUN in families
    )

    loan = verified_loanword_table(repo_root=REPO_ROOT)
    assert loan.evidence_class is EvidenceClass.INDEPENDENTLY_VERIFIED
    assert all(
        e.evidence_class is EvidenceClass.INDEPENDENTLY_VERIFIED for e in loan.entries
    )
    assert loan.license == "CC-BY-SA-4.0"
    assert any(e.coptic_lemma == "ⲙⲁⲕⲁⲣⲓⲟⲥ" for e in loan.entries)
    # No Thomas / P.Oxy. Greek in illustrative spans.
    for e in loan.entries:
        assert e.illustrative_spans_are_non_training is True
        assert e.spot_check.tei_entry_xml_id == e.tla_entry_id
        for sid in e.illustrative_nt_span_ids:
            assert "thomas." not in sid.lower()
            assert "poxy" not in sid.lower()
            assert "luke.6.20" not in sid
            assert "matthew.13.31" not in sid


def test_rejects_thomas_or_poxy_units_in_statistical_table() -> None:
    sents = [
        _sentence("thomas.7", ("ⲁⲛ",), ("ουχ",)),
        _sentence("john.1.1", ("ⲁⲛ",), ("ουχ",)),
    ]
    pairs: dict[str, tuple[tuple[int, int], ...]] = {
        "thomas.7": ((0, 0),),
        "john.1.1": ((0, 0),),
    }
    try:
        build_statistical_construction_table(
            {
                "eflomal": _links(AlignerName.EFLOMAL, sents, pairs),
                "fast_align": _links(AlignerName.FAST_ALIGN, sents, pairs),
            }
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "forbidden" in str(exc).lower() or "thomas" in str(exc).lower()
    assert raised


def test_ablation_arms_wire_loanwords_correctly() -> None:
    packages = {p.condition: p for p in build_ablation_packages()}
    a = packages[AblationCondition.NT_ALIGNER_ONLY]
    b = packages[AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS]
    assert a.includes["verified_loanwords"] is None
    assert b.includes["verified_loanwords"]
    assert a.phase2_active_aligners == ["eflomal", "fast_align"]
    assert b.adopted_aligner is None
    assert validate_ablation_package(a).ok
    assert validate_ablation_package(b).ok


def test_committed_artifacts_validate() -> None:
    assert (MAPPINGS / "statistical-phrase-constructions-v1.json").is_file()
    assert (MAPPINGS / "verified-loanwords-ccl-ddglc-v1.json").is_file()
    assert (MAPPINGS / "ablation-nt-aligner-only-v1.json").is_file()
    assert (MAPPINGS / "ablation-nt-aligner-plus-loanwords-v1.json").is_file()

    from zetodos_thomas.align.splits import SplitRole, load_canonical_splits
    from zetodos_thomas.ingest.paths import (
        resolve_nt_split_path,
        resolve_omission_fixtures_path,
    )
    from zetodos_thomas.mappings.loanwords import load_verified_loanword_table
    from zetodos_thomas.mappings.phrases import load_statistical_construction_table

    phrases = load_statistical_construction_table(
        MAPPINGS / "statistical-phrase-constructions-v1.json"
    )
    loan = load_verified_loanword_table(
        MAPPINGS / "verified-loanwords-ccl-ddglc-v1.json"
    )
    splits = load_canonical_splits(
        resolve_nt_split_path(REPO_ROOT),
        resolve_omission_fixtures_path(REPO_ROOT),
    )
    train_ids = set(splits.unit_ids_for(SplitRole.TRAIN))
    assert validate_statistical_construction_table(phrases, train_unit_ids=train_ids).ok
    assert validate_verified_loanword_table(
        loan, repo_root=REPO_ROOT, train_unit_ids=train_ids
    ).ok
    report = validate_mappings_dir(REPO_ROOT, train_unit_ids=train_ids)
    assert report.ok, report.as_dict()

    # Span IDs present for decoder (JAK-39) consumption.
    assert any(
        ex.coptic_span.span_id for c in phrases.constructions for ex in c.examples
    )
    assert phrases.summary["phase2_active_aligners"] == ["eflomal", "fast_align"]
    assert phrases.summary["adopted_aligner"] is None
    # Known false verbal one-offs must be absent from the committed table.
    blocked = {"matthew.5.40", "matthew.5.48", "john.1.23"}
    for c in phrases.constructions:
        if c.family is ConstructionFamily.VERBAL:
            assert not (blocked & set(c.source_unit_ids))
            assert len(c.source_unit_ids) >= 2


def test_cli_mappings_validate() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["mappings", "validate", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.split("OK:")[0])
    assert payload["ok"] is True


def test_manifest_ccl_present() -> None:
    manifest = REPO_ROOT / "data" / "manifests" / "ccl-ddglc-tei.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["id"] == "ccl-ddglc-tei"
    assert data["spdx_or_custom"] == "CC-BY-SA-4.0"
    assert (
        "ace84fce11b76386a6adb9888b2cb947056954fc9fe9c04aa0676b1f595fd43d"
        in data["checksum"]
    )
