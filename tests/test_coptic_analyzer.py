# SPDX-License-Identifier: Apache-2.0
"""Intrinsic evaluation of CopticAnalyzer on held-out sahidic.ot Ruth & Jonah."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from zetodos_thomas.ingest.ot_swete import load_sahidic_ot_chapters
from zetodos_thomas.morphology.coptic_analyzer import CopticAnalyzer
from zetodos_thomas.schemas.linguistic import CopticMorpheme

REPO_ROOT = Path(__file__).resolve().parents[1]
OT_ZIP = REPO_ROOT / "data" / "downloads" / "sahidic-ot" / "sahidic.ot_TT.zip"
CCL_XML = (
    REPO_ROOT
    / "data"
    / "downloads"
    / "ccl-ddglc"
    / "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
)


def _group_norm_seq(morphemes: list[CopticMorpheme]) -> list[str]:
    return [m.norm for m in morphemes]


def _seg_boundaries(norms: list[str]) -> set[tuple[int, int]]:
    bounds: set[tuple[int, int]] = set()
    offset = 0
    for n in norms:
        bounds.add((offset, offset + len(n)))
        offset += len(n)
    return bounds


def test_coptic_analyzer_heldout_ruth_and_jonah() -> None:
    if not OT_ZIP.is_file():
        pytest.skip("sahidic.ot_TT.zip not downloaded in this test environment")

    analyzer = CopticAnalyzer.train(
        ot_zip=OT_ZIP,
        ccl_xml=CCL_XML if CCL_XML.is_file() else None,
        exclude_books={"Ruth", "Jonah"},
    )
    ot = load_sahidic_ot_chapters(OT_ZIP)

    tp_seg = 0
    pred_seg = 0
    gold_seg = 0
    lemma_ok = 0
    pos_ok = 0
    uas_ok = 0
    uas_total = 0
    aligned_spans = 0
    gold_total = 0

    for book in ("Ruth", "Jonah"):
        book_groups = [
            g
            for (_ch, _vs), u in sorted(ot[book].items())
            for g in u.original_text.split()
        ]
        bare_stems = analyzer._collect_bare_doc_stems(book_groups)
        for (_ch, _vs), gold_unit in sorted(ot[book].items()):
            raw_groups = gold_unit.original_text.split()
            pred_morphs = analyzer.analyze_groups(
                raw_groups,
                unit_id=gold_unit.unit_id,
                source_id=gold_unit.source_id,
                doc_bare_stems=bare_stems,
            )

            gold_by_g: dict[int, list[CopticMorpheme]] = defaultdict(list)
            pred_by_g: dict[int, list[CopticMorpheme]] = defaultdict(list)
            for m in gold_unit.morphemes:
                gold_by_g[m.group_seq].append(m)
            for m in pred_morphs:
                if m.pos != "PUNCT":
                    pred_by_g[m.group_seq].append(m)

            gold_seq_span: dict[int, tuple[int, int, int]] = {}
            gold_span_gm: dict[tuple[int, int, int], CopticMorpheme] = {}
            pred_span_seq: dict[tuple[int, int, int], int] = {}
            aligned_pairs: list[tuple[CopticMorpheme, CopticMorpheme]] = []

            for g_idx in range(len(raw_groups)):
                g_ms = gold_by_g.get(g_idx, [])
                p_ms = pred_by_g.get(g_idx, [])
                g_norms = _group_norm_seq(g_ms)
                p_norms = _group_norm_seq(p_ms)
                g_bounds = _seg_boundaries(g_norms)
                p_bounds = _seg_boundaries(p_norms)
                tp_seg += len(g_bounds & p_bounds)
                gold_seg += len(g_bounds)
                pred_seg += len(p_bounds)
                gold_total += len(g_ms)

                # Align exact-span morphemes within bound group for lemma/POS/UAS
                g_offset = 0
                for gm in g_ms:
                    gn = gm.norm
                    span = (g_idx, g_offset, g_offset + len(gn))
                    gold_seq_span[gm.seq] = span
                    gold_span_gm[span] = gm
                    g_offset += len(gn)
                p_offset = 0
                for pm in p_ms:
                    pn = pm.norm
                    span = (g_idx, p_offset, p_offset + len(pn))
                    pred_span_seq[span] = pm.seq
                    p_offset += len(pn)
                    if span in gold_span_gm:
                        gm = gold_span_gm[span]
                        aligned_spans += 1
                        if pm.lemma == gm.lemma:
                            lemma_ok += 1
                        if pm.pos == gm.pos:
                            pos_ok += 1
                        aligned_pairs.append((pm, gm))

            for pm, gm in aligned_pairs:
                if gm.head_seq is None:
                    uas_total += 1
                    if pm.head_seq is None:
                        uas_ok += 1
                elif (
                    gm.head_seq in gold_seq_span
                    and gold_seq_span[gm.head_seq] in pred_span_seq
                ):
                    exp_pred_head = pred_span_seq[gold_seq_span[gm.head_seq]]
                    uas_total += 1
                    if pm.head_seq == exp_pred_head:
                        uas_ok += 1

    prec = tp_seg / max(pred_seg, 1)
    rec = tp_seg / max(gold_seg, 1)
    seg_f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    lemma_acc_all = lemma_ok / max(gold_total, 1)
    pos_acc_all = pos_ok / max(gold_total, 1)
    uas = uas_ok / max(uas_total, 1)

    print(
        f"\n[CopticAnalyzer Ruth+Jonah held-out] "
        f"seg_P={prec:.4f} seg_R={rec:.4f} seg_F1={seg_f1:.4f} "
        f"lemma_acc={lemma_acc_all:.4f} pos_acc={pos_acc_all:.4f} "
        f"UAS={uas:.4f} ({uas_ok}/{uas_total}) "
        f"(aligned={aligned_spans}/{gold_total})"
    )
    assert seg_f1 >= 0.94, f"Segmentation F1 {seg_f1:.4f} < 0.94"
    assert lemma_acc_all >= 0.90, f"Lemma accuracy {lemma_acc_all:.4f} < 0.90"
    assert uas >= 0.80, f"UD UAS {uas:.4f} < 0.80"


def test_coptic_analyzer_json_roundtrip(tmp_path: Path) -> None:
    if not OT_ZIP.is_file():
        pytest.skip("sahidic.ot_TT.zip not downloaded in this test environment")
    analyzer = CopticAnalyzer.train(ot_zip=OT_ZIP, exclude_books={"Ruth", "Jonah"})
    out = tmp_path / "coptic_analyzer.json"
    analyzer.save_json(out)
    reloaded = CopticAnalyzer.load_json(out)
    m1 = analyzer.analyze_groups(["ⲁϥⲃⲱⲕ", "ⲉϩⲟⲩⲛ", "ⲉⲧⲡⲟⲗⲓⲥ."])
    m2 = reloaded.analyze_groups(["ⲁϥⲃⲱⲕ", "ⲉϩⲟⲩⲛ", "ⲉⲧⲡⲟⲗⲓⲥ."])
    assert [(m.norm, m.lemma, m.pos, m.head_seq, m.dep_rel) for m in m1] == [
        (m.norm, m.lemma, m.pos, m.head_seq, m.dep_rel) for m in m2
    ]
    assert any(m.dep_rel == "root" and m.head_seq is None for m in m1)
