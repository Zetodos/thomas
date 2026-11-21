# SPDX-License-Identifier: Apache-2.0
"""Benchmark evaluation suite: 41 NT Gospel holdouts & lacuna-aware P.Oxy. 1/654/655.

Strictly evaluation-only: reads P.Oxy. fragments (`data/sources/fragments/`) and
clause alignments (`data/evidence/fragment-clause-map.json`), while model
training and decoding remain completely blind to fragment witnesses.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from zetodos_thomas.ingest.dilley_thomas import load_thomas_corpus
from zetodos_thomas.ingest.fragments import load_fragment_corpus
from zetodos_thomas.ingest.holdouts import load_nt_split
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt, pair_horner_morphgnt
from zetodos_thomas.ingest.paths import (
    resolve_fragments_corpus_path,
    resolve_nt_split_path,
    resolve_thomas_path,
)
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.morphology.coptic_analyzer import CopticAnalyzer
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.reconstruct.decoder import KoineBeamDecoder
from zetodos_thomas.schemas.evidence import FragmentClauseMap, FragmentClauseSpan
from zetodos_thomas.schemas.fragment import FragmentCorpus, FragmentWord
from zetodos_thomas.schemas.generated import GeneratedToken, GeneratedUnit
from zetodos_thomas.schemas.holdout import PassageRole
from zetodos_thomas.schemas.linguistic import CopticUnit, GreekMorphFeatures, GreekUnit


class DecoderProtocol(Protocol):
    """Minimal common decoder interface shared by model-v1 and model-v2."""

    ccl_lexicon: CclLexicon

    def decode_unit(self, unit: CopticUnit) -> GeneratedUnit: ...


FRAGMENT_CLAUSE_MAP_REL = Path("data/evidence/fragment-clause-map.json")
EVAL_REPORT_REL = Path("data/generated/eval-report-v1.json")
BENCHMARK_DOC_REL = Path("docs/benchmark-report.md")

CONTENT_POS = frozenset({"N-", "V-", "A-", "D-"})
_PUNCT_RE = re.compile(r"[\s·.,;;:!?()\[\]\"'«»—\-]+")

# Canonical Koine nomina sacra & scribal abbreviations (unaccented -> (lemma, pos))
_NOMINA_SACRA: dict[str, tuple[str, str]] = {
    "ις": ("Ἰησοῦς", "N-"),
    "ιης": ("Ἰησοῦς", "N-"),
    "ιν": ("Ἰησοῦς", "N-"),
    "ιυ": ("Ἰησοῦς", "N-"),
    "ιηυ": ("Ἰησοῦς", "N-"),
    "θς": ("θεός", "N-"),
    "θυ": ("θεός", "N-"),
    "θω": ("θεός", "N-"),
    "θν": ("θεός", "N-"),
    "πρ": ("πατήρ", "N-"),
    "πρς": ("πατήρ", "N-"),
    "πρα": ("πατήρ", "N-"),
    "πρι": ("πατήρ", "N-"),
    "πριδι": ("πατρίς", "N-"),
    "ανς": ("ἄνθρωπος", "N-"),
    "ανου": ("ἄνθρωπος", "N-"),
    "ανω": ("ἄνθρωπος", "N-"),
    "ανον": ("ἄνθρωπος", "N-"),
    "ανων": ("ἄνθρωπος", "N-"),
    "ανοις": ("ἄνθρωπος", "N-"),
    "πνα": ("πνεῦμα", "N-"),
    "πνς": ("πνεῦμα", "N-"),
    "πνι": ("πνεῦμα", "N-"),
    "κς": ("κύριος", "N-"),
    "κυ": ("κύριος", "N-"),
    "κω": ("κύριος", "N-"),
    "κν": ("κύριος", "N-"),
    "χς": ("Χριστός", "N-"),
    "χυ": ("Χριστός", "N-"),
    "χω": ("Χριστός", "N-"),
    "χν": ("Χριστός", "N-"),
}

# Koine inflected-form fallback for rare/unattested NT forms in P.Oxy.
_POXY_LEX_STEM_FALLBACK: dict[str, tuple[str, str]] = {
    "σαββατισητε": ("σαββατίζω", "V-"),
    "διαβλεψεις": ("διαβλέπω", "V-"),
    "θαμβηθησεται": ("θαμβέω", "V-"),
    "θαμβηθεις": ("θαμβέω", "V-"),
    "αναπαησεται": ("ἀναπαύω", "V-"),
    "προφθασαντες": ("προφθάνω", "V-"),
    "καταντησουσιν": ("καταντάω", "V-"),
    "εξεταζουσιν": ("ἐξετάζω", "V-"),
    "αναφαινεται": ("ἀναφαίνω", "V-"),
    "αποκεκρυμμενον": ("ἀποκρύπτω", "V-"),
    "αποκνησει": ("ἀποκνέω", "V-"),
    "ενοικει": ("ἐνοικέω", "V-"),
    "κρεισσονες": ("κρείσσων", "A-"),
    "ξαινει": ("ξαίνω", "V-"),
    "νηθει": ("νήθω", "V-"),
    "ενδειτε": ("ἐνδέω", "V-"),
    "εμφανης": ("ἐμφανής", "A-"),
    "εκδυσησθε": ("ἐκδύω", "V-"),
    "αισχυνθητε": ("αἰσχύνω", "V-"),
    "ωτιον": ("ὠτίον", "N-"),
    "κλειδας": ("κλείς", "N-"),
    "ακεραιοι": ("ἀκέραιος", "A-"),
    "φρονιμοι": ("φρόνιμος", "A-"),
    "αθεοι": ("ἄθεος", "A-"),
}


def strip_greek_diacritics(text: str) -> str:
    """Strip polytonic combining marks & punctuation and normalize final sigma."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    cleaned = _PUNCT_RE.sub("", stripped).lower().replace("ς", "σ")
    return unicodedata.normalize("NFC", cleaned)


def koine_itacism_fold(text: str) -> str:
    """Normalize Koine Greek itacism mergers (ei=e=i, ai=e, oi=y, o=omega)."""
    s = strip_greek_diacritics(text)
    s = s.replace("ει", "ι").replace("αι", "ε").replace("οι", "υ").replace("υι", "υ")
    s = s.replace("η", "ι").replace("ω", "ο")
    return s


def normalize_polytonic_surface(text: str) -> str:
    """Normalize NFC lowercase Greek word preserving accents for surface F1."""
    clean = _PUNCT_RE.sub("", unicodedata.normalize("NFC", text)).lower()
    if clean.endswith("σ") and len(clean) > 1:
        clean = clean[:-1] + "ς"
    return clean


class NtHoldoutMetrics(BaseModel):
    """Metrics evaluated on intact 41-verse NT Gospel holdout split."""

    unit_count: int
    coverage: float
    content_lemma_precision: float
    content_lemma_recall: float
    content_lemma_f1: float
    all_token_lemma_f1: float
    inflected_surface_f1: float
    morph_accuracy_on_matched_lemmas: float
    chrf_pp: float
    bleu_4: float


class FragStratumMetrics(BaseModel):
    """Lacuna-aware P.Oxy. fragment metrics for one divergence stratum."""

    span_count: int
    unit_count: int
    surviving_word_count: int
    surviving_char_count: int
    surviving_char_f1_exact: float
    surviving_char_f1_itacism: float
    surviving_char_3gram_recall_itacism: float
    content_lemma_precision: float
    content_lemma_recall: float
    content_lemma_f1: float
    top3_content_lemma_recall: float
    loanword_exact_match_rate: float
    loanword_total: int


class SpanEvalSummary(BaseModel):
    """Detailed per-span comparison between attested P.Oxy. and generated Category C."""

    span_id: str
    unit_id: str
    transcription_id: str
    divergent_witness: bool
    coptic_text: str
    attested_surviving: str
    attested_editorial: str
    predicted_greek: str
    surviving_words: int
    gold_content_lemmas: list[str] = Field(default_factory=list)
    matched_content_lemmas: list[str] = Field(default_factory=list)
    surviving_char_f1_itacism: float


class EvalReport(BaseModel):
    """Complete Phase F offline evaluation report."""

    id: str = "eval-report-v1"
    model_id: str = "model-v1"
    nt_holdout: NtHoldoutMetrics
    poxy_convergent: FragStratumMetrics
    poxy_divergent: FragStratumMetrics
    poxy_overall: FragStratumMetrics
    spans: list[SpanEvalSummary] = Field(default_factory=list)


def _prf(matched: int, pred_total: int, gold_total: int) -> tuple[float, float, float]:
    p = matched / pred_total if pred_total > 0 else 0.0
    r = matched / gold_total if gold_total > 0 else 0.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def _multiset_overlap(
    pred_items: Iterable[str], gold_items: Iterable[str]
) -> tuple[int, int, int]:
    pc = Counter(pred_items)
    gc = Counter(gold_items)
    matched = sum((pc & gc).values())
    return matched, sum(pc.values()), sum(gc.values())


def _extract_ngrams(seq: str | list[str], n: int) -> Counter[str | tuple[str, ...]]:
    if isinstance(seq, str):
        return Counter(seq[i : i + n] for i in range(max(0, len(seq) - n + 1)))
    return Counter(tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1)))


def compute_chrf_pp(refs: list[str], hyps: list[str], *, beta: float = 2.0) -> float:
    """Compute corpus chrF++ (char n=1..6 + word n=1..2, F_beta=2)."""
    scores: list[float] = []
    b2 = beta * beta

    # Character n-grams 1..6 (ignoring spaces, standard chrF convention)
    ref_chars = [r.replace(" ", "") for r in refs]
    hyp_chars = [h.replace(" ", "") for h in hyps]
    for n in range(1, 7):
        matched = 0
        p_tot = 0
        r_tot = 0
        for rc, hc in zip(ref_chars, hyp_chars, strict=True):
            rg = _extract_ngrams(rc, n)
            hg = _extract_ngrams(hc, n)
            matched += sum((rg & hg).values())
            p_tot += sum(hg.values())
            r_tot += sum(rg.values())
        prec = matched / p_tot if p_tot > 0 else 0.0
        rec = matched / r_tot if r_tot > 0 else 0.0
        f = ((1 + b2) * prec * rec / (b2 * prec + rec)) if (prec + rec) > 0.0 else 0.0
        scores.append(f)

    # Word n-grams 1..2
    ref_words = [
        [
            normalize_polytonic_surface(w)
            for w in r.split()
            if normalize_polytonic_surface(w)
        ]
        for r in refs
    ]
    hyp_words = [
        [
            normalize_polytonic_surface(w)
            for w in h.split()
            if normalize_polytonic_surface(w)
        ]
        for h in hyps
    ]
    for n in (1, 2):
        matched = 0
        p_tot = 0
        r_tot = 0
        for rw, hw in zip(ref_words, hyp_words, strict=True):
            rg = _extract_ngrams(rw, n)
            hg = _extract_ngrams(hw, n)
            matched += sum((rg & hg).values())
            p_tot += sum(hg.values())
            r_tot += sum(rg.values())
        prec = matched / p_tot if p_tot > 0 else 0.0
        rec = matched / r_tot if r_tot > 0 else 0.0
        f = ((1 + b2) * prec * rec / (b2 * prec + rec)) if (prec + rec) > 0.0 else 0.0
        scores.append(f)

    return round(sum(scores) / max(len(scores), 1), 4)


def compute_bleu_4(refs: list[str], hyps: list[str]) -> float:
    """Compute corpus smoothed BLEU-4 on polytonic normalized Greek words."""
    ref_toks = [
        [
            normalize_polytonic_surface(w)
            for w in r.split()
            if normalize_polytonic_surface(w)
        ]
        for r in refs
    ]
    hyp_toks = [
        [
            normalize_polytonic_surface(w)
            for w in h.split()
            if normalize_polytonic_surface(w)
        ]
        for h in hyps
    ]
    c = sum(len(h) for h in hyp_toks)
    r = sum(len(rf) for rf in ref_toks)
    if c == 0 or r == 0:
        return 0.0

    log_precs: list[float] = []
    for n in range(1, 5):
        matched = 0
        total = 0
        for rt, ht in zip(ref_toks, hyp_toks, strict=True):
            rg = _extract_ngrams(rt, n)
            hg = _extract_ngrams(ht, n)
            matched += sum((rg & hg).values())
            total += sum(hg.values())
        # Chen-Cherry add-1 smoothing for higher orders when sparse
        if n == 1:
            p = matched / max(total, 1)
        else:
            p = (matched + 1.0) / (total + 1.0)
        log_precs.append(math.log(max(p, 1e-9)))
    bp = 1.0 if c > r else math.exp(1.0 - r / c)
    return round(bp * math.exp(sum(log_precs) / 4.0), 4)


def _norm_lemma(lemma: str) -> str:
    return strip_greek_diacritics(lemma)


def _morph_matches(pred: GeneratedToken, gold_morph: GreekMorphFeatures) -> bool:
    """Check core grammatical agreement on matched lemma pairs."""
    if pred.morph.pos != gold_morph.pos:
        return False
    if gold_morph.pos in {"N-", "A-", "RA", "RP", "RD", "RI", "RR"}:
        return (pred.morph.case == gold_morph.case) and (
            pred.morph.number == gold_morph.number
        )
    if gold_morph.pos == "V-":
        if gold_morph.mood == "P":
            return pred.morph.mood == "P" and pred.morph.case == gold_morph.case
        return (
            pred.morph.mood == gold_morph.mood
            and pred.morph.person == gold_morph.person
            and pred.morph.number == gold_morph.number
        )
    return True


def evaluate_nt_holdouts(
    *,
    repo_root: Path,
    model_dir: Path,
    coptic_analyzer: CopticAnalyzer,
    decoder: DecoderProtocol,
) -> NtHoldoutMetrics:
    """Run intact 41-verse NT Gospel holdout evaluation (no Thomas leakage)."""
    downloads = repo_root / "data" / "downloads"
    split = load_nt_split(resolve_nt_split_path(repo_root))
    holdout_ids = {
        uid
        for p in split.passages
        if p.role is PassageRole.HOLDOUT
        for uid in p.unit_ids
    }
    raw_pairs = pair_horner_morphgnt(
        sword_root=downloads / "sword",
        morphgnt_dir=downloads / "morphgnt-sblgnt",
    )
    holdout_raw: list[tuple[Any, GreekUnit]] = [
        (c_u, g_u) for c_u, g_u in raw_pairs if g_u.unit_id in holdout_ids
    ]
    analyzed_c = coptic_analyzer.analyze_units([c for c, _ in holdout_raw])
    c_by_id = {c.unit_id: c for c in analyzed_c}

    non_empty = 0
    cl_matched = cl_pred = cl_gold = 0
    all_matched = all_pred = all_gold = 0
    surf_matched = surf_pred = surf_gold = 0
    morph_checked = morph_ok = 0
    refs: list[str] = []
    hyps: list[str] = []

    for _, g_unit in holdout_raw:
        c_unit = c_by_id[g_unit.unit_id]
        gen = decoder.decode_unit(c_unit)
        if gen.greek_text.strip() and gen.tokens:
            non_empty += 1

        refs.append(
            g_unit.normalized_text or " ".join(t.surface for t in g_unit.tokens)
        )
        hyps.append(gen.greek_text)

        p_content = [
            _norm_lemma(t.lemma) for t in gen.tokens if t.morph.pos in CONTENT_POS
        ]
        g_content = [
            _norm_lemma(t.lemma) for t in g_unit.tokens if t.morph.pos in CONTENT_POS
        ]
        m_c, p_c, g_c = _multiset_overlap(p_content, g_content)
        cl_matched += m_c
        cl_pred += p_c
        cl_gold += g_c

        p_all = [_norm_lemma(t.lemma) for t in gen.tokens if t.morph.pos != "X-"]
        g_all = [_norm_lemma(t.lemma) for t in g_unit.tokens if t.morph.pos != "X-"]
        m_a, p_a, g_a = _multiset_overlap(p_all, g_all)
        all_matched += m_a
        all_pred += p_a
        all_gold += g_a

        p_surf = [
            normalize_polytonic_surface(t.surface)
            for t in gen.tokens
            if t.morph.pos != "X-" and normalize_polytonic_surface(t.surface)
        ]
        g_surf = [
            normalize_polytonic_surface(t.normalized or t.surface)
            for t in g_unit.tokens
            if t.morph.pos != "X-"
            and normalize_polytonic_surface(t.normalized or t.surface)
        ]
        m_s, p_s, g_s = _multiset_overlap(p_surf, g_surf)
        surf_matched += m_s
        surf_pred += p_s
        surf_gold += g_s

        # Check morph accuracy on greedily matched content/function lemmas
        used_gold: set[int] = set()
        for pt in gen.tokens:
            if pt.morph.pos == "X-":
                continue
            plem = _norm_lemma(pt.lemma)
            for idx, gt in enumerate(g_unit.tokens):
                if idx in used_gold or gt.morph.pos == "X-":
                    continue
                if _norm_lemma(gt.lemma) == plem:
                    used_gold.add(idx)
                    morph_checked += 1
                    if _morph_matches(pt, gt.morph):
                        morph_ok += 1
                    break

    cp, cr, cf1 = _prf(cl_matched, cl_pred, cl_gold)
    _, _, af1 = _prf(all_matched, all_pred, all_gold)
    _, _, sf1 = _prf(surf_matched, surf_pred, surf_gold)
    return NtHoldoutMetrics(
        unit_count=len(holdout_raw),
        coverage=round(non_empty / max(len(holdout_raw), 1), 4),
        content_lemma_precision=cp,
        content_lemma_recall=cr,
        content_lemma_f1=cf1,
        all_token_lemma_f1=af1,
        inflected_surface_f1=sf1,
        morph_accuracy_on_matched_lemmas=round(morph_ok / max(morph_checked, 1), 4),
        chrf_pp=compute_chrf_pp(refs, hyps),
        bleu_4=compute_bleu_4(refs, hyps),
    )


class _AttestedGreekLemmatiser:
    """Eval-only gold lemmatizer mapping surviving P.Oxy. forms to Koine lemmas."""

    def __init__(
        self,
        exact_map: dict[str, tuple[str, str]],
        ita_map: dict[str, tuple[str, str]],
        inflector: GreekInflector | None = None,
    ) -> None:
        self._exact = exact_map
        self._ita = ita_map
        self._inflector = inflector

    @classmethod
    def from_morphgnt(cls, morphgnt_dir: Path) -> _AttestedGreekLemmatiser:
        exact_counts: dict[str, Counter[tuple[str, str]]] = {}
        ita_counts: dict[str, Counter[tuple[str, str]]] = {}
        units = load_morphgnt_nt(morphgnt_dir)
        for unit in units.values():
            for tok in unit.tokens:
                if tok.morph.pos == "X-":
                    continue
                pair = (unicodedata.normalize("NFC", tok.lemma), tok.morph.pos)
                k1 = strip_greek_diacritics(tok.surface)
                k2 = koine_itacism_fold(tok.surface)
                if k1:
                    exact_counts.setdefault(k1, Counter())[pair] += 1
                if k2:
                    ita_counts.setdefault(k2, Counter())[pair] += 1
        exact_map = {k: cnt.most_common(1)[0][0] for k, cnt in exact_counts.items()}
        ita_map = {k: cnt.most_common(1)[0][0] for k, cnt in ita_counts.items()}
        inflector = GreekInflector.build(morphgnt_dir=morphgnt_dir)
        return cls(exact_map=exact_map, ita_map=ita_map, inflector=inflector)

    def resolve_word(self, word: FragmentWord) -> tuple[str, str] | None:
        """Return ``(lemma, pos)`` ONLY if ``word.surviving_text`` is non-empty.

        Words inside complete lacunae (`surviving_text == ""`) are excluded.
        """
        if not word.surviving_text.strip():
            return None
        if (
            word.is_truncated
            and len(strip_greek_diacritics(word.full_editorial_word)) <= 2
        ):
            return None

        # Check nomina sacra & scribal abbreviations on surviving text first
        surv_key = strip_greek_diacritics(word.surviving_text)
        if surv_key in _NOMINA_SACRA:
            return _NOMINA_SACRA[surv_key]

        form = word.full_editorial_word or word.surviving_text
        bare = strip_greek_diacritics(form)
        if not bare:
            return None
        if bare in _NOMINA_SACRA:
            return _NOMINA_SACRA[bare]
        if bare in _POXY_LEX_STEM_FALLBACK:
            return _POXY_LEX_STEM_FALLBACK[bare]
        if bare in self._exact:
            return self._exact[bare]
        ita = koine_itacism_fold(form)
        if ita in self._ita:
            return self._ita[ita]
        if self._inflector is not None:
            for cand in (form, bare.replace("ει", "ι"), bare.replace("αι", "ε")):
                if not cand:
                    continue
                parses = self._inflector.analyze_surface(cand)
                for lem, mf, _lp in parses:
                    if mf.pos != "UNK":
                        return (unicodedata.normalize("NFC", lem), mf.pos)
        return (unicodedata.normalize("NFC", form.lower()), "N-")


def _monotone_surviving_char_overlap(
    surv_words: list[FragmentWord],
    pred_surfaces: list[str],
    *,
    normalizer: Callable[[str], str],
) -> tuple[int, int, int, int]:
    """Align surviving non-abbreviated FragmentWords monotonically to predicted words.

    Returns ``(matched_chars, pred_aligned_chars, surv_total_chars, three_gram_hits)``.
    Truncated words (`is_truncated=True`) score ONLY against
    `pred_norm[:len(surv_norm)]` so lost lacuna letters never penalize endings.
    """
    surv_items = [
        (w, normalizer(w.surviving_text))
        for w in surv_words
        if w.surviving_text.strip()
        and not w.is_abbreviation
        and normalizer(w.surviving_text)
    ]
    pred_norms = [normalizer(s) for s in pred_surfaces if normalizer(s)]
    surv_total = sum(len(sn) for _, sn in surv_items)
    if not surv_items or not pred_norms:
        return 0, 0, surv_total, 0

    m = len(surv_items)
    n = len(pred_norms)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    choice = [[0] * (n + 1) for _ in range(m + 1)]  # 1=skip_i, 2=skip_j, 3=align

    def _pair_score(w: FragmentWord, sn: str, pn: str) -> tuple[int, int]:
        cmp_p = pn[: len(sn)] if w.is_truncated else pn
        ov = sum((Counter(sn) & Counter(cmp_p)).values())
        return ov, len(cmp_p)

    for i in range(1, m + 1):
        w, sn = surv_items[i - 1]
        for j in range(1, n + 1):
            pn = pred_norms[j - 1]
            ov, cmp_len = _pair_score(w, sn, pn)
            best = dp[i - 1][j]
            ch = 1
            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
                ch = 2
            # Small tie-breaker toward compact exact alignment
            align_val = dp[i - 1][j - 1] + ov * 10 - max(0, cmp_len - len(sn))
            if ov > 0 and align_val > best:
                best = align_val
                ch = 3
            dp[i][j] = best
            choice[i][j] = ch

    matched_chars = 0
    pred_aligned_chars = 0
    i, j = m, n
    while i > 0 and j > 0:
        c = choice[i][j]
        if c == 3:
            w, sn = surv_items[i - 1]
            ov, cmp_len = _pair_score(w, sn, pred_norms[j - 1])
            matched_chars += ov
            pred_aligned_chars += cmp_len
            i -= 1
            j -= 1
        elif c == 1:
            i -= 1
        else:
            j -= 1

    # Also compute surviving character 3-gram recall over the full span string
    full_pred = " ".join(pred_norms)
    all_surv_3g = Counter()
    for _, sn in surv_items:
        k = min(3, len(sn))
        all_surv_3g.update(_extract_ngrams(sn, k))
    pred_3g = Counter()
    for pn in pred_norms:
        for k in (1, 2, 3):
            if len(pn) >= k:
                pred_3g.update(_extract_ngrams(pn, k))
    _ = full_pred  # clarity
    tg_hits = sum((all_surv_3g & pred_3g).values())
    tg_tot = sum(all_surv_3g.values())
    return (
        matched_chars,
        pred_aligned_chars,
        surv_total,
        int(tg_hits * 1000 // max(tg_tot, 1)),
    )


def score_fragment_span_lacuna_aware(
    *,
    span: FragmentClauseSpan,
    frag_words: list[FragmentWord],
    pred_tokens: list[GeneratedToken],
    lemmatiser: _AttestedGreekLemmatiser,
    loanword_lemmas: list[str],
) -> tuple[dict[str, int | float | list[str]], SpanEvalSummary]:
    """Compute exact lacuna-aware metrics for one aligned ``FragmentClauseSpan``."""
    surv_words = frag_words[span.frag_word_start : span.frag_word_end]
    pred_surfs = [t.surface for t in pred_tokens if t.morph.pos != "X-"]

    # 1. Surviving char F1 (diplomatic exact + Koine itacism normalized)
    ex_m, ex_pa, surv_ex_tot, _ = _monotone_surviving_char_overlap(
        surv_words, pred_surfs, normalizer=strip_greek_diacritics
    )
    it_m, it_pa, surv_it_tot, _ = _monotone_surviving_char_overlap(
        surv_words, pred_surfs, normalizer=koine_itacism_fold
    )

    # Character 3-gram recall on surviving chunks (>= 3 chars)
    surv_3g = Counter()
    for w in surv_words:
        if not w.surviving_text.strip() or w.is_abbreviation:
            continue
        sn = koine_itacism_fold(w.surviving_text)
        if len(sn) >= 3:
            surv_3g.update(_extract_ngrams(sn, 3))
    pred_3g = Counter()
    for s in pred_surfs:
        pn = koine_itacism_fold(s)
        if len(pn) >= 3:
            pred_3g.update(_extract_ngrams(pn, 3))
    tg_matched = sum((surv_3g & pred_3g).values())
    tg_total = sum(surv_3g.values())

    # 2. Surviving content lemmas (excluding pure lacunae: surviving_text == "")
    gold_content_raw: list[str] = []
    for w in surv_words:
        res = lemmatiser.resolve_word(w)
        if res is None:
            continue
        lem, pos = res
        if pos in CONTENT_POS:
            gold_content_raw.append(lem)

    gold_content_norm = [_norm_lemma(g) for g in gold_content_raw]
    pred_content_norm = [
        _norm_lemma(t.lemma) for t in pred_tokens if t.morph.pos in CONTENT_POS
    ]
    cl_matched, cl_pred, cl_gold = _multiset_overlap(
        pred_content_norm, gold_content_norm
    )

    # Top-3 content lemma recall across predicted token alternatives
    gold_counter = Counter(gold_content_norm)
    avail_top3: list[set[str]] = [
        {
            _norm_lemma(t.lemma),
            *(_norm_lemma(a.lemma) for a in t.alternatives[:3]),
        }
        for t in pred_tokens
        if t.morph.pos in CONTENT_POS
    ]
    used_pred_idx: set[int] = set()
    top3_hits = 0
    matched_list: list[str] = []
    pred_c_set = Counter(pred_content_norm)
    for g_lem, cnt in gold_counter.items():
        # Exact top-1 hits first
        m1 = min(cnt, pred_c_set.get(g_lem, 0))
        for _ in range(m1):
            matched_list.append(g_lem)
        rem = cnt
        for idx, cand_set in enumerate(avail_top3):
            if rem == 0:
                break
            if idx in used_pred_idx:
                continue
            if g_lem in cand_set:
                used_pred_idx.add(idx)
                top3_hits += 1
                rem -= 1

    # Loanword exact match rate
    lw_gold = [
        _norm_lemma(lw)
        for lw in loanword_lemmas
        if _norm_lemma(lw) in set(gold_content_norm)
    ]
    lw_hits = sum(1 for lw in lw_gold if pred_c_set.get(lw, 0) > 0)

    _, _, it_f1 = _prf(it_m, it_pa, surv_it_tot)
    surv_disp = " ".join(w.surviving_text for w in surv_words if w.surviving_text)
    ed_disp = " ".join(
        w.full_editorial_word for w in surv_words if w.full_editorial_word
    )
    pred_disp = " ".join(t.surface for t in pred_tokens)

    counts: dict[str, Any] = {
        "surv_words": sum(1 for w in surv_words if w.surviving_text.strip()),
        "ex_m": ex_m,
        "ex_pa": ex_pa,
        "surv_ex_tot": surv_ex_tot,
        "it_m": it_m,
        "it_pa": it_pa,
        "surv_it_tot": surv_it_tot,
        "tg_matched": tg_matched,
        "tg_total": tg_total,
        "cl_matched": cl_matched,
        "cl_pred": cl_pred,
        "cl_gold": cl_gold,
        "top3_hits": top3_hits,
        "lw_hits": lw_hits,
        "lw_total": len(lw_gold),
    }
    summary = SpanEvalSummary(
        span_id=span.span_id,
        unit_id=span.unit_id,
        transcription_id=span.transcription_id,
        divergent_witness=span.divergent_witness,
        coptic_text="",
        attested_surviving=surv_disp,
        attested_editorial=ed_disp,
        predicted_greek=pred_disp,
        surviving_words=int(counts["surv_words"]),
        gold_content_lemmas=gold_content_raw,
        matched_content_lemmas=matched_list,
        surviving_char_f1_itacism=it_f1,
    )
    return counts, summary


def _aggregate_frag_stratum(
    rows: list[dict[str, Any]], spans: list[FragmentClauseSpan]
) -> FragStratumMetrics:
    if not rows:
        return FragStratumMetrics(
            span_count=0,
            unit_count=0,
            surviving_word_count=0,
            surviving_char_count=0,
            surviving_char_f1_exact=0.0,
            surviving_char_f1_itacism=0.0,
            surviving_char_3gram_recall_itacism=0.0,
            content_lemma_precision=0.0,
            content_lemma_recall=0.0,
            content_lemma_f1=0.0,
            top3_content_lemma_recall=0.0,
            loanword_exact_match_rate=0.0,
            loanword_total=0,
        )
    tot = Counter()
    for r in rows:
        for k, v in r.items():
            tot[k] += int(v)
    _, _, ex_f1 = _prf(tot["ex_m"], tot["ex_pa"], tot["surv_ex_tot"])
    _, _, it_f1 = _prf(tot["it_m"], tot["it_pa"], tot["surv_it_tot"])
    cp, cr, cf1 = _prf(tot["cl_matched"], tot["cl_pred"], tot["cl_gold"])
    return FragStratumMetrics(
        span_count=len(rows),
        unit_count=len({s.unit_id for s in spans}),
        surviving_word_count=tot["surv_words"],
        surviving_char_count=tot["surv_it_tot"],
        surviving_char_f1_exact=ex_f1,
        surviving_char_f1_itacism=it_f1,
        surviving_char_3gram_recall_itacism=round(
            tot["tg_matched"] / max(tot["tg_total"], 1), 4
        ),
        content_lemma_precision=cp,
        content_lemma_recall=cr,
        content_lemma_f1=cf1,
        top3_content_lemma_recall=round(tot["top3_hits"] / max(tot["cl_gold"], 1), 4),
        loanword_exact_match_rate=round(tot["lw_hits"] / max(tot["lw_total"], 1), 4)
        if tot["lw_total"] > 0
        else 1.0,
        loanword_total=tot["lw_total"],
    )


def evaluate_poxy_fragments(
    *,
    repo_root: Path,
    decoder: DecoderProtocol,
) -> tuple[
    FragStratumMetrics, FragStratumMetrics, FragStratumMetrics, list[SpanEvalSummary]
]:
    """Run span-aligned P.Oxy. 1/654/655 evaluation stratified by divergence."""
    map_path = repo_root / FRAGMENT_CLAUSE_MAP_REL
    clause_map = FragmentClauseMap.model_validate(
        json.loads(map_path.read_text(encoding="utf-8"))
    )
    frags: FragmentCorpus = load_fragment_corpus(
        resolve_fragments_corpus_path(repo_root)
    )
    frag_by_id = {t.id: t for t in frags.transcriptions}
    thomas = load_thomas_corpus(resolve_thomas_path(repo_root))
    c_by_id = {u.unit_id: u for u in thomas.units}
    lemmatiser = _AttestedGreekLemmatiser.from_morphgnt(
        repo_root / "data" / "downloads" / "morphgnt-sblgnt"
    )

    decoded_cache: dict[str, GeneratedUnit] = {}
    conv_rows: list[dict[str, Any]] = []
    conv_spans: list[FragmentClauseSpan] = []
    div_rows: list[dict[str, Any]] = []
    div_spans: list[FragmentClauseSpan] = []
    summaries: list[SpanEvalSummary] = []

    for span in clause_map.spans:
        c_unit = c_by_id[span.unit_id]
        if span.unit_id not in decoded_cache:
            decoded_cache[span.unit_id] = decoder.decode_unit(c_unit)
        gen = decoded_cache[span.unit_id]

        seq_set = set(range(span.coptic_start, span.coptic_end))
        span_tokens = (
            [t for t in gen.tokens if set(t.coptic_seqs) & seq_set] if seq_set else []
        )
        c_slice_ms = c_unit.morphemes[span.coptic_start : span.coptic_end]
        coptic_slice_txt = " ".join(m.norm for m in c_slice_ms)

        loan_lemmas: list[str] = []
        for m in c_slice_ms:
            if m.lang == "Greek":
                for entry in decoder.ccl_lexicon.resolve(
                    norm=m.norm, lemma=m.lemma, pos=m.pos, lang=m.lang
                )[:1]:
                    loan_lemmas.append(entry.greek_lemma)

        tx = frag_by_id[span.transcription_id]
        counts, summary = score_fragment_span_lacuna_aware(
            span=span,
            frag_words=tx.words,
            pred_tokens=span_tokens,
            lemmatiser=lemmatiser,
            loanword_lemmas=loan_lemmas,
        )
        summary.coptic_text = coptic_slice_txt
        summaries.append(summary)
        if span.divergent_witness:
            div_rows.append(counts)
            div_spans.append(span)
        else:
            conv_rows.append(counts)
            conv_spans.append(span)

    return (
        _aggregate_frag_stratum(conv_rows, conv_spans),
        _aggregate_frag_stratum(div_rows, div_spans),
        _aggregate_frag_stratum(conv_rows + div_rows, conv_spans + div_spans),
        summaries,
    )


def format_benchmark_markdown(report: EvalReport) -> str:
    """Render concise Markdown benchmark report with side-by-side P.Oxy. tables."""
    nt = report.nt_holdout
    cv = report.poxy_convergent
    dv = report.poxy_divergent
    ov = report.poxy_overall

    def _frag_row(label: str, m: FragStratumMetrics, *, bold: bool = False) -> str:
        f1_s = f"**{m.content_lemma_f1:.4f}**" if bold else f"{m.content_lemma_f1:.4f}"
        return (
            f"| {label} | {m.span_count} | {m.unit_count} | "
            f"{m.surviving_word_count} | {m.surviving_char_f1_exact:.4f} | "
            f"{m.surviving_char_f1_itacism:.4f} | "
            f"{m.surviving_char_3gram_recall_itacism:.4f} | "
            f"{m.content_lemma_precision:.4f} | {m.content_lemma_recall:.4f} | "
            f"{f1_s} | {m.top3_content_lemma_recall:.4f} | "
            f"{m.loanword_exact_match_rate:.4f} ({m.loanword_total}) |"
        )

    frag_hdr = (
        "| Stratum | Spans | Units | Surv. Words | Char F1 (Exact) | "
        "Char F1 (Itacism) | 3g Recall (Itacism) | Content Lemma P | "
        "Content Lemma R | **Content Lemma F1** | Top-3 Recall | Loanword Exact |"
    )
    frag_sep = (
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |"
    )
    side_hdr = (
        "| Span | Coptic (Dilley) | Attested Greek (Editorial / Surviving) | "
        "Generated Category C Koine | Itacism Char F1 |"
    )

    lines = [
        f"# Benchmark Evaluation Report (`{report.model_id}`)",
        "",
        "Offline, zero-leakage evaluation across 41 NT holdout verses "
        "and span-aligned P.Oxy. 1/654/655 papyri.",
        "",
        "## 1. Intact NT Gospel Holdout Split (`41 verses`)",
        "",
        "| Metric | Score | Notes |",
        "| --- | ---: | --- |",
        f"| Verse Count | {nt.unit_count} | Excluded from train, LM & dev |",
        f"| Non-Empty Greek Coverage | {nt.coverage:.4f} | Zero-null guarantee |",
        (
            f"| Content Lemma Precision | {nt.content_lemma_precision:.4f} | "
            "POS in `{N-, V-, A-, D-}` |"
        ),
        (
            f"| Content Lemma Recall | {nt.content_lemma_recall:.4f} | "
            "Multiset over gold MorphGNT |"
        ),
        f"| **Content Lemma F1** | **{nt.content_lemma_f1:.4f}** | Target >= 0.50 |",
        (
            f"| All-Token Lemma F1 | {nt.all_token_lemma_f1:.4f} | "
            "Articles, prepositions, particles |"
        ),
        (
            f"| Inflected Surface F1 | {nt.inflected_surface_f1:.4f} | "
            "Accent-preserving forms |"
        ),
        (
            f"| Morph Accuracy (matched) | "
            f"{nt.morph_accuracy_on_matched_lemmas:.4f} | Case/Num or Mood/Pers/Num |"
        ),
        f"| chrF++ (beta=2) | {nt.chrf_pp:.4f} | Char 1-6 + word 1-2 |",
        f"| BLEU-4 (smoothed) | {nt.bleu_4:.4f} | Corpus normalized surface |",
        "",
        "## 2. Span-Aligned, Lacuna-Aware P.Oxy. 1 / 654 / 655 (20 units)",
        "",
        frag_hdr,
        frag_sep,
        _frag_row("**Convergent Clauses**", cv, bold=True),
        _frag_row("Divergent / Redactional", dv),
        _frag_row("**Overall**", ov, bold=True),
        "",
        "## 3. Side-by-Side Comparison on Key P.Oxy. Convergent Spans",
        "",
        side_hdr,
        "| --- | --- | --- | --- | ---: |",
    ]

    for sp in report.spans:
        if sp.divergent_witness:
            continue
        c_s = sp.coptic_text.replace("|", "/")
        g_s = sp.attested_editorial.replace("|", "/")
        p_s = sp.predicted_greek.replace("|", "/")
        f1_c = sp.surviving_char_f1_itacism
        lines.append(f"| `{sp.span_id}` | {c_s} | {g_s} | {p_s} | {f1_c:.3f} |")
    lines.append("")
    return "\n".join(lines)


def run_full_evaluation(
    *,
    repo_root: Path,
    model_dir: Path | None = None,
) -> EvalReport:
    """Run complete NT holdout + P.Oxy. evaluation offline (model-v1 or model-v2)."""
    v1_dir = repo_root / "data" / "mappings" / "model-v1"
    mdir = model_dir or v1_dir
    is_v2 = (mdir / "seq2seq-config.json").is_file()
    analyzer_dir = mdir if (mdir / "coptic-meta.json").is_file() else v1_dir
    coptic_analyzer = CopticAnalyzer.load_shards(analyzer_dir)
    decoder: DecoderProtocol
    if is_v2:
        from zetodos_thomas.neural.decoder_v2 import NeuralKoineDecoder

        decoder = NeuralKoineDecoder.from_model_dir(
            mdir, v1_model_dir=v1_dir, device="cpu"
        )
    else:
        decoder = KoineBeamDecoder.from_model_dir(mdir)
    nt = evaluate_nt_holdouts(
        repo_root=repo_root,
        model_dir=mdir,
        coptic_analyzer=coptic_analyzer,
        decoder=decoder,
    )
    conv, div, ov, spans = evaluate_poxy_fragments(
        repo_root=repo_root,
        decoder=decoder,
    )
    model_id = mdir.name
    rep_id = (
        f"eval-report-{model_id.removeprefix('model-')}"
        if model_id.startswith("model-")
        else f"eval-report-{model_id}"
    )
    return EvalReport(
        id=rep_id,
        model_id=model_id,
        nt_holdout=nt,
        poxy_convergent=conv,
        poxy_divergent=div,
        poxy_overall=ov,
        spans=spans,
    )
