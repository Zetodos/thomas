# SPDX-License-Identifier: Apache-2.0
"""Build a Coptic→Greek dictionary from train-only dual-aligner counts.

The only evidence is observed alignment links. Handwritten glosses and the
verified loanword table are not inputs. Thomas / P.Oxy. unit ids are refused.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from zetodos_thomas.align.segment_coptic import peel_bound_group
from zetodos_thomas.dictionary.orthography import (
    common_prefix_len,
    fold_coptic,
    fold_greek,
    greek_forms_similar,
    is_prefix_chain,
    levenshtein,
    loan_shape,
    loan_stem,
    same_paradigm,
    spelling_variants,
    transliterate_coptic,
    transliteration_similar,
)

# Iteration bars for this experiment, fixed before looking at fragment scores.
# They are not the Phase 2 feasibility gate.
MIN_COUNT = 2
MIN_POSTERIOR = 0.40
MIN_MARGIN = 0.15
SNAP_POSTERIOR = 0.55
SNAP_MIN_GREEK_COUNT = 2
DEV_MIN_TRIALS = 5
DEV_MIN_PRECISION = 0.60
LOANWORD_SUFFICIENT_RATE = 0.50
LOANWORD_MIN_SUPPORT = 8
CONTENT_RECALL_BAR = 0.20
CONTENT_PRECISION_BAR = 0.25

FORBIDDEN_UNIT_MARKERS = ("thomas.", "poxy", "p.oxy")
REQUIRED_ALIGNERS = frozenset({"eflomal", "fast_align"})


class DictionaryBuildError(ValueError):
    """Train links are missing, single-aligner, or contaminated."""


@dataclass(frozen=True, slots=True)
class ObservedLink:
    """One aggregated Coptic↔Greek link from the train candidate table."""

    coptic: str
    greek: str
    observation_count: int
    count_eflomal: int
    count_fast_align: int


@dataclass(frozen=True, slots=True)
class Emission:
    """One Greek surface the dictionary is willing to emit."""

    greek: str
    posterior: float
    observation_count: int
    channel: str
    count_eflomal: int
    count_fast_align: int
    greek_extra: tuple[str, ...] = ()


@dataclass
class _Acc:
    observation_count: int = 0
    count_eflomal: int = 0
    count_fast_align: int = 0

    def add(self, link: ObservedLink) -> None:
        self.observation_count += link.observation_count
        self.count_eflomal += link.count_eflomal
        self.count_fast_align += link.count_fast_align


@dataclass
class ParallelDictionary:
    """Train-derived emissions plus attested Greek types for orthographic snap.

    ``snap_enabled`` is chosen from NT dev only. Fragment Greek never enters
    this object.
    """

    version: str
    direct: dict[str, dict[str, _Acc]]
    stems: dict[str, dict[str, _Acc]]
    greek_types: dict[str, int]
    keys_with_pairs: set[str]
    snap_enabled: bool = False
    provenance: dict[str, object] = field(default_factory=dict)
    # coptic -> greek -> previous coptic token -> observation count
    context_counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    use_context: bool = False
    # previous Coptic token -> Greek suffix -> count, pooled across lexemes
    suffix_given_prev: dict[str, dict[str, int]] = field(default_factory=dict)
    use_suffix: bool = False
    # joined Coptic tokens -> (greek tokens, count, posterior)
    phrases: dict[tuple[str, ...], tuple[tuple[str, ...], int, float]] = field(
        default_factory=dict
    )
    use_phrases: bool = False
    use_paradigm: bool = False
    # When true, a loan-shaped token prefers the attested Greek string that
    # its letters already spell, unless context or suffix already chose.
    use_loan_spelling: bool = False

    def statistical_emission(self, coptic: str) -> Emission | None:
        """Best agreeing emission for ``coptic``, including spelling variants."""
        key = fold_coptic(coptic)
        if not key:
            return None
        exact = self._emit_key(key, channel="direct")
        if exact is not None:
            return exact
        variant_hits: list[Emission] = []
        for variant in spelling_variants(key):
            if variant == key:
                continue
            hit = self._emit_key(variant, channel="spelling")
            if hit is not None:
                variant_hits.append(hit)
        greeks = {hit.greek for hit in variant_hits}
        if len(greeks) == 1:
            return variant_hits[0]
        return None

    def choose_surface(
        self,
        emission: Emission,
        coptic: str,
        previous: str | tuple[str, ...] | None,
    ) -> Emission:
        """Pick an attested inflection from local Coptic context.

        ``previous`` may be several keys, most specific first (a bigram, then
        the single preceding token). The first decisive key wins. Otherwise
        the original emission is kept.
        """
        if (
            not self.use_context
            or not previous
            or emission.channel == "orthographic_snap"
        ):
            return emission
        keys = (previous,) if isinstance(previous, str) else previous
        for context_key in keys:
            chosen = self._surface_for_context(emission, coptic, context_key)
            if chosen is not None:
                return chosen
        if self.use_suffix:
            for context_key in keys:
                if "+" in context_key:
                    continue
                transferred = self._surface_for_suffix(emission, coptic, context_key)
                if transferred is not None:
                    return transferred
        return self.prefer_loan_spelling(emission, coptic)

    def prefer_loan_spelling(self, emission: Emission, coptic: str) -> Emission:
        """Prefer the attested Greek spelling of a loan when case was not chosen.

        The Greek string has to already be aligned to this Coptic form by both
        aligners. Nothing is invented, and a context or suffix choice stands.
        """
        if not self.use_loan_spelling:
            return emission
        if emission.channel in {"context", "suffix", "phrase", "paradigm"}:
            return emission
        key = fold_coptic(coptic)
        if not key or not loan_shape(key):
            return emission
        spelled = transliterate_coptic(key)
        if not spelled or spelled == emission.greek:
            return emission
        acc = self.direct.get(key, {}).get(spelled)
        if acc is None or acc.count_eflomal <= 0 or acc.count_fast_align <= 0:
            return emission
        if acc.observation_count < MIN_COUNT:
            return emission
        return Emission(
            greek=spelled,
            posterior=emission.posterior,
            observation_count=acc.observation_count,
            channel="loan_spelling",
            count_eflomal=acc.count_eflomal,
            count_fast_align=acc.count_fast_align,
        )

    def _surface_for_context(
        self,
        emission: Emission,
        coptic: str,
        previous: str,
    ) -> Emission | None:
        key = fold_coptic(coptic)
        prev = previous if "+" in previous else fold_coptic(previous)
        if not key or not prev:
            return None
        by_greek = self.context_counts.get(key, {})
        scored: list[tuple[int, str]] = []
        for greek, prevs in by_greek.items():
            if greek != emission.greek and not same_paradigm(greek, emission.greek):
                continue
            count = prevs.get(prev, 0)
            if count > 0:
                scored.append((count, greek))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_count, best_greek = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0
        if best_count < 2 or best_count < second + 2:
            return None
        if best_greek == emission.greek:
            return emission
        return Emission(
            greek=best_greek,
            posterior=emission.posterior,
            observation_count=best_count,
            channel="context",
            count_eflomal=emission.count_eflomal,
            count_fast_align=emission.count_fast_align,
        )

    def _surface_for_suffix(
        self,
        emission: Emission,
        coptic: str,
        previous: str,
    ) -> Emission | None:
        """Reuse a suffix that this context prefers on other lexemes.

        The full word has to have been aligned to this Coptic form already.
        The transfer only reorders those attested surfaces.
        """
        key = fold_coptic(coptic)
        prev = fold_coptic(previous)
        if not key or not prev:
            return None
        forms = self._attested_paradigm(key, emission.greek)
        if len(forms) < 2:
            return None
        stem = _shared_prefix(forms)
        if len(stem) < 3:
            return None
        table = self.suffix_given_prev.get(prev, {})
        scored: list[tuple[int, str]] = []
        for form in forms:
            if not form.startswith(stem):
                continue
            score = table.get(form[len(stem) :], 0)
            if score > 0:
                scored.append((score, form))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_form = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0
        if best_score < 8 or best_score < second * 2:
            return None
        if best_form == emission.greek:
            return emission
        return Emission(
            greek=best_form,
            posterior=emission.posterior,
            observation_count=emission.observation_count,
            channel="suffix",
            count_eflomal=emission.count_eflomal,
            count_fast_align=emission.count_fast_align,
        )

    def paradigm_extras(self, coptic: str, greek: str) -> tuple[str, ...]:
        """Other attested inflections of the chosen stem, most frequent first."""
        if not self.use_paradigm:
            return ()
        key = fold_coptic(coptic)
        forms = self._attested_paradigm(key, greek)
        ranked: list[tuple[int, str]] = []
        for form in forms:
            if form == greek:
                continue
            acc = self.direct.get(key, {}).get(form)
            count = acc.observation_count if acc is not None else 0
            ranked.append((count, form))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(form for _, form in ranked[:3])

    def phrase_emission(self, keys: tuple[str, ...]) -> Emission | None:
        """Greek phrase both aligners attached to this Coptic token sequence."""
        if not self.use_phrases or len(keys) < 2:
            return None
        found = self.phrases.get(keys)
        if found is None:
            return None
        greek, count, posterior = found
        if not greek:
            return None
        return Emission(
            greek=greek[0],
            posterior=posterior,
            observation_count=count,
            channel="phrase",
            count_eflomal=count,
            count_fast_align=count,
            greek_extra=greek[1:],
        )

    def _attested_paradigm(self, key: str, greek: str) -> list[str]:
        forms = [greek]
        for other, acc in self.direct.get(key, {}).items():
            if other == greek:
                continue
            if acc.count_eflomal <= 0 or acc.count_fast_align <= 0:
                continue
            if same_paradigm(other, greek):
                forms.append(other)
        return forms

    def snap_emission(self, coptic: str) -> Emission | None:
        """Snap an unseen loan-shaped form onto an attested train Greek type.

        Fires only when no train pair exists for the form or its spelling
        variants. The Greek string has to already occur in train alignments.
        """
        if not self.snap_enabled:
            return None
        key = fold_coptic(coptic)
        if not key or not loan_shape(key):
            return None
        if self._has_pairs(key):
            return None
        translated = transliterate_similar_target(key, self.greek_types)
        if translated is None:
            return None
        greek, count = translated
        return Emission(
            greek=greek,
            posterior=SNAP_POSTERIOR,
            observation_count=count,
            channel="orthographic_snap",
            count_eflomal=0,
            count_fast_align=0,
        )

    def _has_pairs(self, key: str) -> bool:
        if key in self.keys_with_pairs:
            return True
        return any(
            variant in self.keys_with_pairs for variant in spelling_variants(key)
        )

    def _emit_key(self, key: str, *, channel: str) -> Emission | None:
        direct = _choose(self.direct.get(key, {}))
        if direct is not None:
            greek, acc, posterior = direct
            return Emission(
                greek=greek,
                posterior=posterior,
                observation_count=acc.observation_count,
                channel=channel if channel == "spelling" else "direct",
                count_eflomal=acc.count_eflomal,
                count_fast_align=acc.count_fast_align,
            )
        stem = _choose(self.stems.get(key, {}))
        if stem is None:
            return None
        greek, acc, posterior = stem
        return Emission(
            greek=greek,
            posterior=posterior,
            observation_count=acc.observation_count,
            channel="stem" if channel == "direct" else channel,
            count_eflomal=acc.count_eflomal,
            count_fast_align=acc.count_fast_align,
        )

    def emitted_entries(self) -> list[dict[str, object]]:
        """Inspectable rows the decoder can actually emit (no abstentions)."""
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        for key in sorted(set(self.direct) | set(self.stems)):
            emission = self._emit_key(key, channel="direct")
            if emission is None:
                continue
            ident = (key, emission.greek, emission.channel)
            if ident in seen:
                continue
            seen.add(ident)
            rows.append(
                {
                    "coptic": key,
                    "greek": emission.greek,
                    "posterior": round(emission.posterior, 6),
                    "observation_count": emission.observation_count,
                    "channel": emission.channel,
                    "count_eflomal": emission.count_eflomal,
                    "count_fast_align": emission.count_fast_align,
                }
            )
        return rows


def _shared_prefix(forms: list[str]) -> str:
    if not forms:
        return ""
    stem = forms[0]
    for form in forms[1:]:
        limit = min(len(stem), len(form))
        index = 0
        while index < limit and stem[index] == form[index]:
            index += 1
        stem = stem[:index]
        if not stem:
            break
    return stem


def build_suffix_table(
    context_counts: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    """Pool Greek suffixes by the preceding Coptic token, across lexemes."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for by_greek in context_counts.values():
        forms = list(by_greek)
        clusters: list[list[str]] = []
        for form in forms:
            placed = False
            for cluster in clusters:
                if same_paradigm(form, cluster[0]):
                    cluster.append(form)
                    placed = True
                    break
            if not placed:
                clusters.append([form])
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            stem = _shared_prefix(cluster)
            if len(stem) < 3:
                continue
            for form in cluster:
                suffix = form[len(stem) :]
                for prev, count in by_greek[form].items():
                    if "+" in prev or count <= 0:
                        continue
                    table[prev][suffix] += count
    return {prev: dict(suffixes) for prev, suffixes in table.items()}


def transliterate_similar_target(
    coptic: str,
    greek_types: dict[str, int],
) -> tuple[str, int] | None:
    """Nearest attested Greek type, or None when the snap is ambiguous."""
    translated = fold_greek(transliterate_coptic(coptic))
    if len(translated) < 5:
        return None
    candidates: list[tuple[int, int, int, str]] = []
    for greek, count in greek_types.items():
        if count < SNAP_MIN_GREEK_COUNT or len(greek) < 4:
            continue
        if not greek_forms_similar(translated, greek):
            continue
        distance = levenshtein(translated, greek)
        prefix = common_prefix_len(translated, greek)
        candidates.append((distance, -prefix, -count, greek))
    if not candidates:
        return None
    candidates.sort()
    best = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        if not greek_forms_similar(best[3], second[3]) and second[0] <= best[0] + 1:
            return None
    return best[3], -best[2]


def _choose(pairs: dict[str, _Acc]) -> tuple[str, _Acc, float] | None:
    """Winning attested surface, after grouping inflections of one stem.

    The emitted Greek is the most frequent surface in the winning cluster.
    Posterior is the cluster's share of agreeing links. Unrelated Greek
    candidates stay in separate clusters, so a tie still abstains.
    """
    agreeing = [
        (greek, acc)
        for greek, acc in pairs.items()
        if acc.count_eflomal > 0
        and acc.count_fast_align > 0
        and acc.observation_count > 0
    ]
    if not agreeing:
        return None
    total = sum(acc.observation_count for _, acc in agreeing)
    if total <= 0:
        return None
    clusters = _inflection_clusters(agreeing)
    clusters.sort(
        key=lambda cluster: (
            -sum(acc.observation_count for _, acc in cluster),
            cluster[0][0],
        )
    )
    best = clusters[0]
    best_count = sum(acc.observation_count for _, acc in best)
    posterior = best_count / total
    if best_count < MIN_COUNT or posterior < MIN_POSTERIOR:
        return None
    if len(clusters) > 1:
        second_count = sum(acc.observation_count for _, acc in clusters[1])
        if posterior - (second_count / total) < MIN_MARGIN:
            return None
    mode_greek, mode_acc = max(
        best,
        key=lambda item: (item[1].observation_count, item[0]),
    )
    reported = _Acc()
    reported.observation_count = best_count
    reported.count_eflomal = mode_acc.count_eflomal
    reported.count_fast_align = mode_acc.count_fast_align
    return mode_greek, reported, posterior


def _inflection_clusters(
    agreeing: list[tuple[str, _Acc]],
) -> list[list[tuple[str, _Acc]]]:
    """Group Greek surfaces that are inflections of one another."""
    ordered = sorted(agreeing, key=lambda item: (-item[1].observation_count, item[0]))
    clusters: list[list[tuple[str, _Acc]]] = []
    for greek, acc in ordered:
        placed = False
        for cluster in clusters:
            if same_paradigm(greek, cluster[0][0]):
                cluster.append((greek, acc))
                placed = True
                break
        if not placed:
            clusters.append([(greek, acc)])
    return clusters


def _assert_safe_units(unit_ids: list[str]) -> None:
    for unit_id in unit_ids:
        lowered = unit_id.lower().replace(" ", "")
        for marker in FORBIDDEN_UNIT_MARKERS:
            if marker in lowered:
                msg = f"refusing evaluation unit in dictionary training: {unit_id}"
                raise DictionaryBuildError(msg)


def build_dictionary(
    links: list[ObservedLink],
    *,
    unit_ids: list[str],
    aligners: list[str],
    provenance: dict[str, object],
    version: str = "parallel-dict-v1",
) -> ParallelDictionary:
    """Aggregate train links into direct and loan-stem tables."""
    if set(aligners) != REQUIRED_ALIGNERS:
        msg = f"keep-both requires eflomal and fast_align; got {aligners}"
        raise DictionaryBuildError(msg)
    _assert_safe_units(unit_ids)
    direct: dict[str, dict[str, _Acc]] = {}
    stems: dict[str, dict[str, _Acc]] = {}
    greek_types: dict[str, int] = {}
    projected: set[tuple[str, str, str]] = set()

    for link in links:
        coptic = fold_coptic(link.coptic)
        greek = fold_greek(link.greek)
        if not coptic or not greek:
            continue
        if link.count_eflomal > 0 and link.count_fast_align > 0:
            greek_types[greek] = greek_types.get(greek, 0) + link.observation_count
        bucket = direct.setdefault(coptic, {})
        acc = bucket.setdefault(greek, _Acc())
        acc.add(link)
        _project_stem(link, coptic, greek, stems, projected)

    keys_with_pairs: set[str] = set()
    for key, pairs in direct.items():
        if any(
            acc.count_eflomal > 0 and acc.count_fast_align > 0 for acc in pairs.values()
        ):
            keys_with_pairs.add(key)
    for key, pairs in stems.items():
        if any(
            acc.count_eflomal > 0 and acc.count_fast_align > 0 for acc in pairs.values()
        ):
            keys_with_pairs.add(key)

    return ParallelDictionary(
        version=version,
        direct=direct,
        stems=stems,
        greek_types=greek_types,
        keys_with_pairs=keys_with_pairs,
        snap_enabled=False,
        provenance={
            **provenance,
            "version": version,
            "aligners": sorted(aligners),
            "train_unit_count": len(unit_ids),
            "uses_thomas_greek": False,
            "uses_handwritten_glosses": False,
            "uses_verified_loanword_table": False,
            "min_count": MIN_COUNT,
            "min_posterior": MIN_POSTERIOR,
            "min_margin": MIN_MARGIN,
            "inflection_clustering": (
                "Attested Greek surfaces in one paradigm share a posterior. "
                "The emitted string is the most frequent surface in that cluster."
            ),
        },
    )


def _project_stem(
    link: ObservedLink,
    coptic: str,
    greek: str,
    stems: dict[str, dict[str, _Acc]],
    projected: set[tuple[str, str, str]],
) -> None:
    """Credit a Greek link to a peeled stem when transliteration agrees."""
    if link.count_eflomal <= 0 or link.count_fast_align <= 0:
        return
    stems_found: list[str] = []
    cut = loan_stem(coptic, greek)
    if cut is not None and cut != coptic:
        stems_found.append(cut)
    pieces = peel_bound_group(coptic)
    if len(pieces) >= 2 and "".join(pieces) == coptic:
        tail = pieces[-1]
        left = "".join(pieces[:-1])
        if (
            len(tail) >= 4
            and tail != coptic
            and is_prefix_chain(left)
            and transliteration_similar(tail, greek)
        ):
            stems_found.append(tail)
    for stem in stems_found:
        ident = (coptic, stem, greek)
        if ident in projected:
            continue
        projected.add(ident)
        bucket = stems.setdefault(stem, {})
        acc = bucket.setdefault(greek, _Acc())
        acc.add(link)
