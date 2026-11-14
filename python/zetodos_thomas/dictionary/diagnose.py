# SPDX-License-Identifier: Apache-2.0
"""NT dev check for orthographic snap. Does not read Thomas Greek."""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.align.bitext import build_bitext
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    load_canonical_splits,
)
from zetodos_thomas.dictionary.build import (
    DEV_MIN_PRECISION,
    DEV_MIN_TRIALS,
    ParallelDictionary,
)
from zetodos_thomas.dictionary.orthography import (
    fold_greek,
    loan_shape,
    transliteration_similar,
)
from zetodos_thomas.ingest.paths import (
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
)
from zetodos_thomas.schemas.source import PilotCorpus


def load_nt_splits(repo_root: Path) -> PilotSplits:
    omission = resolve_omission_fixtures_path(repo_root)
    return load_canonical_splits(resolve_nt_split_path(repo_root), omission)


def nt_dev_snap_diagnostic(
    corpus: PilotCorpus,
    splits: PilotSplits,
    dictionary: ParallelDictionary,
) -> dict[str, object]:
    """Precision of snap on dev tokens the statistical dictionary abstains on.

    A trial is a loan-shaped Coptic token whose verse contains a
    transliteration neighbor, and for which the statistical dictionary emits
    nothing. A hit is a snap that lands on that neighbor. Thomas text is not
    an input.
    """
    bundle = build_bitext(corpus, splits)
    trials = 0
    hits = 0
    statistical_trials = 0
    statistical_hits = 0
    failures: list[dict[str, str]] = []
    # Snap is evaluated even when the published dictionary has it disabled.
    probe = ParallelDictionary(
        version=dictionary.version,
        direct=dictionary.direct,
        stems=dictionary.stems,
        greek_types=dictionary.greek_types,
        keys_with_pairs=dictionary.keys_with_pairs,
        snap_enabled=True,
        provenance=dictionary.provenance,
    )
    for sentence in bundle.sentences_for(SplitRole.DEV):
        gold = [fold_greek(token) for token in sentence.target_tokens]
        gold = [token for token in gold if token]
        for coptic in sentence.source_tokens:
            if not loan_shape(coptic):
                continue
            neighbors = [
                token for token in gold if transliteration_similar(coptic, token)
            ]
            if not neighbors:
                continue
            statistical = dictionary.statistical_emission(coptic)
            if statistical is not None:
                statistical_trials += 1
                if _matches(statistical.greek, neighbors):
                    statistical_hits += 1
                continue
            trials += 1
            snapped = probe.snap_emission(coptic)
            if snapped is not None and _matches(snapped.greek, neighbors):
                hits += 1
            elif len(failures) < 12:
                failures.append(
                    {
                        "coptic": coptic,
                        "snapped": snapped.greek if snapped is not None else "",
                        "neighbor": neighbors[0],
                        "unit_id": sentence.unit_id,
                    }
                )
    precision = (hits / trials) if trials else None
    stat_recall = (
        (statistical_hits / statistical_trials) if statistical_trials else None
    )
    enabled = (
        trials >= DEV_MIN_TRIALS
        and precision is not None
        and precision >= DEV_MIN_PRECISION
    )
    return {
        "split_role": SplitRole.DEV.value,
        "snap_trials": trials,
        "snap_hits": hits,
        "snap_precision": None if precision is None else round(precision, 6),
        "statistical_loan_trials": statistical_trials,
        "statistical_loan_hits": statistical_hits,
        "statistical_loan_recall": None
        if stat_recall is None
        else round(stat_recall, 6),
        "min_trials": DEV_MIN_TRIALS,
        "min_precision": DEV_MIN_PRECISION,
        "snap_enabled": enabled,
        "failures_sample": failures,
        "notes": (
            "Enable orthographic snap only from this NT dev precision. "
            "Fragment Greek is not an input."
        ),
    }


def _matches(greek: str, neighbors: list[str]) -> bool:
    from zetodos_thomas.dictionary.orthography import greek_forms_similar

    folded = fold_greek(greek)
    return any(
        folded == neighbor or greek_forms_similar(folded, neighbor)
        for neighbor in neighbors
    )
