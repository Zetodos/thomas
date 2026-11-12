# SPDX-License-Identifier: Apache-2.0
"""Pharaoh alignment I/O and pure-Python symmetrization heuristics."""

from __future__ import annotations

from zetodos_thomas.integrations.aligners.types import DirectionalAlignment


def parse_pharaoh_line(line: str) -> tuple[tuple[int, int], ...]:
    """Parse a Pharaoh ``i-j`` line into 0-based ``(src, trg)`` pairs."""
    text = line.strip()
    if not text:
        return ()
    # fast_align score mode: ``pairs ||| score``
    if " ||| " in text:
        text = text.split(" ||| ", 1)[0].strip()
    pairs: list[tuple[int, int]] = []
    for token in text.split():
        if "-" not in token:
            continue
        left, right = token.split("-", 1)
        pairs.append((int(left), int(right)))
    return tuple(pairs)


def format_pharaoh_line(
    pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> str:
    """Format pairs as a Pharaoh line."""
    return " ".join(f"{i}-{j}" for i, j in pairs)


def parse_pharaoh_batch(
    lines: list[str],
    unit_ids: list[str],
) -> list[DirectionalAlignment]:
    """Zip Pharaoh lines with unit ids (one alignment per sentence)."""
    if len(lines) != len(unit_ids):
        msg = f"pharaoh line count {len(lines)} != unit count {len(unit_ids)}"
        raise ValueError(msg)
    return [
        DirectionalAlignment(unit_id=uid, pairs=parse_pharaoh_line(line))
        for uid, line in zip(unit_ids, lines, strict=True)
    ]


def _neighbors(i: int, j: int) -> list[tuple[int, int]]:
    return [
        (i + di, j + dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
    ]


def symmetrize_grow_diag_final_and(
    forward: tuple[tuple[int, int], ...],
    reverse: tuple[tuple[int, int], ...],
    *,
    src_len: int,
    trg_len: int,
) -> tuple[tuple[int, int], ...]:
    """Classic grow-diag-final-and symmetrization (no atools dependency).

    ``forward`` is source→target (i-j); ``reverse`` is target→source as stored
    by adapters that emit reverse in source→target coordinates (i-j with the
    same source=left convention). Callers that obtain reverse from tools that
    swap axes must convert first.
    """
    fwd = set(forward)
    rev = set(reverse)
    alignment = fwd & rev
    # Grow-diag from intersection using union as candidates.
    union = fwd | rev
    growing = True
    while growing:
        growing = False
        for i, j in list(alignment):
            for ni, nj in _neighbors(i, j):
                if not (0 <= ni < src_len and 0 <= nj < trg_len):
                    continue
                if (ni, nj) not in union:
                    continue
                src_free = all(a != ni for a, _b in alignment)
                trg_free = all(b != nj for _a, b in alignment)
                if src_free or trg_free:
                    alignment.add((ni, nj))
                    growing = True
    # Final-and: add remaining union points where both sides are unaligned.
    for i, j in union:
        if (i, j) in alignment:
            continue
        src_free = all(a != i for a, _b in alignment)
        trg_free = all(b != j for _a, b in alignment)
        if src_free and trg_free:
            alignment.add((i, j))
    return tuple(sorted(alignment))


def symmetrize_batch(
    forward: list[DirectionalAlignment],
    reverse: list[DirectionalAlignment],
    src_lengths: list[int],
    trg_lengths: list[int],
    *,
    method: str = "grow-diag-final-and",
) -> list[DirectionalAlignment]:
    """Symmetrize parallel forward/reverse alignment lists."""
    if method != "grow-diag-final-and":
        msg = f"unsupported symmetrization method: {method!r}"
        raise ValueError(msg)
    if not (len(forward) == len(reverse) == len(src_lengths) == len(trg_lengths)):
        msg = "forward/reverse/length batch sizes must match"
        raise ValueError(msg)
    out: list[DirectionalAlignment] = []
    for fwd, rev, slen, tlen in zip(
        forward, reverse, src_lengths, trg_lengths, strict=True
    ):
        if fwd.unit_id != rev.unit_id:
            msg = f"unit_id mismatch: {fwd.unit_id} vs {rev.unit_id}"
            raise ValueError(msg)
        out.append(
            DirectionalAlignment(
                unit_id=fwd.unit_id,
                pairs=symmetrize_grow_diag_final_and(
                    fwd.pairs, rev.pairs, src_len=slen, trg_len=tlen
                ),
            )
        )
    return out


def flip_pairs(pairs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Swap ``(i, j)`` → ``(j, i)`` (for tools that reverse axes)."""
    return tuple((j, i) for i, j in pairs)
