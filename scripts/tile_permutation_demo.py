#!/usr/bin/env python3
"""Demo: optimize a mask permutation for fewer active tiles.

This is a practical heuristic (not exact):
- Split sequence positions into contiguous groups.
- Permute group order with local search (swap / insert moves).
- Objective: minimize active tile count after symmetric permutation
  M' = P M P^T for fixed tile size.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_binary_lines(lines: list[str]) -> list[list[int]]:
    matrix: list[list[int]] = []
    width: int | None = None
    for raw in lines:
        row = []
        for ch in raw.strip():
            if ch in ("#", "1"):
                row.append(1)
            elif ch in (".", "0"):
                row.append(0)
        if not row:
            continue
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"Inconsistent row width: expected {width}, got {len(row)}")
        matrix.append(row)
    if not matrix:
        raise ValueError("No binary rows found in input.")
    return matrix


def load_matrix_from_single_sample_log(path: Path, section_name: str) -> list[list[int]]:
    lines = path.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(section_name):
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Section not found: {section_name!r}")
    block: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        if all(ch in ".#" for ch in stripped):
            block.append(stripped)
        else:
            break
    if not block:
        raise ValueError(f"No matrix rows found after section: {section_name!r}")
    return parse_binary_lines(block)


def build_perm_from_group_order(groups: list[list[int]], order: list[int]) -> list[int]:
    out: list[int] = []
    for gidx in order:
        out.extend(groups[gidx])
    return out


def active_tiles(mask: list[list[int]], perm: list[int], tile: int) -> tuple[int, list[list[int]]]:
    s = len(mask)
    ntiles = (s + tile - 1) // tile
    occ = [[0 for _ in range(ntiles)] for _ in range(ntiles)]
    count = 0
    for ti in range(ntiles):
        i0 = ti * tile
        i1 = min((ti + 1) * tile, s)
        for tj in range(ntiles):
            j0 = tj * tile
            j1 = min((tj + 1) * tile, s)
            found = 0
            for i in range(i0, i1):
                pi = perm[i]
                row = mask[pi]
                for j in range(j0, j1):
                    pj = perm[j]
                    if row[pj] == 1:
                        found = 1
                        break
                if found:
                    break
            occ[ti][tj] = found
            count += found
    return count, occ


def render_tile_occ(occ: list[list[int]]) -> str:
    return "\n".join("".join("#" if v else "." for v in row) for row in occ)


def make_groups(s: int, group_size: int) -> list[list[int]]:
    groups: list[list[int]] = []
    i = 0
    while i < s:
        groups.append(list(range(i, min(i + group_size, s))))
        i += group_size
    return groups


def optimize_group_order(
    mask: list[list[int]],
    tile: int,
    group_size: int,
    iters: int,
    seed: int,
) -> tuple[list[int], int, list[list[int]], list[int], int, list[list[int]]]:
    s = len(mask)
    groups = make_groups(s, group_size)
    g = len(groups)
    rng = random.Random(seed)

    order = list(range(g))
    perm = build_perm_from_group_order(groups, order)
    best_score, best_occ = active_tiles(mask, perm, tile)
    best_order = order[:]

    cur_order = order[:]
    cur_score = best_score

    for _ in range(iters):
        candidate = cur_order[:]
        if rng.random() < 0.5:
            # swap
            a, b = rng.sample(range(g), 2)
            candidate[a], candidate[b] = candidate[b], candidate[a]
        else:
            # remove+insert
            a, b = rng.sample(range(g), 2)
            item = candidate.pop(a)
            candidate.insert(b, item)

        cand_perm = build_perm_from_group_order(groups, candidate)
        cand_score, _ = active_tiles(mask, cand_perm, tile)

        # Greedy improvement, with tiny chance to accept equal moves to mix.
        if cand_score < cur_score or (cand_score == cur_score and rng.random() < 0.05):
            cur_order = candidate
            cur_score = cand_score
            if cand_score < best_score:
                best_score = cand_score
                best_order = candidate[:]

    init_perm = build_perm_from_group_order(groups, order)
    init_score, init_occ = active_tiles(mask, init_perm, tile)
    final_perm = build_perm_from_group_order(groups, best_order)
    final_score, final_occ = active_tiles(mask, final_perm, tile)
    return init_perm, init_score, init_occ, final_perm, final_score, final_occ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-sample-log", type=Path, required=True)
    parser.add_argument(
        "--section",
        type=str,
        default="Outbound attention mask",
        help="Section to optimize (e.g. Outbound attention mask, Inbound..., etc).",
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Contiguous token chunk size used as permutation unit.",
    )
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    mask = load_matrix_from_single_sample_log(args.single_sample_log, args.section)
    s = len(mask)
    if s != len(mask[0]):
        raise ValueError("Mask must be square.")

    init_perm, init_score, init_occ, final_perm, final_score, final_occ = optimize_group_order(
        mask=mask,
        tile=max(1, args.tile_size),
        group_size=max(1, args.group_size),
        iters=max(1, args.iters),
        seed=args.seed,
    )

    ntiles = len(init_occ)
    print(f"Section: {args.section}")
    print(f"Matrix size: {s}x{s}")
    print(
        f"Tile size: {args.tile_size}, group size: {args.group_size}, "
        f"iterations: {args.iters}, seed: {args.seed}"
    )
    print(f"Tile grid: {ntiles}x{ntiles}")
    print(f"Active tiles before: {init_score}")
    print(f"Active tiles after:  {final_score}")
    print(f"Delta: {init_score - final_score}")

    print("\nTile occupancy BEFORE (# active, . inactive):")
    print(render_tile_occ(init_occ))
    print("\nTile occupancy AFTER (# active, . inactive):")
    print(render_tile_occ(final_occ))

    # Print first 64 permutation entries so it's easy to inspect.
    preview = min(64, len(final_perm))
    print(f"\nOptimized permutation preview (first {preview} indices):")
    print(final_perm[:preview])


if __name__ == "__main__":
    main()
