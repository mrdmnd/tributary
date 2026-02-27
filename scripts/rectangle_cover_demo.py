#!/usr/bin/env python3
"""Minimum rectangle cover demo for binary attention masks.

This script demonstrates an exact minimum rectangle cover on a *compressed*
matrix (typically a tiled mask grid, e.g. 8x8 from a 256x256 mask at tile=32).

Algorithm outline:
1) Load a binary matrix (from a text file, or from a single_sample log section).
2) Optionally compress to tiles (active tile if density > threshold).
3) Enumerate all all-ones axis-aligned rectangles in the compressed matrix.
4) Solve exact minimum cover of all 1-cells using branch-and-bound set cover.
5) Print rectangle corner coordinates and ASCII visualizations.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.text import Text


@dataclass(frozen=True)
class Rect:
    r0: int
    r1: int
    c0: int
    c1: int

    def corners(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.r0, self.c0), (self.r1, self.c1)


def parse_binary_lines(lines: list[str]) -> list[list[int]]:
    matrix: list[list[int]] = []
    width = None
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


def load_matrix_from_file(path: Path) -> list[list[int]]:
    return parse_binary_lines(path.read_text().splitlines())


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
        # matrix lines in logs are dense #/.; skip unrelated lines
        if any(ch in stripped for ch in ("#", ".")):
            block.append(stripped)
        else:
            break
    if not block:
        raise ValueError(f"No matrix rows after section: {section_name!r}")
    return parse_binary_lines(block)


def tile_compress(matrix: list[list[int]], tile: int, min_density: float) -> list[list[int]]:
    h = len(matrix)
    w = len(matrix[0])
    th = math.ceil(h / tile)
    tw = math.ceil(w / tile)
    out = [[0 for _ in range(tw)] for _ in range(th)]
    for ti in range(th):
        for tj in range(tw):
            r0, r1 = ti * tile, min((ti + 1) * tile, h)
            c0, c1 = tj * tile, min((tj + 1) * tile, w)
            total = (r1 - r0) * (c1 - c0)
            ones = 0
            for r in range(r0, r1):
                row = matrix[r]
                for c in range(c0, c1):
                    ones += row[c]
            density = ones / total if total else 0.0
            out[ti][tj] = 1 if density > min_density else 0
    return out


def enumerate_all_ones_rectangles(grid: list[list[int]]) -> list[Rect]:
    h = len(grid)
    w = len(grid[0])
    rects: list[Rect] = []
    for r0 in range(h):
        col_ok = [True] * w
        for r1 in range(r0, h):
            for c in range(w):
                col_ok[c] = col_ok[c] and (grid[r1][c] == 1)
            c = 0
            while c < w:
                if not col_ok[c]:
                    c += 1
                    continue
                start = c
                while c < w and col_ok[c]:
                    c += 1
                end = c - 1
                for c0 in range(start, end + 1):
                    for c1 in range(c0, end + 1):
                        rects.append(Rect(r0, r1, c0, c1))
    return rects


def bit_count(x: int) -> int:
    return x.bit_count()


def minimum_rectangle_cover(grid: list[list[int]]) -> list[Rect]:
    h = len(grid)
    w = len(grid[0])

    # Map each 1-cell to a bit index.
    cell_to_bit: dict[tuple[int, int], int] = {}
    bit = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1:
                cell_to_bit[(r, c)] = bit
                bit += 1
    num_ones = bit
    if num_ones == 0:
        return []

    universe = (1 << num_ones) - 1
    rects = enumerate_all_ones_rectangles(grid)

    rect_masks: list[int] = []
    for rect in rects:
        mask = 0
        for r in range(rect.r0, rect.r1 + 1):
            for c in range(rect.c0, rect.c1 + 1):
                if grid[r][c] == 1:
                    mask |= 1 << cell_to_bit[(r, c)]
        if mask:
            rect_masks.append(mask)
        else:
            rect_masks.append(0)

    # Coverage lists per cell bit.
    covers_by_bit: list[list[int]] = [[] for _ in range(num_ones)]
    for ridx, mask in enumerate(rect_masks):
        m = mask
        while m:
            lsb = m & -m
            b = lsb.bit_length() - 1
            covers_by_bit[b].append(ridx)
            m ^= lsb

    # Sort each list to prioritize bigger rectangles first (helps pruning).
    rect_size = [bit_count(m) for m in rect_masks]
    for lst in covers_by_bit:
        lst.sort(key=lambda ridx: rect_size[ridx], reverse=True)

    best_solution: list[int] | None = None

    def dfs(uncovered: int, chosen: list[int]) -> None:
        nonlocal best_solution
        if uncovered == 0:
            if best_solution is None or len(chosen) < len(best_solution):
                best_solution = chosen.copy()
            return

        if best_solution is not None and len(chosen) >= len(best_solution):
            return

        # Lower bound: at least ceil(remaining / max_new_cover) rectangles.
        remaining = bit_count(uncovered)
        max_new_cover = 0
        for ridx, mask in enumerate(rect_masks):
            new_cover = bit_count(mask & uncovered)
            if new_cover > max_new_cover:
                max_new_cover = new_cover
        if max_new_cover == 0:
            return
        lb = math.ceil(remaining / max_new_cover)
        if best_solution is not None and len(chosen) + lb >= len(best_solution):
            return

        # Pick uncovered bit with fewest covering rectangles.
        m = uncovered
        target_bit = None
        best_len = None
        while m:
            lsb = m & -m
            b = lsb.bit_length() - 1
            l = len(covers_by_bit[b])
            if best_len is None or l < best_len:
                best_len = l
                target_bit = b
                if l == 1:
                    break
            m ^= lsb
        assert target_bit is not None

        for ridx in covers_by_bit[target_bit]:
            new_uncovered = uncovered & ~rect_masks[ridx]
            if new_uncovered == uncovered:
                continue
            chosen.append(ridx)
            dfs(new_uncovered, chosen)
            chosen.pop()

    dfs(universe, [])
    assert best_solution is not None
    return [rects[i] for i in best_solution]


def render_grid(grid: list[list[int]]) -> str:
    return "\n".join("".join("#" if v else "." for v in row) for row in grid)


def render_cover(grid: list[list[int]], rects: list[Rect]) -> str:
    h = len(grid)
    w = len(grid[0])
    covered = [[0 for _ in range(w)] for _ in range(h)]
    for rect in rects:
        for r in range(rect.r0, rect.r1 + 1):
            for c in range(rect.c0, rect.c1 + 1):
                covered[r][c] = 1
    return render_grid(covered)


def rect_palette() -> list[str]:
    return [
        "bright_cyan",
        "bright_magenta",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_red",
        "cyan",
        "magenta",
        "green",
        "yellow",
        "blue",
        "red",
    ]


def rect_symbol(idx: int) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return alphabet[idx % len(alphabet)]


def render_rich_cover(console: Console, grid: list[list[int]], rects: list[Rect]) -> None:
    h = len(grid)
    w = len(grid[0])
    palette = rect_palette()
    assign = [[-1 for _ in range(w)] for _ in range(h)]
    overlap = [[0 for _ in range(w)] for _ in range(h)]

    for ridx, rect in enumerate(rects):
        for r in range(rect.r0, rect.r1 + 1):
            for c in range(rect.c0, rect.c1 + 1):
                if grid[r][c] == 0:
                    continue
                overlap[r][c] += 1
                if assign[r][c] == -1:
                    assign[r][c] = ridx

    console.print("\n[bold]Colorized cover view[/bold]")
    console.print("[dim].[/dim]=inactive, [bold red]@[/bold red]=overlap, [bold]0-9A-Za-z[/bold]=rectangle id")
    for r in range(h):
        line = Text()
        for c in range(w):
            if grid[r][c] == 0:
                line.append(".", style="dim")
                continue
            if overlap[r][c] > 1:
                line.append("@", style="bold red")
                continue
            ridx = assign[r][c]
            if ridx < 0:
                line.append("?", style="bold white on red")
                continue
            sym = rect_symbol(ridx)
            style = f"bold {palette[ridx % len(palette)]}"
            line.append(sym, style=style)
        console.print(line)

    console.print("\n[bold]Legend[/bold]")
    for i, rect in enumerate(rects):
        (r0, c0), (r1, c1) = rect.corners()
        style = f"bold {palette[i % len(palette)]}"
        console.print(f"[{style}]{rect_symbol(i)}[/{style}] -> ({r0},{c0}) -> ({r1},{c1})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix-file", type=Path, help="Path to text file with #/. or 1/0 matrix.")
    src.add_argument("--single-sample-log", type=Path, help="Path to single_sample output log file.")
    parser.add_argument(
        "--section",
        type=str,
        default="Outbound attention mask (after out_perm)",
        help="Section header to extract from --single-sample-log.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=32,
        help="Tile size for compression before exact cover. Use 1 for no compression.",
    )
    parser.add_argument(
        "--tile-min-density",
        type=float,
        default=0.0,
        help="Tile active iff density > this threshold.",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich colorized rectangle visualization.",
    )
    args = parser.parse_args()

    if args.matrix_file:
        matrix = load_matrix_from_file(args.matrix_file)
    else:
        matrix = load_matrix_from_single_sample_log(args.single_sample_log, args.section)

    h = len(matrix)
    w = len(matrix[0])
    print(f"Loaded matrix: {h}x{w}")

    tile = max(args.tile_size, 1)
    compressed = tile_compress(matrix, tile, args.tile_min_density) if tile > 1 else matrix
    ch = len(compressed)
    cw = len(compressed[0])
    print(f"Compressed grid: {ch}x{cw} (tile={tile}, min_density>{args.tile_min_density})")
    print("\nCompressed grid (# = active tile/cell):")
    print(render_grid(compressed))

    rects = minimum_rectangle_cover(compressed)
    print(f"\nMinimum rectangle cover size: {len(rects)}")
    print("Rectangles (inclusive corners on compressed grid):")
    for i, r in enumerate(rects):
        (r0, c0), (r1, c1) = r.corners()
        print(f"  [{i:02d}] ({r0},{c0}) -> ({r1},{c1})")

    print("\nCovered reconstruction:")
    print(render_cover(compressed, rects))
    if not args.no_rich:
        render_rich_cover(Console(), compressed, rects)

    if tile > 1:
        print("\nApproximate coordinates on original matrix:")
        for i, r in enumerate(rects):
            orig_r0 = r.r0 * tile
            orig_c0 = r.c0 * tile
            orig_r1 = min((r.r1 + 1) * tile - 1, h - 1)
            orig_c1 = min((r.c1 + 1) * tile - 1, w - 1)
            print(f"  [{i:02d}] ({orig_r0},{orig_c0}) -> ({orig_r1},{orig_c1})")


if __name__ == "__main__":
    main()
