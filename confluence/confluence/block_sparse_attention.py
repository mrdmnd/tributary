"""Prototype block-sparse attention over exact active tiles.

This module is intentionally standalone and test-oriented:
- it accepts active tile coords + 64x64 bitmap payloads
- it provides a dense reference path for correctness checks
- it provides a Pallas kernel path for experimentation
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl


def pack_dense_mask_to_tiles(mask: jnp.ndarray, tile_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pack a dense [S, S] bool mask into active tile coords + 64-row u64 bitmaps."""
    s = int(mask.shape[0])
    num_tiles = math.ceil(s / tile_size)
    coords: list[list[int]] = []
    bitmaps: list[list[int]] = []

    mask_np = jax.device_get(mask).astype(bool)
    for ti in range(num_tiles):
        for tj in range(num_tiles):
            row0 = ti * tile_size
            col0 = tj * tile_size
            rows = [0] * 64
            has_any = False
            for r in range(tile_size):
                rr = row0 + r
                if rr >= s:
                    break
                bits = 0
                for c in range(tile_size):
                    cc = col0 + c
                    if cc >= s:
                        break
                    if mask_np[rr, cc]:
                        bits |= 1 << c
                rows[r] = bits
                has_any = has_any or (bits != 0)
            if has_any:
                coords.append([ti, tj])
                bitmaps.append(rows)

    if not coords:
        return (
            jnp.zeros((0, 2), dtype=jnp.uint16),
            jnp.zeros((0, 64), dtype=jnp.uint64),
        )
    return jnp.array(coords, dtype=jnp.uint16), jnp.array(bitmaps, dtype=jnp.uint64)


def decode_tiles_to_dense_mask(
    tile_coords: jnp.ndarray,
    tile_bitmaps: jnp.ndarray,
    sequence_length: int,
    tile_size: int,
) -> jnp.ndarray:
    """Decode active tile payloads into dense [S, S] bool mask."""
    dense = jnp.zeros((sequence_length, sequence_length), dtype=jnp.bool_)
    for tile_idx in range(int(tile_coords.shape[0])):
        ti = int(tile_coords[tile_idx, 0])
        tj = int(tile_coords[tile_idx, 1])
        row0 = ti * tile_size
        col0 = tj * tile_size
        for r in range(tile_size):
            rr = row0 + r
            if rr >= sequence_length:
                break
            bits = int(tile_bitmaps[tile_idx, r])
            if bits == 0:
                continue
            for c in range(tile_size):
                cc = col0 + c
                if cc >= sequence_length:
                    break
                if ((bits >> c) & 1) == 1:
                    dense = dense.at[rr, cc].set(True)
    return dense


def dense_masked_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Reference dense attention for one [S, D] sequence/head."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = q @ k.T * scale
    logits = jnp.where(mask, logits, -1e9)
    probs = jax.nn.softmax(logits, axis=-1)
    return probs @ v


def tile_sparse_attention_reference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    tile_coords: jnp.ndarray,
    tile_bitmaps: jnp.ndarray,
    tile_size: int,
) -> jnp.ndarray:
    """Reference path: decode sparse tiles then run dense masked attention."""
    mask = decode_tiles_to_dense_mask(tile_coords, tile_bitmaps, q.shape[0], tile_size)
    return dense_masked_attention(q, k, v, mask)


def _tile_sparse_attention_kernel(
    q_ref,
    k_ref,
    v_ref,
    coords_ref,
    bitmaps_ref,
    out_ref,
    *,
    tile_size: int,
):
    """Single-program Pallas kernel for one [S, D] sequence/head."""
    s, d = out_ref.shape
    n_tiles = coords_ref.shape[0]
    scale = 1.0 / jnp.sqrt(jnp.asarray(d, dtype=q_ref.dtype))

    def q_body(qi, _):
        q_vec = q_ref[qi, :]
        q_tile = qi // tile_size
        q_row_in_tile = qi % tile_size

        def pass1_body(ti, row_max):
            tile_row = coords_ref[ti, 0].astype(jnp.int32)
            tile_col = coords_ref[ti, 1].astype(jnp.int32)

            def active_tile(m):
                k_start = tile_col * tile_size
                k_idx = k_start + jnp.arange(tile_size, dtype=jnp.int32)
                k_idx_safe = jnp.clip(k_idx, 0, s - 1)
                k_block = k_ref[k_idx_safe, :]
                scores = jnp.dot(k_block, q_vec) * scale
                row_bits = bitmaps_ref[ti, q_row_in_tile].astype(jnp.uint64)
                valid_bits = ((row_bits >> jnp.arange(tile_size, dtype=jnp.uint64)) & jnp.uint64(1)) == 1
                valid = valid_bits & (k_idx < s)
                masked = jnp.where(valid, scores, -jnp.inf)
                return jnp.maximum(m, jnp.max(masked))

            return lax.cond(tile_row == q_tile, active_tile, lambda m: m, row_max)

        row_max = lax.fori_loop(0, n_tiles, pass1_body, jnp.array(-jnp.inf, dtype=q_ref.dtype))

        def pass2_body(ti, carry):
            denom, acc = carry
            tile_row = coords_ref[ti, 0].astype(jnp.int32)
            tile_col = coords_ref[ti, 1].astype(jnp.int32)

            def active_tile(c):
                dsum, davg = c
                k_start = tile_col * tile_size
                k_idx = k_start + jnp.arange(tile_size, dtype=jnp.int32)
                k_idx_safe = jnp.clip(k_idx, 0, s - 1)
                k_block = k_ref[k_idx_safe, :]
                v_block = v_ref[k_idx_safe, :]
                scores = jnp.dot(k_block, q_vec) * scale
                row_bits = bitmaps_ref[ti, q_row_in_tile].astype(jnp.uint64)
                valid_bits = ((row_bits >> jnp.arange(tile_size, dtype=jnp.uint64)) & jnp.uint64(1)) == 1
                valid = valid_bits & (k_idx < s)
                weights = jnp.where(valid, jnp.exp(scores - row_max), 0.0)
                dsum = dsum + jnp.sum(weights)
                davg = davg + (weights[:, None] * v_block).sum(axis=0)
                return dsum, davg

            return lax.cond(tile_row == q_tile, active_tile, lambda c: c, (denom, acc))

        denom0 = jnp.array(0.0, dtype=q_ref.dtype)
        acc0 = jnp.zeros((d,), dtype=q_ref.dtype)
        denom, acc = lax.fori_loop(0, n_tiles, pass2_body, (denom0, acc0))
        out_row = jnp.where(denom > 0, acc / denom, jnp.zeros_like(acc))
        out_ref[qi, :] = out_row
        return None

    lax.fori_loop(0, s, q_body, None)


def tile_sparse_attention_pallas(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    tile_coords: jnp.ndarray,
    tile_bitmaps: jnp.ndarray,
    tile_size: int,
    *,
    interpret: bool = True,
) -> jnp.ndarray:
    """Pallas prototype kernel over active tiles for one [S, D] sequence/head."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        msg = "q, k, v must be rank-2 [S, D] tensors"
        raise ValueError(msg)
    out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    kernel = lambda q_ref, k_ref, v_ref, coords_ref, bitmaps_ref, out_ref: _tile_sparse_attention_kernel(  # noqa: E731
        q_ref,
        k_ref,
        v_ref,
        coords_ref,
        bitmaps_ref,
        out_ref,
        tile_size=tile_size,
    )
    return pl.pallas_call(kernel, out_shape=out_shape, interpret=interpret)(q, k, v, tile_coords, tile_bitmaps)

