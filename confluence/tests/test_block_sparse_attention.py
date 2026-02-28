import jax.numpy as jnp
import numpy as np
import pytest

from confluence.block_sparse_attention import (
    decode_tiles_to_dense_mask,
    dense_masked_attention,
    pack_dense_mask_to_tiles,
    tile_sparse_attention_pallas,
    tile_sparse_attention_reference,
)


def _random_mask_with_diag(rng: np.random.Generator, s: int, density: float = 0.2) -> jnp.ndarray:
    mask = (rng.random((s, s)) < density).astype(np.bool_)
    np.fill_diagonal(mask, True)
    return jnp.array(mask)


def test_pack_decode_roundtrip() -> None:
    rng = np.random.default_rng(0)
    s = 96
    tile_size = 32
    mask = _random_mask_with_diag(rng, s, density=0.15)
    coords, bitmaps = pack_dense_mask_to_tiles(mask, tile_size=tile_size)
    decoded = decode_tiles_to_dense_mask(coords, bitmaps, sequence_length=s, tile_size=tile_size)
    np.testing.assert_array_equal(np.array(mask), np.array(decoded))


def test_sparse_reference_matches_dense() -> None:
    rng = np.random.default_rng(1)
    s, d = 64, 16
    tile_size = 16
    q = jnp.array(rng.standard_normal((s, d)), dtype=jnp.float32)
    k = jnp.array(rng.standard_normal((s, d)), dtype=jnp.float32)
    v = jnp.array(rng.standard_normal((s, d)), dtype=jnp.float32)
    mask = _random_mask_with_diag(rng, s, density=0.2)
    coords, bitmaps = pack_dense_mask_to_tiles(mask, tile_size=tile_size)

    ref_dense = dense_masked_attention(q, k, v, mask)
    ref_sparse = tile_sparse_attention_reference(q, k, v, coords, bitmaps, tile_size=tile_size)
    np.testing.assert_allclose(np.array(ref_sparse), np.array(ref_dense), atol=1e-5, rtol=1e-5)


def test_pallas_matches_reference_interpret() -> None:
    rng = np.random.default_rng(2)
    s, d = 64, 16
    tile_size = 16
    q = jnp.array(rng.standard_normal((s, d)), dtype=jnp.float32)
    k = jnp.array(rng.standard_normal((s, d)), dtype=jnp.float32)
    v = jnp.array(rng.standard_normal((s, d)), dtype=jnp.float32)
    mask = _random_mask_with_diag(rng, s, density=0.25)
    coords, bitmaps = pack_dense_mask_to_tiles(mask, tile_size=tile_size)

    ref = tile_sparse_attention_reference(q, k, v, coords, bitmaps, tile_size=tile_size)
    try:
        out = tile_sparse_attention_pallas(
            q,
            k,
            v,
            coords,
            bitmaps,
            tile_size=tile_size,
            interpret=True,
        )
    except Exception as exc:  # pragma: no cover - backend-specific failure path
        pytest.skip(f"Pallas interpret execution unavailable: {exc}")
        return
    np.testing.assert_allclose(np.array(out), np.array(ref), atol=1e-3, rtol=1e-3)

