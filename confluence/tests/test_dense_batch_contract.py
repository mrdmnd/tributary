import numpy as np

from smoke_test import make_dummy_batch


def test_dummy_batch_uses_csr_attention_masks() -> None:
    batch = make_dummy_batch(
        rng=np.random.default_rng(0),
        batch_size=2,
        seq_len=8,
        max_rows=4,
        d_text=16,
    )

    assert batch["mask_format"].shape == (1,)
    assert batch["mask_format"][0] == 1
    assert batch["outbound_csr_row_ptr"].shape == (2, 9)
    assert batch["outbound_csr_seq_offsets"].shape == (3,)
    assert batch["outbound_csr_col_idx"].ndim == 1
    assert batch["inbound_csr_row_ptr"].shape == (2, 9)
    assert batch["inbound_csr_seq_offsets"].shape == (3,)
    assert batch["inbound_csr_col_idx"].ndim == 1
    assert batch["column_csr_row_ptr"].shape == (2, 9)
    assert batch["column_csr_seq_offsets"].shape == (3,)
    assert batch["column_csr_col_idx"].ndim == 1

    legacy_tile_keys = {
        "attn_tile_size",
        "attn_num_tiles",
        "out_tile_offsets",
        "out_tile_coords",
        "out_tile_bitmaps",
        "in_tile_offsets",
        "in_tile_coords",
        "in_tile_bitmaps",
        "col_tile_offsets",
        "col_tile_coords",
        "col_tile_bitmaps",
    }
    assert legacy_tile_keys.isdisjoint(batch.keys())
