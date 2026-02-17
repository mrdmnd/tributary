"""Smoke test: run the relational transformer training loop on synthetic data.

Bypasses the headwater sampler to verify the full JAX pipeline works:
model init -> forward pass -> loss -> backward pass -> optimizer step.
"""

import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax

from confluence.config import ModelConfig, TrainingConfig
from confluence.model import RelationalTransformer
from confluence.loss import compute_loss
from confluence.optimizer import create_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def make_dummy_batch(rng, batch_size, seq_len, max_rows, d_text, num_cols=32):
    """Generate a synthetic batch with realistic shapes and varied targets."""
    b, s, r = batch_size, seq_len, max_rows

    # Random semantic types (1=Numerical mostly, with some 2,3,4 mixed in)
    semantic_types = rng.choice([1, 2, 3, 4], size=(b, s)).astype(np.int8)

    # Column IDs cycling through num_cols
    column_ids = np.tile(np.arange(s) % num_cols, (b, 1)).astype(np.int32)

    # Row IDs: simple grouping (every ~10 cells same row)
    seq_row_ids = np.clip(
        np.tile(np.arange(s) // 10, (b, 1)), 0, r - 1
    ).astype(np.uint16)

    # Values
    numeric_values = rng.standard_normal((b, s)).astype(np.float32)
    timestamp_values = rng.standard_normal((b, s, 15)).astype(np.float32)
    bool_values = rng.integers(0, 2, size=(b, s)).astype(np.uint8)
    categorical_embed_ids = rng.integers(0, 10, size=(b, s)).astype(np.uint32)
    text_embed_ids = np.zeros((b, s), dtype=np.uint32)

    # Null / target / padding masks
    is_null = rng.choice([0, 1], size=(b, s), p=[0.9, 0.1]).astype(np.uint8)
    is_padding = np.zeros((b, s), dtype=np.uint8)
    is_padding[:, -s // 4:] = 1  # last quarter is padding

    # Mark some positions as targets (about 5% of non-padding)
    is_target = np.zeros((b, s), dtype=np.uint8)
    for i in range(b):
        non_pad = np.where(is_padding[i] == 0)[0]
        n_targets = max(1, len(non_pad) // 20)
        target_idx = rng.choice(non_pad, size=n_targets, replace=False)
        is_target[i, target_idx] = 1

    # FK adjacency: identity (each row connects to itself only)
    fk_adj = np.eye(r, dtype=np.uint8)[None, :, :].repeat(b, axis=0)

    # Permutations: identity
    identity_perm = np.arange(s, dtype=np.uint16)[None, :].repeat(b, axis=0)

    # Text embeddings (just one dummy entry)
    text_batch_embeddings = np.zeros((1, d_text), dtype=np.uint16)

    # Pick a random predictable target stype (Numerical=1)
    target_stype = np.array([1], dtype=np.uint8)
    task_idx = np.array([0], dtype=np.uint32)

    return {
        "semantic_types": semantic_types,
        "column_ids": column_ids,
        "seq_row_ids": seq_row_ids,
        "numeric_values": numeric_values,
        "timestamp_values": timestamp_values,
        "bool_values": bool_values,
        "categorical_embed_ids": categorical_embed_ids,
        "text_embed_ids": text_embed_ids,
        "is_null": is_null,
        "is_target": is_target,
        "is_padding": is_padding,
        "fk_adj": fk_adj,
        "col_perm": identity_perm.copy(),
        "out_perm": identity_perm.copy(),
        "in_perm": identity_perm.copy(),
        "text_batch_embeddings": text_batch_embeddings,
        "target_stype": target_stype,
        "task_idx": task_idx,
    }


def numpy_batch_to_jax(np_batch, device):
    """Convert a numpy batch dict to JAX arrays on the given device."""
    return {k: jax.device_put(jnp.array(v), device) for k, v in np_batch.items()}


def make_categorical_encoder_fn(params):
    """Create a function that applies the categorical encoder to embeddings."""
    def categorical_encoder_fn(cat_embs):
        encoder_params = params["params"]["value_encoder"]["categorical_encoder"]
        return cat_embs @ encoder_params["kernel"] + encoder_params["bias"]
    return categorical_encoder_fn


def main():
    device = jax.local_devices()[0]
    logger.info(f"Device: {device}")
    logger.info(f"JAX backend: {jax.default_backend()}")

    # Small config for smoke test
    model_config = ModelConfig(
        d_model=128,
        d_text=128,
        d_ff=384,
        n_layers=2,
        n_heads=4,
        d_head=32,
        max_seq_len=64,
        max_rows=16,
    )
    training_config = TrainingConfig(
        batch_size=2,
        sequence_length=64,
        num_steps=30,
        warmup_steps=5,
        eval_interval=100,
        num_val_steps=0,
        muon_lr_peak=0.002,
        adamw_lr_peak=3e-4,
    )

    rng_np = np.random.default_rng(42)

    # Fake embedding tables (small random values, not zeros)
    num_cols = 32
    num_cats = 16
    rng_emb = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(rng_emb)
    col_emb_table = jax.device_put(
        jax.random.normal(k1, (num_cols, model_config.d_text), dtype=jnp.bfloat16) * 0.02,
        device,
    )
    cat_emb_table = jax.device_put(
        jax.random.normal(k2, (num_cats, model_config.d_text), dtype=jnp.bfloat16) * 0.02,
        device,
    )

    # Initialize model
    model = RelationalTransformer(config=model_config)
    rng_key = jax.random.PRNGKey(42)

    logger.info("Creating init batch...")
    init_batch_np = make_dummy_batch(
        rng_np,
        training_config.batch_size,
        training_config.sequence_length,
        model_config.max_rows,
        model_config.d_text,
        num_cols=num_cols,
    )
    init_batch = numpy_batch_to_jax(init_batch_np, device)

    logger.info("Initializing model parameters...")
    params = model.init(rng_key, init_batch, col_emb_table, cat_emb_table)

    num_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"Model parameters: {num_params:,}")

    optimizer = create_optimizer(training_config)
    opt_state = optimizer.init(params)
    logger.info("Optimizer initialized (Muon + AdamW).")

    @jax.jit
    def train_step(params, opt_state, batch, col_emb, cat_emb):
        def loss_fn(params):
            output = model.apply(params, batch, col_emb, cat_emb)
            cat_enc_fn = make_categorical_encoder_fn(params)
            return compute_loss(
                output, batch, cat_emb, cat_enc_fn,
                z_loss_weight=training_config.z_loss_weight,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = jax.tree.map(lambda p, u: p + u, params, updates)
        return new_params, new_opt_state, loss

    # Training loop
    num_steps = training_config.num_steps
    logger.info(f"Starting smoke test for {num_steps} steps...")
    logger.info("-" * 60)

    step_times = []
    for step in range(num_steps):
        np_batch = make_dummy_batch(
            rng_np,
            training_config.batch_size,
            training_config.sequence_length,
            model_config.max_rows,
            model_config.d_text,
            num_cols=num_cols,
        )
        batch = numpy_batch_to_jax(np_batch, device)

        t0 = time.perf_counter()
        params, opt_state, loss = train_step(
            params, opt_state, batch, col_emb_table, cat_emb_table
        )
        loss_val = float(loss)
        dt = time.perf_counter() - t0
        step_times.append(dt)

        avg_dt = sum(step_times[-10:]) / len(step_times[-10:])
        logger.info(
            f"step {step:4d}/{num_steps} | loss {loss_val:8.4f} | "
            f"{dt:.3f}s/step | avg {avg_dt:.3f}s/step"
        )

    logger.info("-" * 60)
    logger.info(
        f"Smoke test complete. Final loss: {loss_val:.4f}  |  "
        f"Avg step time (last 10): {avg_dt:.3f}s"
    )


if __name__ == "__main__":
    main()
