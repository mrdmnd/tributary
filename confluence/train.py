"""Distributed training loop for the relational transformer.

Supports single-GPU and multi-GPU (DDP) training via JAX's distributed runtime.
Uses the headwater Rust sampler for efficient batch generation.
"""

import argparse
import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import headwater
from confluence.config import ModelConfig, TrainingConfig
from confluence.loss import compute_loss
from confluence.model import RelationalTransformer
from confluence.optimizer import create_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def numpy_batch_to_jax(np_batch: dict[str, np.ndarray], device: jax.Device) -> dict[str, jax.Array]:
    """Convert a numpy batch dict to JAX arrays on the given device.

    Passes the full pytree in a single ``device_put`` so JAX can batch
    the host-to-device transfers instead of issuing one per array.
    """
    return jax.device_put(np_batch, device)


def numpy_batch_to_pmap(
    np_batch: dict[str, np.ndarray],
) -> dict[str, jax.Array]:
    """Convert a numpy batch dict to pmap-ready arrays (leading local-device axis).

    Uses numpy for the expand then a single ``device_put`` on the pytree.
    """
    expanded = {k: np.expand_dims(v, axis=0) for k, v in np_batch.items()}
    return jax.device_put(expanded)


def create_dummy_batch(config: ModelConfig, training_config: TrainingConfig):
    """Create a dummy batch for model initialization."""
    b = training_config.batch_size
    s = training_config.sequence_length
    d_t = config.d_text
    row_ptr = np.zeros((b, s + 1), dtype=np.uint32)
    seq_offsets = np.zeros((b + 1,), dtype=np.uint32)
    col_idx = np.zeros((0,), dtype=np.uint16)

    return {
        "semantic_types": np.zeros((b, s), dtype=np.uint8),
        "column_embed_ids": np.zeros((b, s), dtype=np.uint32),
        "numeric_values": np.zeros((b, s), dtype=np.float32),
        "timestamp_values": np.zeros((b, s, 15), dtype=np.float32),
        "bool_values": np.zeros((b, s), dtype=np.uint8),
        "categorical_embed_ids": np.zeros((b, s), dtype=np.uint32),
        "text_embed_ids": np.zeros((b, s), dtype=np.uint32),
        "is_null": np.zeros((b, s), dtype=np.uint8),
        "is_padding": np.ones((b, s), dtype=np.uint8),
        "outbound_csr_row_ptr": row_ptr.copy(),
        "outbound_csr_seq_offsets": seq_offsets.copy(),
        "outbound_csr_col_idx": col_idx.copy(),
        "inbound_csr_row_ptr": row_ptr.copy(),
        "inbound_csr_seq_offsets": seq_offsets.copy(),
        "inbound_csr_col_idx": col_idx.copy(),
        "column_csr_row_ptr": row_ptr.copy(),
        "column_csr_seq_offsets": seq_offsets.copy(),
        "column_csr_col_idx": col_idx.copy(),
        "text_batch_embeddings": np.zeros((1, d_t), dtype=np.uint16),
        "target_stype": np.array([1], dtype=np.uint8),
        "task_idx": np.array([0], dtype=np.uint32),
    }


def make_categorical_encoder_fn(params, model):
    """Create a function that applies the categorical encoder to embeddings.

    This is used in the loss computation to project category embeddings
    through the same Linear(D_t -> D) used in value encoding.
    """

    def categorical_encoder_fn(cat_embs):
        # Access the categorical_encoder kernel and bias from params
        encoder_params = params["params"]["value_encoder"]["categorical_encoder"]
        kernel = encoder_params["kernel"]
        bias = encoder_params["bias"]
        return cat_embs @ kernel + bias

    return categorical_encoder_fn


def run_validation(sampler, model, params, col_emb_table, cat_emb_table, device, num_val_steps):
    """Run validation and return average loss."""
    total_loss = 0.0
    for _ in range(num_val_steps):
        np_batch = sampler.next_val_batch()
        batch = numpy_batch_to_jax(np_batch, device)

        output = model.apply(params, batch, col_emb_table, cat_emb_table)
        cat_enc_fn = make_categorical_encoder_fn(params, model)
        loss = compute_loss(output, batch, cat_emb_table, cat_enc_fn)
        total_loss += float(loss)

    return total_loss / max(num_val_steps, 1)


def main():
    parser = argparse.ArgumentParser(description="Train the relational transformer")
    parser.add_argument("--db-path", type=str, required=True, help="Path to preprocessed database directory")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--eval-interval", type=int, default=100, help="Steps between validation runs")
    parser.add_argument("--num-val-steps", type=int, default=10, help="Number of validation steps per eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- JAX Distributed Setup ---
    # Initialize distributed runtime (reads SLURM / coordinator env vars)
    # For single-GPU, this is a no-op.
    try:
        jax.distributed.initialize()
    except Exception:
        pass  # Single-process mode

    rank = jax.process_index()
    world_size = jax.process_count()
    local_devices = jax.local_devices()
    local_device_count = len(local_devices)
    device = local_devices[0]

    logger.info(f"Process {rank}/{world_size} using device: {device}")
    if world_size > 1 and local_device_count != 1:
        raise RuntimeError("This training loop currently expects one process per GPU (local_device_count must be 1 when world_size > 1).")

    # --- Configuration ---
    model_config = ModelConfig(
        n_layers=args.n_layers,
        max_seq_len=args.seq_length,
    )
    training_config = TrainingConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        eval_interval=args.eval_interval,
        num_val_steps=args.num_val_steps,
        seed=args.seed,
    )

    # --- Initialize Sampler ---
    logger.info("Initializing sampler...")
    sampler = headwater.Sampler(
        db_path=args.db_path,
        rank=rank,
        world_size=world_size,
        split_ratios=training_config.split_ratios,
        split_seed=training_config.split_seed,
        seed=training_config.seed,
        num_prefetch=training_config.num_prefetch,
        default_batch_size=training_config.batch_size,
        default_sequence_length=training_config.sequence_length,
        bfs_child_width=training_config.bfs_child_width,
    )
    logger.info("Sampler initialized.")

    # --- Upload GPU-Resident Tables ---
    col_emb_raw = sampler.column_embeddings()  # [C, D_t] as uint16 (f16 bits)
    cat_emb_raw = sampler.categorical_embeddings()  # [Vc, D_t] as uint16

    # Convert f16 bits to bfloat16 via float16 intermediate
    col_emb_table = jax.device_put(jnp.array(col_emb_raw, dtype=jnp.float16).astype(jnp.bfloat16), device)
    cat_emb_table = jax.device_put(jnp.array(cat_emb_raw, dtype=jnp.float16).astype(jnp.bfloat16), device)
    logger.info(f"GPU tables uploaded: col_emb={col_emb_table.shape}, cat_emb={cat_emb_table.shape}")

    # --- Initialize Model ---
    model = RelationalTransformer(config=model_config)
    rng = jax.random.PRNGKey(args.seed)

    dummy_batch = create_dummy_batch(model_config, training_config)
    dummy_batch_jax = numpy_batch_to_jax(dummy_batch, device)

    logger.info("Initializing model parameters...")
    params = model.init(rng, dummy_batch_jax, col_emb_table, cat_emb_table)

    # Count parameters
    num_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"Model parameters: {num_params:,}")

    # --- Initialize Optimizer ---
    optimizer = create_optimizer(training_config)
    opt_state = optimizer.init(params)
    logger.info("Optimizer initialized.")

    # --- JIT-compiled single-process Training Step ---
    @jax.jit
    def train_step_single(params, opt_state, batch, col_emb, cat_emb):
        def loss_fn(params):
            output = model.apply(params, batch, col_emb, cat_emb)
            cat_enc_fn = make_categorical_encoder_fn(params, model)
            return compute_loss(
                output,
                batch,
                cat_emb,
                cat_enc_fn,
                z_loss_weight=training_config.z_loss_weight,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax_apply_updates(params, updates)

        return new_params, new_opt_state, loss

    # --- PMAP training step for one-process-per-GPU distributed runs ---
    @partial(jax.pmap, axis_name="devices")
    def train_step_pmapped(params, opt_state, batch, col_emb, cat_emb):
        def loss_fn(params):
            output = model.apply(params, batch, col_emb, cat_emb)
            cat_enc_fn = make_categorical_encoder_fn(params, model)
            return compute_loss(
                output,
                batch,
                cat_emb,
                cat_enc_fn,
                z_loss_weight=training_config.z_loss_weight,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax_apply_updates(params, updates)

        return new_params, new_opt_state, loss

    if world_size > 1:
        params = jax.device_put_replicated(params, local_devices)
        opt_state = jax.device_put_replicated(opt_state, local_devices)
        col_emb_table = jax.device_put_replicated(col_emb_table, local_devices)
        cat_emb_table = jax.device_put_replicated(cat_emb_table, local_devices)

    # --- Training Loop ---
    logger.info(f"Starting training for {args.num_steps} steps...")
    step_times = []

    for step in range(args.num_steps):
        t0 = time.perf_counter()

        # Pull batch from Rust sampler (GIL released during wait)
        np_batch = sampler.next_train_batch()
        if world_size > 1:
            batch = numpy_batch_to_pmap(np_batch)
            params, opt_state, loss = train_step_pmapped(params, opt_state, batch, col_emb_table, cat_emb_table)
            loss_val = float(jax.device_get(loss)[0])
        else:
            batch = numpy_batch_to_jax(np_batch, device)
            params, opt_state, loss = train_step_single(params, opt_state, batch, col_emb_table, cat_emb_table)
            # Block on loss for logging (forces sync)
            loss_val = float(loss)
        dt = time.perf_counter() - t0
        step_times.append(dt)

        if step % 10 == 0:
            avg_dt = sum(step_times[-10:]) / len(step_times[-10:])
            logger.info(f"step {step:6d} | loss {loss_val:.4f} | {dt:.3f}s/step | avg {avg_dt:.3f}s/step")

        # Periodic validation
        if step > 0 and step % args.eval_interval == 0:
            eval_params = params
            eval_col_emb = col_emb_table
            eval_cat_emb = cat_emb_table
            if world_size > 1:
                eval_params = jax.tree.map(lambda x: x[0], params)
                eval_col_emb = col_emb_table[0]
                eval_cat_emb = cat_emb_table[0]
            val_loss = run_validation(
                sampler,
                model,
                eval_params,
                eval_col_emb,
                eval_cat_emb,
                device,
                args.num_val_steps,
            )
            logger.info(f"  val_loss: {val_loss:.4f}")

    # --- Final Validation ---
    logger.info("Running final validation...")
    eval_params = params
    eval_col_emb = col_emb_table
    eval_cat_emb = cat_emb_table
    if world_size > 1:
        eval_params = jax.tree.map(lambda x: x[0], params)
        eval_col_emb = col_emb_table[0]
        eval_cat_emb = cat_emb_table[0]
    val_loss = run_validation(
        sampler,
        model,
        eval_params,
        eval_col_emb,
        eval_cat_emb,
        device,
        training_config.num_val_steps,
    )
    logger.info(f"Final val_loss: {val_loss:.4f}")

    # --- Cleanup ---
    sampler.shutdown()
    logger.info("Training complete.")


def optax_apply_updates(params, updates):
    """Apply optax updates to parameters."""
    return jax.tree.map(lambda p, u: p + u, params, updates)


if __name__ == "__main__":
    main()
