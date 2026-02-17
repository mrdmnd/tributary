"""Split optimizer: Muon for 2D matrices, AdamW for everything else.

Uses optax.contrib.muon which handles Newton-Schulz orthogonalization and
automatic partitioning (2D weights -> Muon, rest -> AdamW) out of the box.
See: https://optax.readthedocs.io/en/stable/_collections/examples/contrib/muon.html

LR schedule: linear warmup + cosine decay to 10% of peak.
"""

import jax.numpy as jnp
import optax

from confluence.config import TrainingConfig


def make_schedule(peak_lr, warmup_steps, total_steps, min_ratio=0.1):
    """Linear warmup + cosine decay to min_ratio * peak_lr."""
    def schedule_fn(count):
        warmup = jnp.minimum(count / jnp.maximum(warmup_steps, 1), 1.0)
        progress = jnp.clip(
            (count - warmup_steps) / jnp.maximum(total_steps - warmup_steps, 1),
            0.0, 1.0,
        )
        decay = min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + jnp.cos(jnp.pi * progress)
        )
        return peak_lr * warmup * decay
    return schedule_fn


def create_optimizer(config: TrainingConfig):
    """Create the split optimizer via optax.contrib.muon.

    By default, optax.contrib.muon applies Muon (momentum + Newton-Schulz
    orthogonalization) to all 2D weight matrices and AdamW to everything else.
    We wrap it with gradient clipping and our LR schedule.
    """
    muon_schedule = make_schedule(
        config.muon_lr_peak,
        config.warmup_steps,
        config.num_steps,
        config.lr_min_ratio,
    )
    adamw_schedule = make_schedule(
        config.adamw_lr_peak,
        config.warmup_steps,
        config.num_steps,
        config.lr_min_ratio,
    )

    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.contrib.muon(
            learning_rate=muon_schedule,
            beta=config.muon_beta1,
            ns_steps=config.muon_ns_iters,
            adam_learning_rate=adamw_schedule,
            adam_b1=config.adamw_beta1,
            adam_b2=config.adamw_beta2,
            adam_weight_decay=config.adamw_weight_decay,
        ),
    )
