"""Core building blocks for the relational transformer.

Implements:
- ZeroCenteredRMSNorm
- QKNormedMultiHeadAttention
- GatedAttentionSublayer (RMSNorm -> permute -> MHA -> unpermute -> gate -> residual)
- SwiGLU FFN
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from confluence.config import ModelConfig


class ZeroCenteredRMSNorm(nn.Module):
    """Zero-centered RMSNorm: (1 + gamma) * x / sqrt(mean(x^2) + eps).

    gamma is initialized to 0 so the layer starts as pure normalization.
    """
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        gamma = self.param(
            "gamma", nn.initializers.zeros_init(), (x.shape[-1],)
        )
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (1.0 + gamma) * x / rms


class QKNormedMultiHeadAttention(nn.Module):
    """Multi-head attention with QK normalization and learned temperature.

    Q and K are L2-normalized after projection. A learnable scalar temperature
    per head scales the dot products. No bias in W_Q, W_K, W_V, W_O.
    """
    config: ModelConfig
    output_scale: float = 1.0  # For scaled residual init (1/sqrt(4*N_layers))

    @nn.compact
    def __call__(self, x, mask):
        """
        Args:
            x: [B, S, D] input hidden states.
            mask: [B, 1, S, S] or [B, S, S] attention mask (True = attend).

        Returns:
            [B, S, D] attention output.
        """
        cfg = self.config
        b, s, d = x.shape

        # Project Q, K, V without bias
        qkv = nn.Dense(3 * d, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape to [B, S, H, D_head] then transpose to [B, H, S, D_head]
        q = q.reshape(b, s, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, cfg.n_heads, cfg.d_head).transpose(0, 2, 1, 3)

        # L2 normalize Q and K
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)

        # Learnable temperature per head, initialized to sqrt(d_head)
        init_tau = float(cfg.d_head ** 0.5)
        tau = self.param(
            "tau",
            lambda rng, shape: jnp.full(shape, init_tau),
            (cfg.n_heads, 1, 1),
        )
        q = q * tau

        # Attention logits: [B, H, S, S]
        attn_logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(cfg.d_head)

        # Apply mask
        if mask.ndim == 3:
            mask = mask[:, None, :, :]  # [B, 1, S, S]
        attn_logits = jnp.where(mask, attn_logits, jnp.finfo(attn_logits.dtype).min)

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        # Weighted sum: [B, H, S, D_head]
        attn_out = jnp.matmul(attn_weights, v)

        # Reshape back: [B, S, D]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b, s, d)

        # Output projection W_O (no bias), with optional scaling
        out = nn.Dense(
            d,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="wo",
        )(attn_out)

        return out * self.output_scale


class GatedAttentionSublayer(nn.Module):
    """One attention sublayer: RMSNorm -> permute -> MHA -> unpermute -> gate -> residual.

    The gate is sigmoid(x_norm @ W_gate) applied elementwise to the attention
    output before the residual connection.
    """
    config: ModelConfig
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, x, mask, perm):
        """
        Args:
            x: [B, S, D] input hidden states.
            mask: [B, S, S] attention mask (True = attend).
            perm: [B, S] permutation indices for block sparsity.

        Returns:
            [B, S, D] output after residual connection.
        """
        cfg = self.config
        b, s, d = x.shape

        # Pre-norm
        x_norm = ZeroCenteredRMSNorm(eps=cfg.rms_norm_eps, name="norm")(x)

        # Gather into permuted order
        perm_expanded = perm[:, :, None].astype(jnp.int32)
        perm_expanded = jnp.broadcast_to(perm_expanded, (b, s, d))
        x_perm = jnp.take_along_axis(x_norm, perm_expanded, axis=1)

        # Permute mask: mask_perm[b, i, j] = mask[b, perm[b, i], perm[b, j]]
        perm_i = perm[:, :, None].astype(jnp.int32)  # [B, S, 1]
        perm_j = perm[:, None, :].astype(jnp.int32)  # [B, 1, S]
        # Gather rows then columns
        mask_perm = jnp.take_along_axis(
            jnp.take_along_axis(mask, jnp.broadcast_to(perm_i, (b, s, s)), axis=1),
            jnp.broadcast_to(perm_j, (b, s, s)),
            axis=2,
        )

        # Attention in permuted space
        attn_out = QKNormedMultiHeadAttention(
            config=cfg, output_scale=self.output_scale, name="mha"
        )(x_perm, mask_perm)

        # Scatter back from permuted order using inverse permutation
        perm_i32 = perm.astype(jnp.int32)
        inv_perm = jnp.zeros((b, s), dtype=jnp.int32)
        batch_idx = jnp.arange(b)[:, None]
        seq_idx = jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32)[None, :], (b, s))
        inv_perm = inv_perm.at[batch_idx, perm_i32].set(seq_idx)
        inv_expanded = inv_perm[:, :, None].astype(jnp.int32)
        inv_expanded = jnp.broadcast_to(inv_expanded, (b, s, d))
        attn_unperm = jnp.take_along_axis(attn_out, inv_expanded, axis=1)

        # Sigmoid gate: gate = sigmoid(x_norm @ W_gate)
        gate = nn.Dense(d, use_bias=False, name="w_gate")(x_norm)
        gate = jax.nn.sigmoid(gate)

        # Gated output + residual
        return x + attn_unperm * gate


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: W_2 * (SiLU(W_gate * x) * (W_up * x)).

    No bias terms. D_ff = 768 for D = 256.
    """
    config: ModelConfig
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        d = cfg.d_model
        d_ff = cfg.d_ff

        # Gate and up projections (no bias)
        gate = nn.Dense(d_ff, use_bias=False, name="w_gate")(x)
        up = nn.Dense(d_ff, use_bias=False, name="w_up")(x)

        # SwiGLU activation
        hidden = jax.nn.silu(gate) * up

        # Down projection (no bias), with optional scaling
        out = nn.Dense(d, use_bias=False, name="w_down")(hidden)
        return out * self.output_scale


class TransformerLayer(nn.Module):
    """One full transformer layer: 3 attention sublayers + FFN.

    Attention order: outbound -> inbound -> column.
    """
    config: ModelConfig
    layer_idx: int = 0

    @nn.compact
    def __call__(self, x, outbound_mask, inbound_mask, column_mask,
                 out_perm, in_perm, col_perm):
        cfg = self.config
        n_layers = cfg.n_layers
        # W_O and W_2 scaling factor for residual variance control
        output_scale = 1.0 / (4.0 * n_layers) ** 0.5

        # Outbound attention (self + FK parents)
        x = GatedAttentionSublayer(
            config=cfg, output_scale=output_scale, name="outbound"
        )(x, outbound_mask, out_perm)

        # Inbound attention (FK children)
        x = GatedAttentionSublayer(
            config=cfg, output_scale=output_scale, name="inbound"
        )(x, inbound_mask, in_perm)

        # Column attention (same-column peers)
        x = GatedAttentionSublayer(
            config=cfg, output_scale=output_scale, name="column"
        )(x, column_mask, col_perm)

        # FFN with pre-norm
        x_norm = ZeroCenteredRMSNorm(eps=cfg.rms_norm_eps, name="ffn_norm")(x)
        x = x + SwiGLUFFN(
            config=cfg, output_scale=output_scale, name="ffn"
        )(x_norm)

        return x
