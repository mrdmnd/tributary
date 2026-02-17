# Training Configuration

## Precision: BF16

BF16 compute, FP32 state:

- **Activations**: BF16. Hidden states, attention logits, FFN intermediates.
- **Master weights**: FP32. Optimizer maintains full-precision copy; cast to BF16 for forward/backward.
- **Optimizer state**: FP32. Momentum buffers and second-moment estimates.
- **Loss computation**: FP32. Loss reduction, log/exp operations (BCE, softmax) are upcast.
- **Gradient accumulation**: FP32 across micro-batches.

## Normalization: Zero-Centered RMSNorm

```
ZC-RMSNorm(x) = (1 + γ) ⊙ x / √(mean(x²) + ε)
```

`γ` initialized to **0**, so the layer starts as pure normalization (effective scale = 1). Pre-norm residual blocks
begin as near-identity functions.

Applied at: before each attention sublayer, before FFN, after value encoding, after the final layer.

## QK Norm

L2 normalization on queries and keys after projection, with a learnable scalar temperature per head:

```
Q = L2Norm(W_Q @ x) · τ       (τ initialized to √d_head)
K = L2Norm(W_K @ x)
attn_logits = Q @ K^T / √d_head
```

Dot products bounded in [-1, 1] before scaling, preventing logit explosion in BF16.

## Optimizer: Muon + AdamW

Split optimizer strategy:

- **Muon** for all 2D weight matrices (attention projections, FFN weights, W_gate)
- **AdamW** for everything else (embeddings, RMSNorm scales, encoder/decoder biases, 1D parameters)

### Muon hyperparameters

| Parameter | Value |
|-----------|-------|
| β₁        | 0.95  |
| lr_peak   | 0.02  |
| Newton-Schulz iterations | 5 |

### AdamW hyperparameters

| Parameter    | Value |
|------------- |-------|
| β₁           | 0.9   |
| β₂           | 0.95  |
| weight_decay | 0.1 (not applied to biases or norms) |
| ε            | 1e-8  |
| lr_peak      | 3e-4  |

### Learning Rate Schedule

Both optimizers share a synchronized schedule:

```
lr(t) = lr_peak · warmup(t) · decay(t)

warmup(t) = min(1, t / T_warmup)
decay(t)  = 0.1 + 0.9 · (1 + cos(π · (t - T_warmup) / (T_total - T_warmup))) / 2
```

- `T_warmup`: ~2000 steps (or ~1% of total training steps, whichever is larger)
- Cosine decay to 10% of peak (not zero)

## Weight Initialization

| Parameter | Initialization | Scale factor |
|-----------|---------------|--------------|
| W_Q, W_K, W_V | Xavier uniform | 1 |
| W_O | Xavier uniform | `1/√(4 · N_layers)` |
| W_gate (SDPA) | Xavier uniform | 1 (gate starts ≈ 0.5, halving attn output) |
| W_2 (FFN down) | Xavier uniform | `1/√(4 · N_layers)` |
| W_ffn_gate, W_up | Xavier uniform | 1 |
| RMSNorm γ | 0.0 | — |
| null_emb, mask_emb, identifier_emb, boolean_encoder | Normal(0, 0.02) | — |
| Value encoder Linears | Xavier uniform | 1 |

The `1/√(4 · N_layers)` factor on W_O and W_2 ensures O(1) residual variance at init. The factor of 4 accounts for
the four residual-adding sublayers per layer (three attention + one FFN).

## Gradient Clipping

Global gradient norm clipping at **max_norm = 1.0**, applied in FP32 before the optimizer step.

## Z-Loss Regularization (Categorical)

For categorical targets, a z-loss penalty discourages large pre-softmax logits:

```
z_loss = 1e-4 · mean(log(Σ_k exp(logits_k))²)
cat_loss_total = cross_entropy(logits, target) + z_loss
```

Prevents logit drift and softmax saturation in BF16.

## Summary of Stability Techniques

| Technique | Where | Why |
|-----------|-------|-----|
| BF16 mixed precision | All compute | Wide dynamic range, no loss scaling needed |
| FP32 master weights | Optimizer state | Prevent weight update drift |
| FP32 loss reduction | Loss computation | Numerical accuracy in log/exp |
| Pre-norm ZC-RMSNorm | Every sublayer | Stable residual stream, near-identity init |
| QK norm + learned temperature | Attention | Bounded logits, prevents softmax saturation |
| SDPA output gating | Attention sublayers | Input-dependent sparsity, breaks W_V/W_O rank bottleneck |
| SwiGLU FFN | Feed-forward | Gated expressivity |
| No bias in core layers | Attention + FFN + gate | Cleaner optimization, kernel fusion |
| Muon (2D) + AdamW (1D) | Optimizer | Spectral preconditioning for matrix weights |
| Scaled residual init | W_O, W_2 | O(1) residual variance at init |
| Linear warmup + cosine decay | LR schedule | Stable early training |
| Gradient clipping | All parameters | Safety net against gradient spikes |
| Z-loss | Categorical CE | Prevents logit drift in BF16 |
