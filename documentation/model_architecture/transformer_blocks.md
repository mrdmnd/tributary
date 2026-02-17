# Transformer Block Structure

Confluence uses a **pre-norm** architecture. Each "layer" is a triple of attention sublayers followed by a
feed-forward sublayer. Each attention sublayer applies SDPA output gating before the residual add:

```
# One Confluence layer (repeated N_layers times)
x_norm = RMSNorm(x)
x = x + GatedAttention(OutboundSDPA, x_norm, W_gate_out)    # outbound (self + FK parents)
x_norm = RMSNorm(x)
x = x + GatedAttention(InboundSDPA, x_norm, W_gate_in)      # inbound (FK children)
x_norm = RMSNorm(x)
x = x + GatedAttention(ColumnSDPA, x_norm, W_gate_col)      # same-column peers
x = x + FFN(RMSNorm(x))                                     # feed-forward
```

A final RMSNorm is applied after the last layer, before the decoder heads:

```
h_final = RMSNorm(x)                        # [B, S, D]
```

Each attention sublayer internally performs: permute → QK-normed multi-head attention → unpermute.
See [attention.md](attention.md) for mask derivation and permutation details.

## SDPA Output Gating

Following [Qiu et al., 2025](https://arxiv.org/abs/2505.06708), each attention sublayer applies a sigmoid gate to
the SDPA output before the residual connection:

```
def GatedAttention(sdpa_fn, x_norm, W_gate):
    attn_out = sdpa_fn(x_norm)               # permute → block-sparse MHA → unpermute → W_O
    gate = sigmoid(x_norm @ W_gate)          # [B, S, D]
    return attn_out * gate                   # elementwise modulation before residual add
```

The gate operates **outside** the block-sparse attention kernel:

```
┌──────────────────────────────────────┐
│  Block-sparse SDPA kernel            │
│  permute → Q,K,V → attn → W_O       │
└──────────────────┬───────────────────┘
                   │ attn_out [B, S, D]
                   ▼
         ┌─────────────────────┐
         │  Gate (standard ops) │    x_norm @ W_gate → sigmoid
         │  attn_out * gate     │    elementwise multiply
         └─────────┬───────────┘
                   │ gated_out [B, S, D]
                   ▼
              residual add: x = x + gated_out
```

No fused kernel modifications needed. XLA fuses the sigmoid and elementwise multiply automatically.

`W_gate ∈ ℝ^{D × D}` — one per attention sublayer, three per layer. No bias. Initialized with Xavier uniform.
At initialization, gate outputs ≈ 0.5, halving the attention output. Falls under **Muon** in the optimizer split.

**Parameter overhead:** 65,536 per sublayer (D=256), 196,608 per layer — comparable to a single W_Q or W_V matrix.

## SwiGLU Feed-Forward Network

```
SwiGLU(x) = W_2 · (SiLU(W_gate · x) ⊙ (W_up · x))
```

- `W_gate ∈ ℝ^{D × D_ff}` — gate projection
- `W_up   ∈ ℝ^{D × D_ff}` — value projection
- `W_2    ∈ ℝ^{D_ff × D}` — down projection

**Hidden dimension:** `D_ff = ⌈(8/3)D⌉` rounded to nearest multiple of 256. At D=256: **D_ff = 768**.

**No bias terms** in attention projections (W_Q, W_K, W_V, W_O) or FFN (W_gate, W_up, W_2). Value encoders and
decoder heads retain biases (small boundary layers between frozen embeddings and model space).

## RMSNorm Sites

Zero-centered RMSNorm (`(1 + γ) ⊙ x / √(mean(x²) + ε)`, γ initialized to 0) applied at:
- Before each of the three attention sublayers
- Before the FFN sublayer
- After value encoding (producing h₀)
- After the final layer (before decoder heads)
