#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optimum[onnxruntime-gpu]",
#     "onnxruntime-gpu",
#     "onnx",
# ]
# ///
"""Export BAAI/bge-base-en-v1.5 to an optimized, inference-ready ONNX model.

Pipeline
--------
1. **Export** — converts the HuggingFace model to ONNX via ``optimum``.
2. **Graph optimise** — applies ORT's BERT-specific fusions (``Attention``,
   ``BiasGelu``, ``SkipLayerNormalization``, ``EmbedLayerNormalization``).
3. **FP16 conversion** — converts weights and activations to FP16 for tensor-
   core acceleration.
4. **Post-processing** — applied on the FP16 graph:
   a. *INT32 inputs* — removes the three INT64→INT32 Cast nodes inserted by
      the ORT optimizer and re-declares inputs as INT32, halving host→device
      input transfer size.
   b. *FP16 mean-pooling + L2 normalisation* — fuses pooling into the graph
      using only CUDA-friendly primitives (Mul, ReduceSum, Sqrt, Clip, Div).
      The standard ONNX ``LpNormalization`` op is deliberately avoided because
      ORT lacks a CUDA kernel for it, which would force a GPU→CPU→GPU bounce.
      The final ``[B, 768]`` output stays in FP16 — no cast to FP32.  This
      halves the output transfer size and lets the Rust runtime store
      embeddings natively in FP16.

The final model (``model_fp16_pooled.onnx``) is the only artifact consumed by
the tributary Rust runtime.  Intermediate files are retained for debugging.

Usage
-----
::

    uv run scripts/export_onnx.py
    uv run scripts/export_onnx.py --model BAAI/bge-base-en-v1.5 \\
        --output-dir ort/models/bge-base-en-v1.5-onnx

Output files
------------
::

    model.onnx              - Base ONNX export (FP32, debugging only)
    model_opt.onnx          - After graph optimisation (FP32, debugging only)
    model_fp16.onnx         - After FP16 conversion (debugging only)
    model_fp16_pooled.onnx  - Final model (FP16 in/out, used by tributary)
"""

import argparse
import os
from pathlib import Path

import numpy as np


def optimize_inputs(model) -> None:
    """Optimize model inputs **in-place**.

    Args:
        model: An ``onnx.ModelProto`` loaded via ``onnx.load``.

    **INT64 → INT32**: removes the three ``Cast`` nodes inserted by the ORT
    transformer optimizer and changes the model input declarations to INT32.
    This halves the host→device input transfer size.

    ``token_type_ids`` is kept as a model input (INT32) rather than replaced
    with a graph-computed constant.  Although it's always zero for single-
    segment models like BGE, a ``ConstantOfShape`` node would allocate a fresh
    ``[B, S]`` tensor on every inference call (ORT cannot cache it for dynamic
    shapes).  The Rust side handles this more efficiently: a pre-allocated
    pinned tensor zeroed with a single ``memset``.
    """
    from onnx import TensorProto

    graph = model.graph

    # ── Remove INT64 → INT32 Cast nodes ─────────────────────────────────
    input_names = {"input_ids", "attention_mask", "token_type_ids"}
    cast_nodes_to_remove = []
    cast_remap: dict[str, str] = {}  # cast_output → original_input

    for node in graph.node:
        if node.op_type == "Cast" and len(node.input) == 1 and node.input[0] in input_names:
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.INT32:
                    cast_nodes_to_remove.append(node)
                    cast_remap[node.output[0]] = node.input[0]

    for node in cast_nodes_to_remove:
        graph.node.remove(node)
        print(f"    Removed INT64→INT32 Cast: {node.input[0]}")

    # Rewrite downstream consumers to reference the (now-INT32) inputs.
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in cast_remap:
                node.input[i] = cast_remap[inp]

    # Change input type declarations from INT64 → INT32.
    for inp in graph.input:
        if inp.name in input_names:
            inp.type.tensor_type.elem_type = TensorProto.INT32


def fuse_mean_pool_l2_norm(model) -> None:
    """Append mean-pooling + L2 normalisation to an ONNX transformer model
    **in-place**.

    Args:
        model: An ``onnx.ModelProto`` loaded via ``onnx.load``.

    Transforms the graph so that:
      - Old output  ``last_hidden_state  [B, S, 768]``
      - New output  ``embeddings         [B, 768]``

    Mean-pooling is implemented as a **batched MatMul** rather than the naive
    ``Mul([B,S,768]) + ReduceSum`` pattern.  The operation ``Σ_s h[b,s,d] *
    mask[b,s]`` is expressed as ``MatMul(mask[B,1,S], hidden[B,S,768])``
    which cuBLAS executes in a single fused kernel — no intermediate
    ``[B,S,768]`` tensor is ever materialised.

    L2 normalisation is decomposed into ``Mul`` → ``ReduceSum`` → ``Sqrt``
    → ``Clip`` → ``Div`` rather than using the ONNX ``LpNormalization`` op,
    because ORT has no CUDA kernel for ``LpNormalization`` — it falls back to
    CPU, causing a GPU→CPU→GPU data bounce.  The manual decomposition keeps
    the entire pooling tail on-GPU.

    The ``attention_mask`` model input (already present) is reused as the
    pooling weight.  All new ops use **opset 17** conventions.  The final
    output stays in FP16.
    """
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph

    # ── Identify the existing output and attention_mask input ────────────
    assert len(graph.output) >= 1, "Model must have at least one output"
    old_output_name = graph.output[0].name  # e.g. "last_hidden_state"

    attn_mask_input = None
    for inp in graph.input:
        if inp.name == "attention_mask":
            attn_mask_input = inp
            break
    assert attn_mask_input is not None, "Model must have an 'attention_mask' input"

    # ── Tap the FP16 tensor before the keep_io_types Cast ───────────────
    # convert_float_to_float16(keep_io_types=True) inserts a Cast(FP16→FP32)
    # at the output.  We remove it and feed the FP16 tensor directly into the
    # pooling ops.  The final output stays in FP16 (no cast back to FP32).
    fp16_hidden_name = None
    cast_to_remove = None
    for node in graph.node:
        if node.op_type == "Cast" and old_output_name in node.output:
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    fp16_hidden_name = node.input[0]
                    cast_to_remove = node
                    break

    if fp16_hidden_name is not None:
        graph.node.remove(cast_to_remove)
        print(f"    Removed FP16→FP32 Cast on [B, S, 768] ('{cast_to_remove.name}')")
    else:
        # No cast found — model output is already FP16 or FP32 directly.
        fp16_hidden_name = old_output_name
        print("    Warning: no FP16→FP32 Cast found; pooling in output dtype")

    # ── Constant tensors (axes, epsilon) ────────────────────────────────
    # Unsqueeze axes = [1]  (expand [B,S] → [B,1,S] for MatMul)
    unsqueeze_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="pool_unsqueeze_axes")
    # Squeeze axes = [1]  (collapse [B,1,H] → [B,H] after MatMul)
    squeeze_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="pool_squeeze_axes")
    # ReduceSum axes = [1]  (sum across seq_len / hidden dimension)
    reduce_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="pool_reduce_axes")
    # Clip min in FP16 — use 1e-4 (safely in the FP16 normal range; smaller
    # values like 1e-7 fall into subnormal territory and some GPUs flush
    # subnormals to zero).  The mask sum is always >= 1 for valid inputs,
    # so this is purely a safety net.
    clip_min = numpy_helper.from_array(np.array(1e-4, dtype=np.float16), name="pool_clip_min")

    graph.initializer.extend([unsqueeze_axes, squeeze_axes, reduce_axes, clip_min])

    # ── New nodes ───────────────────────────────────────────────────────
    new_nodes = []

    # 1. Cast attention_mask [B,S] INT32 → FP16 (before Unsqueeze so the
    #    mask ReduceSum can branch off the [B,S] tensor directly).
    new_nodes.append(
        helper.make_node(
            "Cast",
            inputs=["attention_mask"],
            outputs=["pool_mask_fp16"],
            name="pool_Cast",
            to=TensorProto.FLOAT16,
        )
    )

    # 2. Unsqueeze mask: [B,S] → [B,1,S] for the MatMul.
    new_nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["pool_mask_fp16", "pool_unsqueeze_axes"],
            outputs=["pool_mask_3d"],
            name="pool_Unsqueeze",
        )
    )

    # 3. MatMul: [B,1,S] × [B,S,768] → [B,1,768]  (fused mask + reduce)
    #    Replaces the old Mul([B,S,768]) + ReduceSum pattern — cuBLAS
    #    executes this in a single kernel with no intermediate tensor.
    new_nodes.append(
        helper.make_node(
            "MatMul",
            inputs=["pool_mask_3d", fp16_hidden_name],
            outputs=["pool_sum_hidden_3d"],
            name="pool_MatMul",
        )
    )

    # 4. Squeeze: [B,1,768] → [B,768]
    new_nodes.append(
        helper.make_node(
            "Squeeze",
            inputs=["pool_sum_hidden_3d", "pool_squeeze_axes"],
            outputs=["pool_sum_hidden"],
            name="pool_Squeeze",
        )
    )

    # 5. ReduceSum(mask_fp16[B,S], axes=[1], keepdims=1) → [B,1] FP16
    new_nodes.append(
        helper.make_node(
            "ReduceSum",
            inputs=["pool_mask_fp16", "pool_reduce_axes"],
            outputs=["pool_mask_sum"],
            name="pool_ReduceSum_mask",
            keepdims=1,
        )
    )

    # 6. Clip mask_sum >= 1e-4 (FP16 — avoids div-by-zero)
    new_nodes.append(
        helper.make_node(
            "Clip",
            inputs=["pool_mask_sum", "pool_clip_min", ""],
            outputs=["pool_mask_clamped"],
            name="pool_Clip",
        )
    )

    # 7. Div: mean pooling (FP16)
    new_nodes.append(
        helper.make_node(
            "Div",
            inputs=["pool_sum_hidden", "pool_mask_clamped"],
            outputs=["pool_mean"],
            name="pool_Div",
        )
    )

    # 8–12. Manual L2 normalisation (replaces LpNormalization which lacks
    #        a CUDA EP kernel and would force a GPU→CPU→GPU bounce).
    #        All primitives (Mul, ReduceSum, Sqrt, Clip, Div) have FP16
    #        CUDA kernels.

    # 8. Square: pool_mean² → [B, 768] FP16
    new_nodes.append(
        helper.make_node(
            "Mul",
            inputs=["pool_mean", "pool_mean"],
            outputs=["pool_squared"],
            name="pool_Square",
        )
    )

    # 9. ReduceSum(squared, axis=1, keepdims=1) → [B, 1] FP16
    new_nodes.append(
        helper.make_node(
            "ReduceSum",
            inputs=["pool_squared", "pool_reduce_axes"],
            outputs=["pool_sum_sq"],
            name="pool_ReduceSum_sq",
            keepdims=1,
        )
    )

    # 10. Sqrt → L2 norm per row → [B, 1] FP16
    new_nodes.append(
        helper.make_node(
            "Sqrt",
            inputs=["pool_sum_sq"],
            outputs=["pool_l2_raw"],
            name="pool_Sqrt",
        )
    )

    # 11. Clip L2 norm ≥ 1e-4 (reuse pool_clip_min; avoids div-by-zero)
    new_nodes.append(
        helper.make_node(
            "Clip",
            inputs=["pool_l2_raw", "pool_clip_min", ""],
            outputs=["pool_l2_safe"],
            name="pool_ClipL2",
        )
    )

    # 12. Div: normalise → [B, 768] FP16 (broadcasts [B, 1])
    new_nodes.append(
        helper.make_node(
            "Div",
            inputs=["pool_mean", "pool_l2_safe"],
            outputs=["embeddings"],
            name="pool_DivNorm",
        )
    )

    graph.node.extend(new_nodes)

    # ── Replace output ──────────────────────────────────────────────────
    # Remove the old output(s) and add the pooled one.
    while len(graph.output) > 0:
        graph.output.pop()

    graph.output.append(helper.make_tensor_value_info("embeddings", TensorProto.FLOAT16, ["batch_size", 768]))

    print("    Fused FP16 mean-pooling + L2 norm → [B, 768] FP16 output")


def main():
    parser = argparse.ArgumentParser(description="Export BGE embedding model to optimized ONNX format")
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model ID (default: BAAI/bge-base-en-v1.5)",
    )
    parser.add_argument(
        "--output-dir",
        default="ort/models/bge-base-en-v1.5-onnx",
        help="Output directory for ONNX files (default: ort/models/bge-base-en-v1.5-onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Number of attention heads (default: 12 for BERT-base)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="Hidden size (default: 768 for BERT-base)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Export to ONNX ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 1: Exporting {args.model} to ONNX (opset {args.opset})")
    print(f"{'=' * 60}")

    from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    model.save_pretrained(str(output_dir))
    base_path = output_dir / "model.onnx"
    print(f"  Base ONNX model saved to: {base_path}")
    print(f"  Size: {os.path.getsize(base_path) / 1024 / 1024:.1f} MB")

    # ── Step 2: Optimize with ORT transformer optimizer ──────────────────
    print(f"\n{'=' * 60}")
    print("Step 2: Applying BERT-specific graph optimizations (O2)")
    print(f"{'=' * 60}")

    from onnxruntime.transformers import optimizer

    opt_model = optimizer.optimize_model(
        str(base_path),
        model_type="bert",
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
    )

    opt_path = output_dir / "model_opt.onnx"
    opt_model.save_model_to_file(str(opt_path))
    print(f"  Optimized model saved to: {opt_path}")
    print(f"  Size: {os.path.getsize(opt_path) / 1024 / 1024:.1f} MB")

    # Log fused operators
    fused_ops = {}
    for node in opt_model.model.graph.node:
        if node.domain == "com.microsoft":
            fused_ops[node.op_type] = fused_ops.get(node.op_type, 0) + 1
    if fused_ops:
        print("  Fused operators:")
        for op, count in sorted(fused_ops.items()):
            print(f"    {op}: {count}")

    # ── Step 3: Convert to FP16 ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 3: Converting to FP16")
    print(f"{'=' * 60}")

    opt_model.convert_float_to_float16(
        keep_io_types=True  # Keep inputs/outputs as FP32 for compatibility
    )

    fp16_path = output_dir / "model_fp16.onnx"
    opt_model.save_model_to_file(str(fp16_path))
    print(f"  FP16 model saved to: {fp16_path}")
    print(f"  Size: {os.path.getsize(fp16_path) / 1024 / 1024:.1f} MB")

    # ── Step 4: Post-processing (INT32 inputs + fused FP16 pooling) ─────
    print(f"\n{'=' * 60}")
    print("Step 4: Post-processing (INT32 inputs + fused FP16 pooling)")
    print(f"{'=' * 60}")

    import onnx

    post_model = onnx.load(str(fp16_path))
    optimize_inputs(post_model)
    fuse_mean_pool_l2_norm(post_model)

    final_path = output_dir / "model_fp16_pooled.onnx"
    onnx.checker.check_model(post_model)
    onnx.save(post_model, str(final_path))
    print(f"  Final model saved to: {final_path}")
    print(f"  Size: {os.path.getsize(final_path) / 1024 / 1024:.1f} MB")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    summary_model = onnx.load(str(final_path))
    print(f"  Model: {args.model}")
    print(f"  Opset: {summary_model.opset_import[0].version}")
    print("  Inputs:")
    for inp in summary_model.graph.input:
        shape = [d.dim_param if d.dim_param else str(d.dim_value) for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"    {inp.name}: [{', '.join(shape)}] (dtype={dtype})")
    print("  Outputs:")
    for out in summary_model.graph.output:
        shape = [d.dim_param if d.dim_param else str(d.dim_value) for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f"    {out.name}: [{', '.join(shape)}] (dtype={dtype})")

    final_size = os.path.getsize(final_path)
    print(f"\n  Final model size: {final_size / 1024 / 1024:.1f} MB")
    print(f"  Output path:      {final_path}")


if __name__ == "__main__":
    main()
