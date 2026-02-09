#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optimum[onnxruntime-gpu]",
#     "onnxruntime-gpu",
#     "onnx",
# ]
# ///
"""Export BAAI/bge-base-en-v1.5 to an optimized ONNX model.

This script:
1. Exports the model to ONNX format using HuggingFace optimum.
2. Applies ONNX Runtime's BERT-specific graph optimizations (attention fusion,
   layer normalization fusion, etc.) at level O2.
3. Converts the optimized model to FP16 for faster inference on GPUs with
   tensor cores.
4. Fuses mean-pooling + L2 normalisation into the graph so the model outputs
   [batch_size, 768] embeddings directly (instead of [batch_size, seq_len, 768]
   hidden states).  This eliminates CPU-side pooling and reduces the GPU→CPU
   output transfer by ~10x.

The resulting model can be used with the ORT backend in tributary.

Usage:
    ./ort/scripts/export_onnx.py
    uv run ort/scripts/export_onnx.py
    uv run ort/scripts/export_onnx.py --model BAAI/bge-base-en-v1.5 --output-dir ort/models/bge-base-en-v1.5-onnx

The output directory will contain:
    model.onnx              - Base ONNX model (FP32)
    model_opt.onnx          - Optimized model (FP32, fused attention)
    model_fp16.onnx         - Optimized model (FP16, fused attention)
    model_fp16_pooled.onnx  - Optimized model (FP16, fused attention, fused pooling)
"""

import argparse
import os
from pathlib import Path

import numpy as np


def fuse_mean_pool_l2_norm(input_path: str, output_path: str) -> None:
    """Append mean-pooling + L2 normalisation to an ONNX transformer model.

    Transforms the graph so that:
      - Old output  ``last_hidden_state  [B, S, 768]``
      - New output  ``embeddings         [B, 768]``

    The ``attention_mask`` model input (already present) is reused as the
    pooling weight.

    All new ops use **opset 17** conventions (axes-as-input for ReduceSum /
    Unsqueeze).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(input_path)
    graph = model.graph

    # ── Identify the existing output and attention_mask input ────────────
    assert len(graph.output) >= 1, "Model must have at least one output"
    hidden_output_name = graph.output[0].name  # e.g. "last_hidden_state"

    attn_mask_input = None
    for inp in graph.input:
        if inp.name == "attention_mask":
            attn_mask_input = inp
            break
    assert attn_mask_input is not None, "Model must have an 'attention_mask' input"

    # ── Constant tensors (axes, epsilon) ────────────────────────────────
    # Unsqueeze axes = [2]  (expand [B,S] → [B,S,1])
    unsqueeze_axes = numpy_helper.from_array(
        np.array([2], dtype=np.int64), name="pool_unsqueeze_axes"
    )
    # ReduceSum axes = [1]  (sum across seq_len dimension)
    reduce_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="pool_reduce_axes"
    )
    # Clip min = 1e-7  (avoid division by zero)
    clip_min = numpy_helper.from_array(
        np.array(1e-7, dtype=np.float32), name="pool_clip_min"
    )
    # Clip has an optional max — we leave it empty (unused).
    clip_max_name = ""

    graph.initializer.extend([unsqueeze_axes, reduce_axes, clip_min])

    # ── New nodes ───────────────────────────────────────────────────────
    new_nodes = []

    # 1. Unsqueeze attention_mask: [B,S] → [B,S,1]
    new_nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["attention_mask", "pool_unsqueeze_axes"],
            outputs=["pool_mask_3d"],
            name="pool_Unsqueeze",
        )
    )

    # 2. Cast mask from INT64 → FLOAT
    new_nodes.append(
        helper.make_node(
            "Cast",
            inputs=["pool_mask_3d"],
            outputs=["pool_mask_float"],
            name="pool_Cast",
            to=TensorProto.FLOAT,
        )
    )

    # 3. Mul: hidden_states * mask → zero out padding positions
    new_nodes.append(
        helper.make_node(
            "Mul",
            inputs=[hidden_output_name, "pool_mask_float"],
            outputs=["pool_masked_hidden"],
            name="pool_Mul",
        )
    )

    # 4. ReduceSum(masked_hidden, axes=[1], keepdims=0) → [B, H]
    new_nodes.append(
        helper.make_node(
            "ReduceSum",
            inputs=["pool_masked_hidden", "pool_reduce_axes"],
            outputs=["pool_sum_hidden"],
            name="pool_ReduceSum_hidden",
            keepdims=0,
        )
    )

    # 5. ReduceSum(float_mask, axes=[1], keepdims=0) → [B, 1]
    new_nodes.append(
        helper.make_node(
            "ReduceSum",
            inputs=["pool_mask_float", "pool_reduce_axes"],
            outputs=["pool_mask_sum"],
            name="pool_ReduceSum_mask",
            keepdims=0,
        )
    )

    # 6. Clip mask_sum >= 1e-7
    new_nodes.append(
        helper.make_node(
            "Clip",
            inputs=["pool_mask_sum", "pool_clip_min", clip_max_name],
            outputs=["pool_mask_clamped"],
            name="pool_Clip",
        )
    )

    # 7. Div: mean pooling
    new_nodes.append(
        helper.make_node(
            "Div",
            inputs=["pool_sum_hidden", "pool_mask_clamped"],
            outputs=["pool_mean"],
            name="pool_Div",
        )
    )

    # 8. LpNormalization (L2, axis=1)
    new_nodes.append(
        helper.make_node(
            "LpNormalization",
            inputs=["pool_mean"],
            outputs=["embeddings"],
            name="pool_LpNorm",
            axis=1,
            p=2,
        )
    )

    graph.node.extend(new_nodes)

    # ── Replace output ──────────────────────────────────────────────────
    # Remove the old output(s) and add the pooled one.
    while len(graph.output) > 0:
        graph.output.pop()

    graph.output.append(
        helper.make_tensor_value_info(
            "embeddings", TensorProto.FLOAT, ["batch_size", 768]
        )
    )

    # ── Validate and save ───────────────────────────────────────────────
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export BGE embedding model to optimized ONNX format"
    )
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
    parser.add_argument(
        "--skip-fp16",
        action="store_true",
        help="Skip FP16 conversion (useful for debugging)",
    )
    parser.add_argument(
        "--skip-fuse-pooling",
        action="store_true",
        help="Skip fusing mean-pooling + L2 normalisation into the model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Export to ONNX ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 1: Exporting {args.model} to ONNX (opset {args.opset})")
    print(f"{'=' * 60}")

    from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(
        args.model, export=True
    )
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
    if not args.skip_fp16:
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
    else:
        fp16_path = opt_path
        print("\n  Skipping FP16 conversion (--skip-fp16)")

    # ── Step 4: Fuse mean-pooling + L2 norm ─────────────────────────────
    if not args.skip_fuse_pooling:
        print(f"\n{'=' * 60}")
        print("Step 4: Fusing mean-pooling + L2 normalisation")
        print(f"{'=' * 60}")

        pooled_path = output_dir / "model_fp16_pooled.onnx"
        fuse_mean_pool_l2_norm(str(fp16_path), str(pooled_path))
        print(f"  Pooled model saved to: {pooled_path}")
        print(f"  Size: {os.path.getsize(pooled_path) / 1024 / 1024:.1f} MB")
        fp16_path = pooled_path
    else:
        print("\n  Skipping pooling fusion (--skip-fuse-pooling)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    import onnx

    final_model = onnx.load(str(fp16_path))
    print(f"  Model: {args.model}")
    print(f"  Opset: {final_model.opset_import[0].version}")
    print(f"  Inputs:")
    for inp in final_model.graph.input:
        shape = [
            d.dim_param if d.dim_param else str(d.dim_value)
            for d in inp.type.tensor_type.shape.dim
        ]
        dtype = inp.type.tensor_type.elem_type
        print(f"    {inp.name}: [{', '.join(shape)}] (dtype={dtype})")
    print(f"  Outputs:")
    for out in final_model.graph.output:
        shape = [
            d.dim_param if d.dim_param else str(d.dim_value)
            for d in out.type.tensor_type.shape.dim
        ]
        dtype = out.type.tensor_type.elem_type
        print(f"    {out.name}: [{', '.join(shape)}] (dtype={dtype})")

    final_size = os.path.getsize(fp16_path)
    print(f"\n  Final model size: {final_size / 1024 / 1024:.1f} MB")
    print(f"  Output path:      {fp16_path}")


if __name__ == "__main__":
    main()
