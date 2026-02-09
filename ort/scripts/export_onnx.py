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

The resulting model can be used with the ORT backend in tributary.

Usage:
    ./ort/scripts/export_onnx.py
    uv run ort/scripts/export_onnx.py
    uv run ort/scripts/export_onnx.py --model BAAI/bge-base-en-v1.5 --output-dir ort/models/bge-base-en-v1.5-onnx

The output directory will contain:
    model.onnx       - Base ONNX model (FP32)
    model_opt.onnx   - Optimized model (FP32, fused attention)
    model_fp16.onnx  - Optimized model (FP16, fused attention)
"""

import argparse
import os
from pathlib import Path


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
