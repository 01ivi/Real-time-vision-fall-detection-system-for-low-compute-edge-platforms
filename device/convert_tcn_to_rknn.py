#!/usr/bin/env python3
"""Convert GeometryTCN FP32 ONNX → RKNN for RK3588 NPU.

Target: rknn-toolkit2 1.6.0 + librknnrt 1.6.0 + NPU driver 0.9.8 (Firefly RK3588)
Memory: single_core_mode + compress_weight for minimal NPU footprint
TCN input is (1, 16, 10) — independent of YOLO imgsz, but recompile for consistency

This script runs on an **x86_64 development machine** with RKNN-Toolkit2 1.6.0.
It can export either FP16 RKNN without calibration or INT8 RKNN with a
calibration dataset (geometry .npy files).

Prerequisites (x86_64, Python 3.10, rknn-toolkit2 1.6.0):
  pip install rknn_toolkit2-1.6.0+...-cp310-cp310-linux_x86_64.whl --no-deps
  pip install onnx==1.14.1 onnxruntime==1.16.0 numpy

Calibration dataset must be prepared first:
  python scripts/prepare_rknn_calibration.py \
    --manifest data/baseline_to_itw_cs/of-itw_train.jsonl \
    --detector-weights checkpoint/yolo26m.pt \
    --output-dir data/tcn_rknn_calibration \
    --max-samples 256

For FP16 RKNN, no calibration dataset is needed:
  python device/convert_tcn_to_rknn.py \
    --onnx runs/final/export/temporal_head_fp32_single_ln_decomposed.onnx \
    --output runs/final/export/temporal_head_fp16_ln_decomposed_rk3588.rknn \
    --no-quantize

For INT8 RKNN, run this script:
  python device/convert_tcn_to_rknn.py \
    --onnx runs/final/export/temporal_head_fp32_single_ln_decomposed.onnx \
    --dataset data/tcn_rknn_calibration/dataset.txt \
    --output runs/final/export/temporal_head_int8_rk3588.rknn

Output size should be < 200 KB (easily satisfies < 20 MB constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GeometryTCN ONNX to RK3588 NPU INT8 RKNN (toolkit 1.6.0)"
    )
    parser.add_argument(
        "--onnx", required=True,
        help="Path to FP32 ONNX model (temporal_head_fp32_single.onnx)",
    )
    parser.add_argument(
        "--dataset",
        help="Path to calibration dataset.txt (one .npy per line, shape (16,10) float32)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .rknn file path",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip INT8 quant (FP16 RKNN for comparison)",
    )
    parser.add_argument(
        "--no-memory-optimize", action="store_true",
        help="Disable single_core_mode and compress_weight",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="RKNN optimization level. Use 0 if toolkit fusion creates runtime-unsupported ops.",
    )
    parser.add_argument(
        "--disable-fuse",
        action="store_true",
        help="Disable supported output optimization flags. Toolkit 1.6.0 still may fuse internal ops.",
    )
    args = parser.parse_args()

    # ── Import RKNN-Toolkit2 ────────────────────────────────────────────
    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 1.6.0 not installed.")
        print("Install from Rockchip repo:")
        print("  pip install rknn_toolkit2-1.6.0+...-cp310-cp310-linux_x86_64.whl --no-deps")
        sys.exit(1)

    onnx_path = Path(args.onnx)
    dataset_path = Path(args.dataset) if args.dataset else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not args.no_quantize and dataset_path is None:
        raise ValueError("--dataset is required unless --no-quantize is set")
    if dataset_path is not None and not dataset_path.exists():
        raise FileNotFoundError(f"Dataset list not found: {dataset_path}")

    onnx_kb = onnx_path.stat().st_size / 1024
    print(f"ONNX: {onnx_path}  ({onnx_kb:.1f} KB)")
    print(f"Dataset: {dataset_path if dataset_path is not None else 'not used'}")
    print(f"Output: {output_path}")

    # ── Create RKNN session ─────────────────────────────────────────────
    rknn = RKNN(verbose=True)

    # ── 1. Configure ────────────────────────────────────────────────────
    print("\n[1/5] Configuring RKNN for RK3588 NPU (toolkit 1.6.0)...")

    if args.no_quantize:
        quantized_dtype = "asymmetric_quantized-8"
        quant_mode = "FP16"
    else:
        quantized_dtype = "asymmetric_quantized-8"
        quant_mode = "INT8"

    fuse_kwargs = {}
    if args.disable_fuse:
        fuse_kwargs = {
            "output_optimize": False,
        }

    rknn.config(
        target_platform="rk3588",
        quantized_dtype=quantized_dtype,
        optimization_level=args.optimization_level,
        # ── Memory optimization ──
        single_core_mode=not args.no_memory_optimize,
        compress_weight=not args.no_memory_optimize,
        # ── Dynamic input: TCN has dynamic batch dim ──
        dynamic_input=[[[1, 16, 10]]],
        **fuse_kwargs,
    )
    print(f"  target_platform  = rk3588")
    print(f"  quantized_dtype  = {quantized_dtype}")
    print(f"  quant_mode       = {quant_mode}")
    print(f"  optimization     = {args.optimization_level}")
    print(f"  disable_fuse     = {args.disable_fuse}")
    print(f"  single_core_mode = {not args.no_memory_optimize}")
    print(f"  compress_weight  = {not args.no_memory_optimize}")
    print(f"  dynamic_input    = [[[1, 16, 10]]]")

    # ── 2. Load ONNX ────────────────────────────────────────────────────
    print("\n[2/5] Loading FP32 ONNX...")
    # TCN has dynamic batch dim — must specify inputs/input_size_list
    ret = rknn.load_onnx(
        model=str(onnx_path),
        inputs=["geometry"],
        input_size_list=[[1, 16, 10]],  # [batch=1, clip_len, geometry_dim]
    )
    if ret != 0:
        print(f"ERROR: load_onnx failed with code {ret}")
        print()
        print("Common causes:")
        print("  - Dynamic batch dim: ensure dynamic_input is set in config()")
        print("  - onnx version mismatch: ensure onnx==1.14.1 for toolkit 1.6.0")
        rknn.release()
        sys.exit(1)
    print("  ONNX loaded OK")

    # ── 3. Build (with INT8 calibration) ────────────────────────────────
    print(f"\n[3/5] Building RKNN model ({quant_mode})...")
    if args.no_quantize:
        print("  Quantization: DISABLED (FP16)")
        ret = rknn.build(do_quantization=False)
    else:
        print(f"  Quantization: INT8 (static, per-channel)")
        print(f"  Calibration dataset: {dataset_path}")
        ret = rknn.build(
            do_quantization=True,
            dataset=str(dataset_path),
        )
    if ret != 0:
        print(f"ERROR: build failed with code {ret}")
        rknn.release()
        sys.exit(1)
    print("  Build OK")

    # ── 4. Export ───────────────────────────────────────────────────────
    print(f"\n[4/5] Exporting RKNN to {output_path}...")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        print(f"ERROR: export failed with code {ret}")
        rknn.release()
        sys.exit(1)
    print(f"  Exported: {output_path}")

    # ── 5. Report sizes ─────────────────────────────────────────────────
    print(f"\n[5/5] Model summary")
    onnx_kb = onnx_path.stat().st_size / 1024
    rknn_kb = output_path.stat().st_size / 1024
    print(f"  FP32 ONNX            : {onnx_kb:.1f} KB")
    print(f"  RKNN ({quant_mode})  : {rknn_kb:.1f} KB")
    print(f"  single_core_mode     : {not args.no_memory_optimize}")
    print(f"  compress_weight      : {not args.no_memory_optimize}")
    print(f"  optimization_level   : {args.optimization_level}")
    print(f"  NPU memory target    : < 20 MB ({20480} KB)")
    print(f"  Status               : {'PASS' if rknn_kb < 20480 else 'FAIL'}")

    # Eval memory if available
    try:
        mem = rknn.eval_memory(is_print=False)
        if mem:
            print(f"  eval_memory()        : {mem}")
    except Exception:
        pass

    rknn.release()
    print(f"\nDone. Copy {output_path.name} to your RK3588 board.")
    print(f"  scp {output_path} firefly@<board-ip>:~/FD/runs/final/export/")


if __name__ == "__main__":
    main()
