#!/usr/bin/env python3
"""Convert RKNN-friendly 9-output YOLO ONNX to RK3588 RKNN.

Input ONNX must be produced by device/rewrite_yolo_onnx_model_zoo.py and should
have 9 outputs: box/class/score_sum for strides 8, 16, and 32.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--dataset", help="Calibration dataset.txt for INT8")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--algorithm",
        default="normal",
        choices=["normal", "mmse", "kl_divergence"],
        help="RKNN quantized_algorithm for INT8 builds",
    )
    parser.add_argument("--no-quantize", action="store_true", help="Build FP16 RKNN")
    parser.add_argument("--no-memory-optimize", action="store_true")
    parser.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--rknn-normalize",
        action="store_true",
        help="Set mean/std to let RKNN divide uint8-like input by 255. Do not use with pre-normalized .npy calibration.",
    )
    args = parser.parse_args()

    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 is not installed in this environment.")
        sys.exit(1)

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    dataset_path = Path(args.dataset) if args.dataset else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not args.no_quantize and dataset_path is None:
        raise ValueError("--dataset is required unless --no-quantize is set")
    if dataset_path is not None and not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"ONNX: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Dataset: {dataset_path if dataset_path else 'not used'}")
    print(f"Output: {output_path}")
    print(f"Mode: {'FP16' if args.no_quantize else 'INT8'}")

    rknn = RKNN(verbose=True)
    try:
        config_kwargs = dict(
            target_platform="rk3588",
            quantized_dtype="asymmetric_quantized-8",
            quantized_algorithm=args.algorithm,
            optimization_level=args.optimization_level,
            single_core_mode=not args.no_memory_optimize,
            compress_weight=not args.no_memory_optimize,
        )
        if args.rknn_normalize:
            config_kwargs.update(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])

        print("\n[1/4] Config")
        rknn.config(**config_kwargs)
        for key, value in config_kwargs.items():
            print(f"  {key}: {value}")

        print("\n[2/4] Load ONNX")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")

        print("\n[3/4] Build")
        if args.no_quantize:
            ret = rknn.build(do_quantization=False)
        else:
            ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")

        print("\n[4/4] Export")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: {ret}")

        print("\nDone")
        print(f"  RKNN: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        try:
            mem = rknn.eval_memory(is_print=False)
            if mem:
                print(f"  eval_memory: {mem}")
        except Exception:
            pass
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
