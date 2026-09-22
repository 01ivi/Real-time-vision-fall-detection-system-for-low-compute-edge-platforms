#!/usr/bin/env python3
"""Compare model-zoo YOLO ONNXRuntime vs RKNN Toolkit simulator outputs.

This does not execute an exported .rknn file. RKNN Toolkit 1.6 does not allow
simulator inference from load_rknn(); it must load/build from the source ONNX.
The goal is to check whether ONNX -> RKNN build changes the 9-output model-zoo
tensor semantics before moving the model to RK3588.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from benchmark_yolo_onnx_person import (
    UnifiedONNXPersonDetector,
    _draw_frame,
    _load_manifest,
    _resolve_path,
    _sample_frames,
    _select_videos,
)
from edgefall.models.detector import Detection


class RKNNSimulatorModelZooDetector(UnifiedONNXPersonDetector):
    def __init__(
        self,
        onnx_path: Path,
        imgsz: int,
        conf: float,
        person_class_id: int,
        nms_iou_threshold: float = 0.7,
        final_topk: int = 300,
        do_quantization: bool = False,
        dataset: str | None = None,
        algorithm: str = "normal",
    ) -> None:
        from rknn.api import RKNN

        self.onnx_path = onnx_path
        self.conf = conf
        self.person_class_id = person_class_id
        self.nms_iou_threshold = nms_iou_threshold
        self.final_topk = final_topk
        self.requested_imgsz = imgsz
        self.imgsz = imgsz
        self.last_scale = 1.0
        self.last_pad_left = 0.0
        self.last_pad_top = 0.0

        self.rknn = RKNN(verbose=False)
        self.rknn.config(
            target_platform="rk3588",
            quantized_dtype="asymmetric_quantized-8",
            quantized_algorithm=algorithm,
            optimization_level=3,
            single_core_mode=True,
            compress_weight=True,
        )
        ret = self.rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed: {ret}")
        if do_quantization:
            if not dataset:
                raise ValueError("dataset is required when do_quantization=True")
            ret = self.rknn.build(do_quantization=True, dataset=dataset)
        else:
            ret = self.rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN simulator init_runtime failed: {ret}")
        self.output_names = [f"output_{i}" for i in range(9)]
        self.output_shapes = []

    def raw_outputs(self, frame: np.ndarray) -> tuple[list[np.ndarray], float]:
        tensor = self._preprocess(frame)
        started = time.perf_counter()
        outputs = self.rknn.inference(inputs=[tensor], data_format=["nchw"])
        infer_ms = (time.perf_counter() - started) * 1000.0
        return outputs, infer_ms

    def predict(self, frame: np.ndarray) -> tuple[list[Detection], float, list[np.ndarray]]:
        outputs, infer_ms = self.raw_outputs(frame)
        return self._postprocess_modelzoo(outputs), infer_ms, outputs

    def release(self) -> None:
        self.rknn.release()


def _tensor_stats(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _compare_tensors(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"same_shape": False, "onnx_shape": list(a.shape), "rknn_shape": list(b.shape)}
    af = a.astype(np.float32, copy=False)
    bf = b.astype(np.float32, copy=False)
    diff = np.abs(af - bf)
    return {
        "same_shape": True,
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "p95_abs": float(np.percentile(diff, 95)),
    }


def _best_iou(det: Detection, candidates: list[Detection]) -> float:
    if not candidates:
        return 0.0
    return max(UnifiedONNXPersonDetector._iou(det, other) for other in candidates)


def _draw_side_by_side(frame: np.ndarray, onnx_dets: list[Detection], rknn_dets: list[Detection], title: str, out_path: Path) -> None:
    left = frame.copy()
    right = frame.copy()
    _draw_frame(left, onnx_dets, "ONNX " + title, out_path.with_suffix(".tmp_left.jpg"))
    _draw_frame(right, rknn_dets, "RKNN-sim " + title, out_path.with_suffix(".tmp_right.jpg"))
    left = cv2.imread(str(out_path.with_suffix(".tmp_left.jpg")))
    right = cv2.imread(str(out_path.with_suffix(".tmp_right.jpg")))
    combined = np.concatenate([left, right], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    out_path.with_suffix(".tmp_left.jpg").unlink(missing_ok=True)
    out_path.with_suffix(".tmp_right.jpg").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare modelzoo ONNX vs RKNN simulator")
    parser.add_argument("--onnx", default="runs/final/export/yolo26m_fp32_imgsz960_modelzoo.onnx")
    parser.add_argument("--manifest", default="data/baseline/of-sta-cs_test.jsonl")
    parser.add_argument("--video", action="append", default=None)
    parser.add_argument("--output-dir", default="runs/final/modelzoo_onnx_rknn_sim_compare")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--frame-interval", type=int, default=30)
    parser.add_argument("--max-frames-per-video", type=int, default=2)
    parser.add_argument("--positive-labels", default="fall,fallen")
    parser.add_argument("--do-quantization", action="store_true")
    parser.add_argument("--dataset", default="data/yolo_rknn_calib_staged_960_sub100/dataset.txt")
    parser.add_argument("--algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"])
    args = parser.parse_args()

    repo_root = Path.cwd()
    onnx_path = _resolve_path(repo_root, args.onnx)
    out_dir = _resolve_path(repo_root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = _load_manifest(_resolve_path(repo_root, args.manifest))
    positive_labels = {item.strip() for item in args.positive_labels.split(",") if item.strip()}
    videos = args.video[: args.max_videos] if args.video else _select_videos(grouped, args.max_videos, positive_labels)

    print("Loading ONNXRuntime detector")
    onnx_detector = UnifiedONNXPersonDetector(
        onnx_path=onnx_path,
        imgsz=args.imgsz,
        conf=args.conf,
        person_class_id=0,
        providers=["CPUExecutionProvider"],
    )
    print("Building RKNN simulator model")
    rknn_detector = RKNNSimulatorModelZooDetector(
        onnx_path=onnx_path,
        imgsz=onnx_detector.imgsz,
        conf=args.conf,
        person_class_id=0,
        do_quantization=args.do_quantization,
        dataset=str(_resolve_path(repo_root, args.dataset)) if args.dataset else None,
        algorithm=args.algorithm,
    )

    rows = []
    tensor_rows = []
    detections_jsonl = out_dir / "detections.jsonl"
    try:
        with detections_jsonl.open("w", encoding="utf-8") as det_fh:
            for video in videos:
                frames, _ = _sample_frames(_resolve_path(repo_root, video), args.frame_interval, args.max_frames_per_video)
                for frame_index, frame in frames:
                    tensor = onnx_detector._preprocess(frame)
                    started = time.perf_counter()
                    onnx_outputs = onnx_detector.session.run(onnx_detector.output_names, {onnx_detector.input_name: tensor})
                    onnx_ms = (time.perf_counter() - started) * 1000.0
                    onnx_dets = onnx_detector._postprocess_modelzoo(onnx_outputs)

                    rknn_dets, rknn_ms, rknn_outputs = rknn_detector.predict(frame)

                    for i, (onnx_out, rknn_out) in enumerate(zip(onnx_outputs, rknn_outputs)):
                        tensor_rows.append({
                            "video": video,
                            "frame_index": frame_index,
                            "tensor_index": i,
                            "onnx_stats": _tensor_stats(onnx_out),
                            "rknn_stats": _tensor_stats(rknn_out),
                            "diff": _compare_tensors(onnx_out, rknn_out),
                        })

                    ious = [_best_iou(det, rknn_dets) for det in onnx_dets]
                    row = {
                        "video": video,
                        "frame_index": frame_index,
                        "onnx_boxes": len(onnx_dets),
                        "rknn_boxes": len(rknn_dets),
                        "onnx_max_score": max([d.score for d in onnx_dets], default=0.0),
                        "rknn_max_score": max([d.score for d in rknn_dets], default=0.0),
                        "mean_best_iou_onnx_to_rknn": mean(ious) if ious else 0.0,
                        "onnx_ms": onnx_ms,
                        "rknn_sim_ms": rknn_ms,
                    }
                    rows.append(row)
                    det_fh.write(json.dumps({
                        **row,
                        "onnx_detections": [asdict(det) for det in onnx_dets],
                        "rknn_detections": [asdict(det) for det in rknn_dets],
                    }, ensure_ascii=False) + "\n")
                    image_name = f"{video.replace('/', '__')}_f{frame_index:05d}.jpg"
                    _draw_side_by_side(frame, onnx_dets, rknn_dets, f"{Path(video).name} f={frame_index}", out_dir / "frames" / image_name)
                    print(row)
    finally:
        rknn_detector.release()

    with (out_dir / "frame_compare.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "video",
            "frame_index",
            "onnx_boxes",
            "rknn_boxes",
            "onnx_max_score",
            "rknn_max_score",
            "mean_best_iou_onnx_to_rknn",
            "onnx_ms",
            "rknn_sim_ms",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "tensor_compare.json").write_text(json.dumps(tensor_rows, indent=2), encoding="utf-8")
    summary = {
        "onnx": str(onnx_path),
        "do_quantization": args.do_quantization,
        "algorithm": args.algorithm,
        "frames": len(rows),
        "mean_onnx_boxes": mean([r["onnx_boxes"] for r in rows]) if rows else 0.0,
        "mean_rknn_boxes": mean([r["rknn_boxes"] for r in rows]) if rows else 0.0,
        "mean_iou": mean([r["mean_best_iou_onnx_to_rknn"] for r in rows]) if rows else 0.0,
        "mean_onnx_score": mean([r["onnx_max_score"] for r in rows]) if rows else 0.0,
        "mean_rknn_score": mean([r["rknn_max_score"] for r in rows]) if rows else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:", summary)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
