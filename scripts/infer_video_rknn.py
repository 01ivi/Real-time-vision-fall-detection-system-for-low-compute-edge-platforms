#!/usr/bin/env python3
"""Run video fall-detection inference — both YOLO detector AND TCN on RK3588 NPU.

  YOLO yolo26m  → RKNN INT8 on NPU (backbone + detection head)
  GeometryTCN   → RKNN INT8 on NPU (temporal classification; FP16 is fallback)
  IoU tracker   → CPU
  State machine → CPU

This is the FULL NPU deployment variant.  All heavy neural-network
computation runs on the RK3588 NPU; only lightweight tracking,
state-machine logic, and coordinate remapping remain on CPU.

Example (on RK3588 board)
-------------------------
python3 scripts/infer_video_rknn.py \
  --video /path/to/video.mp4 \
  --detector-rknn runs/final/export/yolo26m_int8_imgsz960_modelzoo_sub100_rk3588.rknn \
  --temporal-rknn runs/final/export/temporal_head_int8_ln_addrelu_decomposed_cliprelu_b1_rk3588.rknn \
  --temporal-labels runs/final/export/temporal_labels.json \
  --output events.jsonl \
  --conf 0.02 \
  --imgsz 960
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgefall.infer.pipeline import EdgeFallPipeline
from edgefall.infer.state_machine import AlarmConfig
from edgefall.infer.tracker import IoUTracker
from edgefall.models.detector_rknn import RKNNDetector
from edgefall.models.temporal_tcn_rknn import RKNNTemporalHead
from edgefall.utils.deps import require_cv2, require_torch


def _load_checkpoint_config(checkpoint_path: str) -> dict:
    torch = require_torch()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {"config": ckpt["config"], "labels": ckpt["labels"]}


def _load_labels_from_json(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RK3588 NPU fall-detection — dual-NPU (YOLO + TCN) backend"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--detector-rknn", required=True,
        help="YOLO RKNN model (.rknn) — runs on NPU",
    )
    parser.add_argument(
        "--temporal-rknn", required=True,
        help="TCN RKNN model (.rknn) — runs on NPU",
    )
    parser.add_argument(
        "--temporal-checkpoint", default=None,
        help="PyTorch checkpoint for config/labels (alternative to --temporal-labels)",
    )
    parser.add_argument(
        "--temporal-labels", default=None,
        help="JSON list of class labels",
    )
    parser.add_argument("--output", required=True, help="JSONL alarm/event output")
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--write-all-events", action="store_true")
    # Tracker
    parser.add_argument("--tracker-iou-threshold", type=float, default=0.1)
    parser.add_argument("--tracker-min-hits", type=int, default=1)
    parser.add_argument("--tracker-max-age", type=int, default=75)
    # Alarm
    parser.add_argument("--min-fall-transition-frames", type=int, default=1)
    parser.add_argument("--allow-fall-transition-alarm", action="store_true", default=True)
    parser.add_argument("--fall-transition-alarm-frames", type=int, default=1)
    parser.add_argument("--fall-transition-score-threshold", type=float, default=0.0)
    parser.add_argument("--min-fallen-seconds", type=float, default=0.02)
    parser.add_argument("--allow-fallen-without-transition", action="store_true", default=True)
    parser.add_argument("--fallen-without-transition-seconds", type=float, default=0.02)
    parser.add_argument("--fallen-score-threshold", type=float, default=0.0)
    parser.add_argument("--descent-threshold", type=float, default=0.0)
    parser.add_argument("--aspect-change-threshold", type=float, default=0.0)
    parser.add_argument("--fallback-fps", type=float, default=25.0)
    args = parser.parse_args()

    cv2 = require_cv2()

    # ── Labels & config ─────────────────────────────────────────────────
    if args.temporal_labels:
        labels = _load_labels_from_json(args.temporal_labels)
        clip_len, sample_stride = 16, 1
    elif args.temporal_checkpoint:
        ckpt_info = _load_checkpoint_config(args.temporal_checkpoint)
        labels = ckpt_info["labels"]
        cfg = ckpt_info["config"]
        clip_len = int(cfg["data"]["clip_len"])
        sample_stride = int(cfg["data"].get("sample_stride", 1))
    else:
        raise SystemExit("Either --temporal-checkpoint or --temporal-labels required")

    print(f"Labels: {labels}")
    print(f"TCN clip_len={clip_len} stride={sample_stride}")

    # ── YOLO Detector (RKNN on NPU) ─────────────────────────────────────
    print(f"Loading detector (RKNN/NPU): {args.detector_rknn}")
    detector = RKNNDetector(
        rknn_path=args.detector_rknn,
        conf=args.conf,
        person_class_id=args.person_class_id,
        imgsz=args.imgsz,
        num_classes=args.num_classes,
    )

    # ── TCN (RKNN on NPU) ───────────────────────────────────────────────
    print(f"Loading TCN (RKNN/NPU): {args.temporal_rknn}")
    temporal_model = RKNNTemporalHead(args.temporal_rknn)

    # ── Video ───────────────────────────────────────────────────────────
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    print(f"Video: {args.video} ({video_fps:.1f} fps)")

    # ── Pipeline ────────────────────────────────────────────────────────
    pipeline = EdgeFallPipeline(
        detector=detector,
        temporal_model=temporal_model,
        labels=labels,
        clip_len=clip_len,
        sample_stride=sample_stride,
        tracker=IoUTracker(
            iou_threshold=float(args.tracker_iou_threshold),
            max_age=int(args.tracker_max_age),
            min_hits=int(args.tracker_min_hits),
        ),
        alarm_config=AlarmConfig(
            min_fall_transition_frames=int(args.min_fall_transition_frames),
            allow_fall_transition_alarm=bool(args.allow_fall_transition_alarm),
            min_fall_transition_alarm_frames=int(args.fall_transition_alarm_frames),
            fall_transition_score_threshold=float(args.fall_transition_score_threshold),
            min_fallen_seconds=float(args.min_fallen_seconds),
            allow_fallen_without_transition=bool(args.allow_fallen_without_transition),
            min_fallen_without_transition_seconds=float(args.fallen_without_transition_seconds),
            fallen_score_threshold=float(args.fallen_score_threshold),
            fps=video_fps or args.fallback_fps,
            descent_threshold=float(args.descent_threshold),
            aspect_change_threshold=float(args.aspect_change_threshold),
            fall_transition_label="fall",
        ),
        device="cpu",
    )

    # ── Run ─────────────────────────────────────────────────────────────
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        frame_index = 0
        alarm_count = 0
        with output.open("w", encoding="utf-8") as handle:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = pipeline.process_frame(frame, frame_index)
                for event in result.events:
                    if not event.alarm and not args.write_all_events:
                        continue
                    if event.alarm:
                        alarm_count += 1
                    handle.write(json.dumps({
                        "video": args.video,
                        "frame_index": frame_index,
                        "track_id": event.track_id,
                        "state": event.state.value,
                        "alarm": event.alarm,
                        "label": event.label,
                        "score": event.score,
                        "delta_y": event.delta_y,
                        "delta_aspect": event.delta_aspect,
                        "reason": event.reason,
                        "num_detections": result.num_detections,
                        "num_tracks": result.num_tracks,
                    }, ensure_ascii=False) + "\n")
                frame_index += 1
    finally:
        capture.release()
        try:
            detector.release()
        except Exception:
            pass
        try:
            temporal_model.release()
        except Exception:
            pass

    print(f"Done. {frame_index} frames, {alarm_count} alarms → {output}")


if __name__ == "__main__":
    main()
