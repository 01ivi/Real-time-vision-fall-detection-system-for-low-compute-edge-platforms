#!/usr/bin/env python3
"""RK3588 DUAL-NPU video inference with real-time bounding-box & alarm visualisation.

Both YOLO detector AND GeometryTCN run on the RK3588 NPU.
Draws per-person boxes, state-machine status, and alarm overlays.

Example (on RK3588 board)
-------------------------
python3 scripts/infer_video_rknn_visual.py \
  --video /path/to/video.mp4 \
  --detector-rknn runs/final/export/yolo26m_int8_imgsz960_modelzoo_sub100_rk3588.rknn \
  --temporal-rknn runs/final/export/temporal_head_int8_ln_addrelu_decomposed_cliprelu_b1_rk3588.rknn \
  --temporal-labels runs/final/export/temporal_labels.json \
  --output-video out/fall_annotated_rknn.mp4 \
  --output-events out/events_rknn.jsonl \
  --conf 0.02 \
  --imgsz 960
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from edgefall.infer.pipeline import EdgeFallPipeline
from edgefall.infer.state_machine import AlarmConfig
from edgefall.infer.tracker import IoUTracker
from edgefall.models.detector_rknn import RKNNDetector
from edgefall.models.temporal_tcn_rknn import RKNNTemporalHead
from edgefall.utils.deps import require_torch

# ── Colours (BGR) ───────────────────────────────────────────────────────
COLOR_NORMAL   = (0, 255, 0)     # green
COLOR_UNSTABLE = (0, 255, 255)   # yellow
COLOR_ALARM    = (0, 0, 255)     # red
COLOR_FLASH    = (0, 0, 200)     # dark red
COLOR_TEXT     = (255, 255, 255)

FLASH_INTERVAL = 15


def _put_text_with_bg(img, text, org, font_scale=0.6, thickness=2,
                      color=COLOR_TEXT, bg_color=(0, 0, 0)):
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (org[0], org[1] - th - baseline),
                  (org[0] + tw, org[1] + baseline), bg_color, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)


def _load_labels(args) -> list[str]:
    if args.temporal_labels:
        with open(args.temporal_labels, "r", encoding="utf-8") as fh:
            return json.load(fh)
    if args.temporal_checkpoint:
        torch = require_torch()
        ckpt = torch.load(args.temporal_checkpoint, map_location="cpu",
                          weights_only=False)
        return ckpt["labels"]
    raise SystemExit("Need --temporal-labels or --temporal-checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RK3588 dual-NPU fall detection with visual output")
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector-rknn", required=True,
                        help="YOLO RKNN model (.rknn) — NPU")
    parser.add_argument("--temporal-rknn", required=True,
                        help="TCN RKNN model (.rknn) — NPU")
    parser.add_argument("--temporal-checkpoint", default=None)
    parser.add_argument("--temporal-labels", default=None)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-events", default=None)
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=80,
                        help="Number of classes in YOLO detector model (1 for person-only, 80 for COCO)")
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
    # Display
    parser.add_argument("--alarm-persist-frames", type=int, default=60)
    args = parser.parse_args()

    labels = _load_labels(args)

    # ── open video ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fallback_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = Path(args.output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    video_writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", format="ffmpeg",
        output_params=["-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"],
    )

    # ── NPU models ─────────────────────────────────────────────────────
    print(f"Detector (RKNN/NPU): {args.detector_rknn}")
    print(f"TCN      (RKNN/NPU): {args.temporal_rknn}")

    detector = RKNNDetector(
        rknn_path=args.detector_rknn, conf=args.conf,
        person_class_id=args.person_class_id, imgsz=args.imgsz,
        num_classes=args.num_classes,
    )
    temporal_model = RKNNTemporalHead(args.temporal_rknn)

    # ── Pipeline ───────────────────────────────────────────────────────
    pipeline = EdgeFallPipeline(
        detector=detector, temporal_model=temporal_model,
        labels=labels, clip_len=16, sample_stride=1,
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
            fps=fps or args.fallback_fps,
            descent_threshold=float(args.descent_threshold),
            aspect_change_threshold=float(args.aspect_change_threshold),
            fall_transition_label="fall",
        ),
        device="cpu",
    )

    # ── Event writer ────────────────────────────────────────────────────
    event_handle = None
    if args.output_events:
        events_path = Path(args.output_events)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        event_handle = events_path.open("w", encoding="utf-8")

    # ── State ───────────────────────────────────────────────────────────
    frame_index = 0
    alarm_count = 0
    last_alarm_frame = -9999
    flash_toggle = False
    track_boxes: dict[int, tuple] = {}

    print(f"\nProcessing {args.video} ({width}x{height}, {fps:.1f} fps, ~{total_frames} frames)")
    print(f"Output video  → {args.output_video}")
    if args.output_events:
        print(f"Output events → {args.output_events}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.predict(frame)
            result = pipeline.process_detections(detections, frame.shape[:2], frame_index)

            for track in pipeline.tracker.tracks:
                if track.hits >= pipeline.tracker.min_hits:
                    b = track.box
                    track_boxes[track.track_id] = (b.x1, b.y1, b.x2, b.y2)

            current_alarm = False
            for event in result.events:
                tid = event.track_id
                is_alarm = bool(event.alarm)
                if is_alarm:
                    current_alarm = True
                    last_alarm_frame = frame_index

                if is_alarm:
                    box_color = COLOR_ALARM
                elif event.state.value == "unstable":
                    box_color = COLOR_UNSTABLE
                else:
                    box_color = COLOR_NORMAL

                if tid in track_boxes:
                    bx1, by1, bx2, by2 = track_boxes[tid]
                    cv2.rectangle(frame, (int(bx1), int(by1)),
                                  (int(bx2), int(by2)), box_color, 2)
                    lbl = f"ID:{tid} {event.state.value} | {event.label} ({event.score:.3f})"
                    _put_text_with_bg(
                        frame, lbl, (int(bx1), max(0, int(by1) - 8)),
                        font_scale=0.45, thickness=1,
                        color=COLOR_TEXT, bg_color=box_color,
                    )

            # ── Alarm overlay ────────────────────────────────────────────
            frames_since = frame_index - last_alarm_frame
            if current_alarm or frames_since < args.alarm_persist_frames:
                if frame_index % FLASH_INTERVAL < FLASH_INTERVAL // 2:
                    flash_toggle = not flash_toggle
                if flash_toggle and current_alarm:
                    cv2.rectangle(frame, (0, 0), (width - 1, height - 1),
                                  COLOR_FLASH, 8)

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 200), -1)
                frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

                alarm_text = "!!! FALL DETECTED !!!" if current_alarm else "(recent alarm)"
                (tw, _), _ = cv2.getTextSize(alarm_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.putText(frame, alarm_text, ((width - tw) // 2, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3)

                if current_alarm:
                    alarm_count += 1
                _put_text_with_bg(
                    frame, f"Alarms: {alarm_count}  [NPU×2]", (10, height - 15),
                    font_scale=0.6, thickness=2,
                    color=COLOR_TEXT, bg_color=(0, 0, 200),
                )

            # ── Frame info ───────────────────────────────────────────────
            info = f"Frame: {frame_index}"
            if total_frames:
                info += f" / {total_frames}"
            (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            _put_text_with_bg(frame, info, (width - tw - 10, height - 15),
                              font_scale=0.5, thickness=1,
                              color=COLOR_TEXT, bg_color=(50, 50, 50))

            # ── Write events ─────────────────────────────────────────────
            if event_handle is not None:
                for event in result.events:
                    if not event.alarm:
                        continue
                    event_handle.write(json.dumps({
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
                    }, ensure_ascii=False) + "\n")

            video_writer.append_data(frame[:, :, ::-1])
            frame_index += 1

            if frame_index % 100 == 0:
                print(f"  ... {frame_index} frames")

    finally:
        cap.release()
        video_writer.close()
        if event_handle is not None:
            event_handle.close()
        try:
            detector.release()
        except Exception:
            pass
        try:
            temporal_model.release()
        except Exception:
            pass

    print(f"Done. {frame_index} frames → {args.output_video}, {alarm_count} alarm frames.")


if __name__ == "__main__":
    main()
