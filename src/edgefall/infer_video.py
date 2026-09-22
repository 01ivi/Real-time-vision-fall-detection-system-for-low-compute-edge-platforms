"""Run video inference with YOLO person detection and the temporal state machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgefall.infer.pipeline import EdgeFallPipeline
from edgefall.infer.state_machine import AlarmConfig
from edgefall.infer.tracker import IoUTracker
from edgefall.models.detector import UltralyticsPersonDetector
from edgefall.models.temporal_tcn import build_geometry_tcn
from edgefall.utils.deps import require_cv2, require_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector-weights", required=True, help="YOLO weights, e.g. yolov8n.pt")
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="JSONL alarm/event output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--debug-output", default=None, help="Optional JSONL with every TCN/state-machine event, not only alarms")
    parser.add_argument("--write-all-events", action="store_true", help="Write non-alarm events to --output too")
    parser.add_argument("--tracker-iou-threshold", type=float, default=None)
    parser.add_argument("--tracker-min-hits", type=int, default=None)
    parser.add_argument("--min-fall-transition-frames", type=int, default=None)
    parser.add_argument("--allow-fall-transition-alarm", action="store_true")
    parser.add_argument("--fall-transition-alarm-frames", type=int, default=None)
    parser.add_argument("--fall-transition-score-threshold", type=float, default=None)
    parser.add_argument("--min-fallen-seconds", type=float, default=None)
    parser.add_argument("--allow-fallen-without-transition", action="store_true")
    parser.add_argument("--fallen-without-transition-seconds", type=float, default=None)
    parser.add_argument("--fallen-score-threshold", type=float, default=None)
    parser.add_argument("--descent-threshold", type=float, default=None)
    parser.add_argument("--aspect-change-threshold", type=float, default=None)
    args = parser.parse_args()

    torch = require_torch()
    cv2 = require_cv2()
    checkpoint = torch.load(args.temporal_checkpoint, map_location=args.device)
    cfg = checkpoint["config"]
    labels = checkpoint["labels"]

    model = build_geometry_tcn(
        geometry_dim=int(cfg["data"]["geometry_dim"]),
        num_classes=len(labels),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    detector = UltralyticsPersonDetector(args.detector_weights, conf=args.conf, person_class_id=args.person_class_id)
    tracker_cfg = cfg.get("inference", {}).get("tracker", {})
    alarm_cfg = cfg.get("inference", {}).get("alarm", {})
    tracker_iou_threshold = args.tracker_iou_threshold
    if tracker_iou_threshold is None:
        tracker_iou_threshold = float(tracker_cfg.get("iou_threshold", 0.35))
    tracker_min_hits = args.tracker_min_hits
    if tracker_min_hits is None:
        tracker_min_hits = int(tracker_cfg.get("min_hits", 2))
    min_fall_transition_frames = args.min_fall_transition_frames
    if min_fall_transition_frames is None:
        min_fall_transition_frames = int(alarm_cfg.get("min_fall_transition_frames", 3))
    fall_transition_alarm_frames = args.fall_transition_alarm_frames
    if fall_transition_alarm_frames is None:
        fall_transition_alarm_frames = int(alarm_cfg.get("min_fall_transition_alarm_frames", 8))
    fall_transition_score_threshold = args.fall_transition_score_threshold
    if fall_transition_score_threshold is None:
        fall_transition_score_threshold = float(alarm_cfg.get("fall_transition_score_threshold", 0.0))
    min_fallen_seconds = args.min_fallen_seconds
    if min_fallen_seconds is None:
        min_fallen_seconds = float(alarm_cfg.get("min_fallen_seconds", 1.0))
    fallen_without_transition_seconds = args.fallen_without_transition_seconds
    if fallen_without_transition_seconds is None:
        fallen_without_transition_seconds = float(alarm_cfg.get("min_fallen_without_transition_seconds", 1.0))
    fallen_score_threshold = args.fallen_score_threshold
    if fallen_score_threshold is None:
        fallen_score_threshold = float(alarm_cfg.get("fallen_score_threshold", 0.7))
    descent_threshold = args.descent_threshold
    if descent_threshold is None:
        descent_threshold = float(alarm_cfg.get("descent_threshold", 0.08))
    aspect_change_threshold = args.aspect_change_threshold
    if aspect_change_threshold is None:
        aspect_change_threshold = float(alarm_cfg.get("aspect_change_threshold", 0.25))
    pipeline = EdgeFallPipeline(
        detector=detector,
        temporal_model=model,
        labels=labels,
        clip_len=int(cfg["data"]["clip_len"]),
        sample_stride=int(cfg["data"].get("sample_stride", 1)),
        tracker=IoUTracker(
            iou_threshold=float(tracker_iou_threshold),
            max_age=int(tracker_cfg.get("max_age", 12)),
            min_hits=int(tracker_min_hits),
        ),
        alarm_config=AlarmConfig(
            min_fall_transition_frames=int(min_fall_transition_frames),
            allow_fall_transition_alarm=bool(args.allow_fall_transition_alarm or alarm_cfg.get("allow_fall_transition_alarm", False)),
            min_fall_transition_alarm_frames=int(fall_transition_alarm_frames),
            fall_transition_score_threshold=float(fall_transition_score_threshold),
            min_fallen_seconds=float(min_fallen_seconds),
            allow_fallen_without_transition=bool(args.allow_fallen_without_transition or alarm_cfg.get("allow_fallen_without_transition", False)),
            min_fallen_without_transition_seconds=float(fallen_without_transition_seconds),
            fallen_score_threshold=float(fallen_score_threshold),
            fps=video_fps or float(alarm_cfg.get("fps", 25.0)),
            descent_threshold=float(descent_threshold),
            aspect_change_threshold=float(aspect_change_threshold),
            fall_transition_label=str(alarm_cfg.get("fall_transition_label", "fall")),
        ),
        device=args.device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_handle = None
    if args.debug_output is not None:
        debug_output = Path(args.debug_output)
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        debug_handle = debug_output.open("w", encoding="utf-8")
    frame_index = 0
    alarm_count = 0
    with output.open("w", encoding="utf-8") as handle:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = pipeline.process_frame(frame, frame_index)
                if debug_handle is not None and not result.events:
                    debug_handle.write(
                        json.dumps(
                            {
                                "video": args.video,
                                "frame_index": frame_index,
                                "num_detections": result.num_detections,
                                "num_tracks": result.num_tracks,
                                "event": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                for event in result.events:
                    item = {
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
                    }
                    if debug_handle is not None:
                        debug_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    if not event.alarm and not args.write_all_events:
                        continue
                    if event.alarm:
                        alarm_count += 1
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                frame_index += 1
        finally:
            if debug_handle is not None:
                debug_handle.close()
    capture.release()
    print(f"Processed {frame_index} frames, wrote {alarm_count} alarm rows to {output}")


if __name__ == "__main__":
    main()
