#!/usr/bin/env python3
"""Run final state-machine event inference from a manifest using RKNN models.

This is the RK3588 counterpart of scripts/infer_manifest_events.py. It emits
event intervals compatible with eval_alarm_metrics.py and
search_alarm_postprocess.py:

{"video": "...", "start": 12.0, "end": 14.2, "score": 0.91}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edgefall.infer.pipeline import EdgeFallPipeline
from edgefall.infer.state_machine import AlarmConfig
from edgefall.infer.tracker import IoUTracker
from edgefall.models.detector_rknn import RKNNDetector
from edgefall.models.temporal_tcn_rknn import RKNNTemporalHead
from edgefall.utils.deps import require_cv2


def unique_videos_from_manifest(path: str) -> list[str]:
    videos = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            video = str(item["video"])
            if video not in seen:
                seen.add(video)
                videos.append(video)
    return videos


def load_labels(path: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close_event(events: list[dict], current: dict | None, fps: float, merge_gap_frames: int) -> None:
    if current is None:
        return
    start_frame = int(current["start_frame"])
    end_frame = int(current["end_frame"])
    end_frame = max(start_frame, end_frame - merge_gap_frames)
    events.append(
        {
            "video": current["video"],
            "start": start_frame / fps,
            "end": max((end_frame + 1) / fps, start_frame / fps + 1.0 / fps),
            "score": float(current["score"]),
            "track_id": current["track_id"],
        }
    )


def run_video(video: str, detector, model, labels: list[str], args: argparse.Namespace) -> list[dict]:
    cv2 = require_cv2()
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or float(args.fps)
    pipeline = EdgeFallPipeline(
        detector=detector,
        temporal_model=model,
        labels=labels,
        clip_len=int(args.clip_len),
        sample_stride=int(args.sample_stride),
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
            fps=fps,
            descent_threshold=float(args.descent_threshold),
            aspect_change_threshold=float(args.aspect_change_threshold),
            fall_transition_label="fall",
        ),
        device="cpu",
    )

    events = []
    active_by_track: dict[int, dict] = {}
    gap_by_track: dict[int, int] = {}
    frame_index = 0

    def update_active_events(result, frame_index: int) -> None:
        alarm_tracks = set()
        for event in result.events:
            if not event.alarm:
                continue
            alarm_tracks.add(event.track_id)
            if event.track_id not in active_by_track:
                active_by_track[event.track_id] = {
                    "video": video,
                    "start_frame": frame_index,
                    "end_frame": frame_index,
                    "score": float(event.score),
                    "track_id": event.track_id,
                }
            else:
                active = active_by_track[event.track_id]
                active["end_frame"] = frame_index
                active["score"] = max(float(active["score"]), float(event.score))
            gap_by_track[event.track_id] = 0

        for track_id in list(active_by_track):
            if track_id in alarm_tracks:
                continue
            gap_by_track[track_id] = gap_by_track.get(track_id, 0) + 1
            active_by_track[track_id]["end_frame"] = frame_index
            if gap_by_track[track_id] > args.merge_gap_frames:
                close_event(events, active_by_track.pop(track_id), fps, args.merge_gap_frames)
                gap_by_track.pop(track_id, None)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process_frame(frame, frame_index)
            update_active_events(result, frame_index)
            frame_index += 1
    finally:
        capture.release()

    for track_id in list(active_by_track):
        close_event(events, active_by_track.pop(track_id), fps, 0)
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detector-rknn", required=True)
    parser.add_argument("--temporal-rknn", required=True)
    parser.add_argument("--temporal-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--tracker-iou-threshold", type=float, default=0.1)
    parser.add_argument("--tracker-min-hits", type=int, default=1)
    parser.add_argument("--tracker-max-age", type=int, default=75)
    parser.add_argument("--min-fall-transition-frames", type=int, default=2)
    parser.add_argument("--allow-fall-transition-alarm", action="store_true", default=True)
    parser.add_argument("--fall-transition-alarm-frames", type=int, default=2)
    parser.add_argument("--fall-transition-score-threshold", type=float, default=0.0)
    parser.add_argument("--min-fallen-seconds", type=float, default=0.04)
    parser.add_argument("--allow-fallen-without-transition", action="store_true", default=True)
    parser.add_argument("--fallen-without-transition-seconds", type=float, default=0.04)
    parser.add_argument("--fallen-score-threshold", type=float, default=0.2)
    parser.add_argument("--descent-threshold", type=float, default=0.04)
    parser.add_argument("--aspect-change-threshold", type=float, default=0.15)
    parser.add_argument("--merge-gap-frames", type=int, default=75)
    args = parser.parse_args()

    labels = load_labels(args.temporal_labels)
    detector = RKNNDetector(
        rknn_path=args.detector_rknn,
        conf=args.conf,
        person_class_id=args.person_class_id,
        imgsz=args.imgsz,
        num_classes=args.num_classes,
    )
    model = RKNNTemporalHead(args.temporal_rknn)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_events = []
    try:
        for index, video in enumerate(unique_videos_from_manifest(args.manifest), start=1):
            print(f"[{index}] {video}", flush=True)
            all_events.extend(run_video(video, detector, model, labels, args))
    finally:
        detector.release()
        model.release()

    all_events.sort(key=lambda item: (item["video"], item["start"], -item["score"]))
    with output.open("w", encoding="utf-8") as handle:
        for event in all_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_events)} events to {output}")


if __name__ == "__main__":
    main()
