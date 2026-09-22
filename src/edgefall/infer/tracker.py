"""Tiny IoU tracker suitable for edge fall-detection pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

from edgefall.models.detector import Detection


def iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


@dataclass
class Track:
    track_id: int
    box: Detection
    age: int = 0
    hits: int = 1
    history: list[Detection] = field(default_factory=list)

    def update(self, detection: Detection) -> None:
        self.box = detection
        self.age = 0
        self.hits += 1
        self.history.append(detection)

    def mark_missed(self) -> None:
        self.age += 1


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.35, max_age: int = 12, min_hits: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))
        pairs: list[tuple[float, int, int]] = []

        for track_idx, track in enumerate(self.tracks):
            for det_idx, detection in enumerate(detections):
                score = iou(track.box, detection)
                if score >= self.iou_threshold:
                    pairs.append((score, track_idx, det_idx))

        for _, track_idx, det_idx in sorted(pairs, reverse=True):
            if track_idx not in unmatched_tracks or det_idx not in unmatched_detections:
                continue
            self.tracks[track_idx].update(detections[det_idx])
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            self.tracks.append(Track(track_id=self.next_id, box=detection, history=[detection]))
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track.age <= self.max_age]
        return [track for track in self.tracks if track.hits >= self.min_hits and track.age == 0]
