"""End-to-end inference pipeline."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from edgefall.data.geometry import Box, geometry_sequence
from edgefall.infer.state_machine import AlarmConfig, AlarmEvent, FallStateMachine
from edgefall.infer.tracker import IoUTracker
from edgefall.models.detector import Detector
from edgefall.utils.deps import require_torch


@dataclass
class PipelineResult:
    frame_index: int
    events: list[AlarmEvent]
    num_detections: int = 0
    num_tracks: int = 0


class EdgeFallPipeline:
    def __init__(
        self,
        detector: Detector,
        temporal_model,
        labels: list[str],
        clip_len: int = 16,
        sample_stride: int = 1,
        image_size: tuple[int, int] = (288, 512),
        tracker: Optional[IoUTracker] = None,
        alarm_config: Optional[AlarmConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.detector = detector
        self.temporal_model = temporal_model
        self.labels = labels
        self.clip_len = clip_len
        self.sample_stride = max(1, int(sample_stride))
        self.image_size = image_size
        self.tracker = tracker or IoUTracker()
        self.device = device
        self.torch = require_torch()
        self.positive_indices = [idx for idx, label in enumerate(labels) if label in {"fall", "fallen"}]
        self.history_len = max(clip_len, clip_len * self.sample_stride)
        self.histories = defaultdict(lambda: deque(maxlen=self.history_len))
        self.machines: dict[int, FallStateMachine] = defaultdict(lambda: FallStateMachine(alarm_config))

    def process_frame(self, frame, frame_index: int) -> PipelineResult:
        detections = self.detector.predict(frame)
        return self.process_detections(detections, frame.shape[:2], frame_index)

    def process_detections(self, detections, frame_shape: tuple[int, int], frame_index: int) -> PipelineResult:
        tracks = self.tracker.update(detections)
        h, w = frame_shape
        events: list[AlarmEvent] = []

        for track in tracks:
            self.histories[track.track_id].append(Box(track.box.x1, track.box.y1, track.box.x2, track.box.y2))
            if len(self.histories[track.track_id]) < self.history_len:
                continue
            history = list(self.histories[track.track_id])
            sampled = history[-self.history_len :: self.sample_stride]
            features = geometry_sequence(sampled[-self.clip_len :], w, h)
            x = self.torch.tensor([features], dtype=self.torch.float32, device=self.device)
            with self.torch.no_grad():
                logits = self.temporal_model(x)
                probs = self.torch.softmax(logits, dim=-1)[0]
                class_idx = int(self.torch.argmax(probs).item())
                if self.positive_indices:
                    score = float(probs[self.positive_indices].max().item())
                else:
                    score = float(probs[class_idx].item())
            label = self.labels[class_idx]
            delta_y = features[-1][5]
            delta_aspect = features[-1][6]
            event = self.machines[track.track_id].update(track.track_id, label, score, delta_y, delta_aspect)
            event.positive_score = score
            events.append(event)

        return PipelineResult(frame_index=frame_index, events=events, num_detections=len(detections), num_tracks=len(tracks))
