"""Detector interfaces.

The temporal and state-machine pipeline is detector-agnostic. This module keeps
YOLO, ONNX Runtime, or vendor NPU wrappers behind one small interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0


class Detector(Protocol):
    def predict(self, frame) -> list[Detection]:
        ...


class UltralyticsPersonDetector:
    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        person_class_id: int = 0,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        from edgefall.utils.deps import require_module

        ultralytics = require_module("ultralytics", "pip install ultralytics")
        self.model = ultralytics.YOLO(weights)
        self.conf = conf
        self.person_class_id = person_class_id
        self.imgsz = imgsz
        self.device = device

    def _predict_kwargs(self) -> dict:
        predict_kwargs = {"conf": self.conf, "verbose": False, "classes": [self.person_class_id]}
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            predict_kwargs["device"] = self.device
        return predict_kwargs

    def _detections_from_result(self, result) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls != self.person_class_id:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=float(box.conf.item()), class_id=cls))
        return detections

    def predict(self, frame) -> list[Detection]:
        predict_kwargs = self._predict_kwargs()
        results = self.model.predict(frame, **predict_kwargs)
        return self._detections_from_result(results[0])

    def predict_batch(self, frames: list) -> list[list[Detection]]:
        if not frames:
            return []
        results = self.model.predict(frames, **self._predict_kwargs())
        return [self._detections_from_result(result) for result in results]
