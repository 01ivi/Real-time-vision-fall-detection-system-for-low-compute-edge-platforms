"""ONNX Runtime person detector.

Replaces UltralyticsPersonDetector for deployment scenarios where PyTorch /
ultralytics is not available or the model must run via ONNX Runtime (e.g. after
INT8 quantization).

Supports plain .onnx and gzipped .onnx.gz models.
"""

from __future__ import annotations

import gzip
import os
from typing import Optional

import numpy as np

from edgefall.models.detector import Detection


def _load_onnx_session(onnx_path: str, providers: Optional[list[str]] = None):
    """Load an ONNX Runtime session, optionally from a gzipped file."""
    import onnxruntime as ort

    if providers is None:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    if onnx_path.endswith(".gz"):
        with gzip.open(onnx_path, "rb") as fh:
            model_bytes = fh.read()
        return ort.InferenceSession(model_bytes, providers=providers)
    return ort.InferenceSession(onnx_path, providers=providers)


class ONNXPersonDetector:
    """Person detector backed by an ONNX YOLO model + ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str,
        conf: float = 0.25,
        person_class_id: int = 0,
        imgsz: int = 640,
        providers: Optional[list[str]] = None,
    ) -> None:
        self.conf = conf
        self.person_class_id = person_class_id
        self.imgsz = imgsz
        self.session = _load_onnx_session(onnx_path, providers)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        # Cached per-frame letterbox params for coordinate mapping
        self._last_scale: float = 1.0
        self._last_pad_left: float = 0.0
        self._last_pad_top: float = 0.0

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 HWC -> RGB float32 NCHW, letterboxed & normalized to [0,1].

        Matches ultralytics YOLO preprocessing: letterbox resize (preserving
        aspect ratio with gray padding), BGR→RGB, normalize, CHW.
        Caches scale/pad for _box_to_original.
        """
        import cv2

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        self._last_scale = scale
        self._last_pad_left = float(left)
        self._last_pad_top = float(top)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        array = rgb.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, axis=0)

    def _box_to_original(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        """Map detection box from letterboxed 640×640 back to original frame coords."""
        s = self._last_scale
        return (
            max(0.0, (x1 - self._last_pad_left) / s),
            max(0.0, (y1 - self._last_pad_top) / s),
            (x2 - self._last_pad_left) / s,
            (y2 - self._last_pad_top) / s,
        )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection on a single BGR frame."""
        tensor = self._preprocess(frame)
        outputs = self.session.run([self._output_name], {self._input_name: tensor})
        dets = outputs[0]  # shape (1, 300, 6)
        return self._postprocess(dets)

    def _nms(self, dets: list[Detection], iou_threshold: float = 0.7) -> list[Detection]:
        """Greedy IoU-based NMS, keeping highest-score boxes first."""
        if not dets:
            return []
        # Sort by score descending
        dets = sorted(dets, key=lambda d: d.score, reverse=True)
        keep: list[Detection] = []
        suppressed = [False] * len(dets)

        def _iou(a: Detection, b: Detection) -> float:
            xa = max(a.x1, b.x1)
            ya = max(a.y1, b.y1)
            xb = min(a.x2, b.x2)
            yb = min(a.y2, b.y2)
            inter = max(0.0, xb - xa) * max(0.0, yb - ya)
            area_a = max(0.0, (a.x2 - a.x1) * (a.y2 - a.y1))
            area_b = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        for i, det in enumerate(dets):
            if suppressed[i]:
                continue
            keep.append(det)
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                if _iou(det, dets[j]) > iou_threshold:
                    suppressed[j] = True
        return keep

    def _postprocess(self, dets: np.ndarray) -> list[Detection]:
        """Convert raw ONNX output to Detection list, with NMS and conf/class filter.

        Boxes are mapped from letterboxed 640×640 space back to original frame
        coordinates via the cached scale/pad params from _preprocess.
        """
        dets = np.squeeze(dets, axis=0)  # (300, 6)
        raw: list[Detection] = []
        for row in dets:
            x1, y1, x2, y2, score, cls_id = row.tolist()
            if score < self.conf:
                continue
            if int(cls_id) != self.person_class_id:
                continue
            ox1, oy1, ox2, oy2 = self._box_to_original(float(x1), float(y1), float(x2), float(y2))
            raw.append(
                Detection(
                    x1=ox1,
                    y1=oy1,
                    x2=ox2,
                    y2=oy2,
                    score=float(score),
                    class_id=int(cls_id),
                )
            )
        return self._nms(raw)

    def predict_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run person detection on a batch of BGR frames."""
        return [self.predict(frame) for frame in frames]
