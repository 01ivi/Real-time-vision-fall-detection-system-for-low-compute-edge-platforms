"""RKNN (rknn_toolkit_lite2) wrapper for YOLO person detector on RK3588 NPU.

Provides identical interface to ONNXPersonDetector so EdgeFallPipeline
works without changes.  NPU runs YOLO backbone+head; NMS + coordinate
remapping are done on CPU.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import cv2
import numpy as np


class _CDetection(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int),
    ]


def _load_cpp_postprocess():
    lib_path = os.environ.get("EDGEFALL_YOLO_POSTPROCESS_SO")
    candidates = []
    if lib_path:
        candidates.append(Path(lib_path))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "deploy_rk3588" / "libedgefall_yolo_postprocess.so")

    for candidate in candidates:
        if not candidate.exists():
            continue
        lib = ctypes.CDLL(str(candidate))
        lib.edgefall_yolo_postprocess.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(_CDetection),
            ctypes.c_int,
        ]
        lib.edgefall_yolo_postprocess.restype = ctypes.c_int
        try:
            lib.edgefall_yolo_postprocess_modelzoo.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(_CDetection),
                ctypes.c_int,
            ]
            lib.edgefall_yolo_postprocess_modelzoo.restype = ctypes.c_int
        except AttributeError:
            pass
        try:
            lib.edgefall_yolo_postprocess_modelzoo_v2.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(_CDetection),
                ctypes.c_int,
            ]
            lib.edgefall_yolo_postprocess_modelzoo_v2.restype = ctypes.c_int
        except AttributeError:
            pass
        return lib
    return None


class RKNNDetector:
    """YOLO person detector backed by RKNN on RK3588 NPU."""

    def __init__(
        self,
        rknn_path: str,
        conf: float = 0.25,
        person_class_id: int = 0,
        imgsz: int = 640,
        num_classes: int = 80,
    ) -> None:
        from rknnlite.api import RKNNLite

        self.conf = conf
        self.person_class_id = person_class_id
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.anchor_topk = 300
        self.final_topk = 300
        self.nms_iou_threshold = 0.7
        self.max_detections = 300
        self._cpp_postprocess = _load_cpp_postprocess()

        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {rknn_path} (code={ret})")
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN NPU runtime (code={ret})")

        self._last_scale: float = 1.0
        self._last_pad_left: float = 0.0
        self._last_pad_top: float = 0.0

    # ── Preprocessing (matches ultralytics + calibration prep) ──────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h, pad_w = self.imgsz - new_h, self.imgsz - new_w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left

        self._last_scale = scale
        self._last_pad_left = float(left)
        self._last_pad_top = float(top)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def _box_to_original(self, x1, y1, x2, y2):
        s = self._last_scale
        return (
            max(0.0, (x1 - self._last_pad_left) / s),
            max(0.0, (y1 - self._last_pad_top) / s),
            (x2 - self._last_pad_left) / s,
            (y2 - self._last_pad_top) / s,
        )

    # ── NMS ─────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(dets: list, iou_threshold: float = 0.7) -> list:
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d.score, reverse=True)
        keep = []
        suppressed = [False] * len(dets)

        def _iou(a, b) -> float:
            xa, ya = max(a.x1, b.x1), max(a.y1, b.y1)
            xb, yb = min(a.x2, b.x2), min(a.y2, b.y2)
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

    # ── Public API ──────────────────────────────────────────────────────

    def _postprocess_topk_cut_numpy(self, boxes: np.ndarray, class_scores: np.ndarray) -> list:
        """Equivalent CPU implementation of the original YOLO TopK tail.

        This is the fallback path when the C++ shared library is not present.
        RKNN output tensors are:
          boxes        [1, N, 4]
          class_scores [1, N, C]
        """
        from edgefall.models.detector import Detection

        boxes = np.squeeze(boxes, axis=0).astype(np.float32, copy=False)
        class_scores = np.squeeze(class_scores, axis=0).astype(np.float32, copy=False)
        anchor_scores = class_scores.max(axis=1)
        anchor_topk = min(self.anchor_topk, anchor_scores.shape[0])
        anchor_idx = np.argpartition(anchor_scores, -anchor_topk)[-anchor_topk:]
        anchor_idx = anchor_idx[np.argsort(anchor_scores[anchor_idx])[::-1]]

        selected_scores = class_scores[anchor_idx]
        flat_scores = selected_scores.reshape(-1)
        final_topk = min(self.final_topk, flat_scores.shape[0])
        final_idx = np.argpartition(flat_scores, -final_topk)[-final_topk:]
        final_idx = final_idx[np.argsort(flat_scores[final_idx])[::-1]]

        dets = []
        num_classes = class_scores.shape[1]
        for flat_idx in final_idx:
            score = float(flat_scores[flat_idx])
            selected_anchor_pos = int(flat_idx // num_classes)
            cls_id = int(flat_idx % num_classes)
            if score < self.conf or cls_id != self.person_class_id:
                continue
            anchor = int(anchor_idx[selected_anchor_pos])
            x1, y1, x2, y2 = boxes[anchor].tolist()
            ox1, oy1, ox2, oy2 = self._box_to_original(x1, y1, x2, y2)
            dets.append(Detection(
                x1=ox1, y1=oy1, x2=ox2, y2=oy2,
                score=score, class_id=cls_id,
            ))
        return self._nms(dets, self.nms_iou_threshold)

    def _postprocess_topk_cut_cpp(self, boxes: np.ndarray, class_scores: np.ndarray) -> list | None:
        if self._cpp_postprocess is None:
            return None

        from edgefall.models.detector import Detection

        boxes = np.ascontiguousarray(np.squeeze(boxes, axis=0), dtype=np.float32)
        class_scores = np.ascontiguousarray(np.squeeze(class_scores, axis=0), dtype=np.float32)
        if boxes.ndim != 2 or class_scores.ndim != 2 or boxes.shape[0] != class_scores.shape[0]:
            return None

        out = (_CDetection * self.max_detections)()
        count = self._cpp_postprocess.edgefall_yolo_postprocess(
            boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            class_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(boxes.shape[0]),
            int(class_scores.shape[1]),
            float(self.conf),
            int(self.person_class_id),
            int(self.anchor_topk),
            int(self.final_topk),
            float(self.nms_iou_threshold),
            float(self._last_scale),
            float(self._last_pad_left),
            float(self._last_pad_top),
            out,
            int(self.max_detections),
        )
        return [
            Detection(
                x1=float(out[i].x1),
                y1=float(out[i].y1),
                x2=float(out[i].x2),
                y2=float(out[i].y2),
                score=float(out[i].score),
                class_id=int(out[i].class_id),
            )
            for i in range(max(0, count))
        ]

    @staticmethod
    def _as_chw(output: np.ndarray, channels: int) -> np.ndarray:
        arr = np.squeeze(output, axis=0).astype(np.float32, copy=False)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D output after batch squeeze, got {arr.shape}")
        if arr.shape[0] == channels:
            return np.ascontiguousarray(arr)
        if arr.shape[-1] == channels:
            return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
        raise ValueError(f"Cannot infer CHW layout for shape {arr.shape}, channels={channels}")

    @staticmethod
    def _as_chw_any(output: np.ndarray, channels: tuple[int, ...]) -> tuple[np.ndarray, int]:
        arr = np.squeeze(output, axis=0).astype(np.float32, copy=False)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D output after batch squeeze, got {arr.shape}")
        for channel_count in channels:
            if arr.shape[0] == channel_count:
                return np.ascontiguousarray(arr), channel_count
            if arr.shape[-1] == channel_count:
                return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))), channel_count
        raise ValueError(f"Cannot infer CHW layout for shape {arr.shape}, channels={channels}")

    @staticmethod
    def _dfl_decode(box_logits: np.ndarray, reg_max: int = 16) -> np.ndarray:
        dist = box_logits.reshape(4, reg_max).astype(np.float32, copy=False)
        dist = dist - dist.max(axis=1, keepdims=True)
        prob = np.exp(dist)
        prob /= prob.sum(axis=1, keepdims=True)
        bins = np.arange(reg_max, dtype=np.float32)
        return (prob * bins[None, :]).sum(axis=1)

    def _postprocess_modelzoo_numpy(self, outputs: list[np.ndarray]) -> list:
        """Postprocess RKNN model-zoo YOLO outputs.

        Expected outputs are three branches of:
          box [1, 4 or 64, H, W], class scores [1, C, H, W],
          score_sum [1, 1, H, W].
        64-channel boxes are DFL logits. 4-channel boxes are already decoded
        distances. TopK/NMS run on CPU.
        """
        from edgefall.models.detector import Detection

        if len(outputs) < 9:
            raise ValueError(f"model-zoo YOLO expects 9 outputs, got {len(outputs)}")

        branches = []
        for i in range(3):
            box, box_channels = self._as_chw_any(outputs[i * 3], (4, 64))
            cls = self._as_chw(outputs[i * 3 + 1], self.num_classes)
            score_sum = self._as_chw(outputs[i * 3 + 2], 1)
            branches.append((box, box_channels, cls, score_sum))

        candidates = []
        for box, box_channels, cls, score_sum in branches:
            _, grid_h, grid_w = box.shape
            stride = self.imgsz / float(grid_h)
            score_gate = score_sum[0]
            ys, xs = np.where(score_gate >= self.conf)
            for y, x in zip(ys.tolist(), xs.tolist()):
                class_vec = cls[:, y, x]
                cls_id = int(np.argmax(class_vec))
                score = float(class_vec[cls_id])
                if cls_id != self.person_class_id or score < self.conf:
                    continue
                if box_channels == 4:
                    dist_l, dist_t, dist_r, dist_b = box[:, y, x].tolist()
                else:
                    dist_l, dist_t, dist_r, dist_b = self._dfl_decode(box[:, y, x], reg_max=16)
                x_center = x + 0.5
                y_center = y + 0.5
                x1 = (x_center - float(dist_l)) * stride
                y1 = (y_center - float(dist_t)) * stride
                x2 = (x_center + float(dist_r)) * stride
                y2 = (y_center + float(dist_b)) * stride
                candidates.append((score, cls_id, x1, y1, x2, y2))

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[: self.final_topk]

        dets = []
        for score, cls_id, x1, y1, x2, y2 in candidates:
            ox1, oy1, ox2, oy2 = self._box_to_original(x1, y1, x2, y2)
            dets.append(
                Detection(
                    x1=ox1,
                    y1=oy1,
                    x2=ox2,
                    y2=oy2,
                    score=score,
                    class_id=cls_id,
                )
            )
        return self._nms(dets, self.nms_iou_threshold)

    def _postprocess_modelzoo_cpp(self, outputs: list[np.ndarray]) -> list | None:
        if self._cpp_postprocess is None:
            return None

        from edgefall.models.detector import Detection

        try:
            b8, box_channels = self._as_chw_any(outputs[0], (4, 64))
            c8 = self._as_chw(outputs[1], self.num_classes)
            s8 = self._as_chw(outputs[2], 1)
            b16, box_channels16 = self._as_chw_any(outputs[3], (4, 64))
            c16 = self._as_chw(outputs[4], self.num_classes)
            s16 = self._as_chw(outputs[5], 1)
            b32, box_channels32 = self._as_chw_any(outputs[6], (4, 64))
            c32 = self._as_chw(outputs[7], self.num_classes)
            s32 = self._as_chw(outputs[8], 1)
        except Exception:
            return None

        if box_channels != box_channels16 or box_channels != box_channels32:
            return None

        out = (_CDetection * self.max_detections)()
        if hasattr(self._cpp_postprocess, "edgefall_yolo_postprocess_modelzoo_v2"):
            count = self._cpp_postprocess.edgefall_yolo_postprocess_modelzoo_v2(
                b8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(b8.shape[1]),
                int(b8.shape[2]),
                b16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(b16.shape[1]),
                int(b16.shape[2]),
                b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(b32.shape[1]),
                int(b32.shape[2]),
                int(self.imgsz),
                int(self.num_classes),
                16,
                int(box_channels),
                float(self.conf),
                int(self.person_class_id),
                int(self.final_topk),
                float(self.nms_iou_threshold),
                float(self._last_scale),
                float(self._last_pad_left),
                float(self._last_pad_top),
                out,
                int(self.max_detections),
            )
        elif box_channels == 64 and hasattr(self._cpp_postprocess, "edgefall_yolo_postprocess_modelzoo"):
            count = self._cpp_postprocess.edgefall_yolo_postprocess_modelzoo(
                b8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(b8.shape[1]),
                int(b8.shape[2]),
                b16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(b16.shape[1]),
                int(b16.shape[2]),
                b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(b32.shape[1]),
                int(b32.shape[2]),
                int(self.imgsz),
                int(self.num_classes),
                16,
                float(self.conf),
                int(self.person_class_id),
                int(self.final_topk),
                float(self.nms_iou_threshold),
                float(self._last_scale),
                float(self._last_pad_left),
                float(self._last_pad_top),
                out,
                int(self.max_detections),
            )
        else:
            return None
        return [
            Detection(
                x1=float(out[i].x1),
                y1=float(out[i].y1),
                x2=float(out[i].x2),
                y2=float(out[i].y2),
                score=float(out[i].score),
                class_id=int(out[i].class_id),
            )
            for i in range(max(0, count))
        ]

    def predict(self, frame: np.ndarray, max_raw: int = 3000) -> list:  # list[Detection]
        """Run NPU inference and return NMS-filtered Detection list.

        Supported YOLO RKNN detector output formats:
        - RKNN model-zoo format: 9 outputs, box/class/score_sum for strides 8/16/32.
          DFL/decode/NMS run in C++ when available, otherwise numpy.
        - TopK-cut final format: two outputs, boxes [1,N,4] and class_scores [1,N,C].
          The original TopK/Gather/NMS tail runs in C++ when available.
        - Older stripped format: one output [1,N,5] = [x1,y1,x2,y2,score].
        - Legacy format: one output [1,300,6] = [x1,y1,x2,y2,score,class_id].
        """
        from edgefall.models.detector import Detection

        inp = self._preprocess(frame)
        outputs = self._rknn.inference(inputs=[inp], data_format=["nchw"])

        if len(outputs) >= 9:
            dets = self._postprocess_modelzoo_cpp(outputs)
            if dets is not None:
                return dets
            return self._postprocess_modelzoo_numpy(outputs)

        if len(outputs) >= 2:
            dets = self._postprocess_topk_cut_cpp(outputs[0], outputs[1])
            if dets is not None:
                return dets
            return self._postprocess_topk_cut_numpy(outputs[0], outputs[1])

        raw = np.squeeze(outputs[0], axis=0)  # (N_all, 5) or (300, 6)

        if raw.shape[1] == 5:
            # TopK-stripped model: [x1, y1, x2, y2, score], no class_id
            # Sort by score and take top max_raw
            scores = raw[:, 4]
            if len(scores) > max_raw:
                top_indices = np.argpartition(scores, -max_raw)[-max_raw:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                raw = raw[top_indices]
            else:
                raw = raw[np.argsort(scores)[::-1]]

            dets = []
            for row in raw:
                x1, y1, x2, y2, score = row.tolist()
                if score < self.conf:
                    continue
                ox1, oy1, ox2, oy2 = self._box_to_original(x1, y1, x2, y2)
                dets.append(Detection(
                    x1=ox1, y1=oy1, x2=ox2, y2=oy2,
                    score=float(score), class_id=self.person_class_id,
                ))
        else:
            # Legacy format: (300, 6) = [x1, y1, x2, y2, score, class_id]
            dets = []
            for row in raw:
                x1, y1, x2, y2, score, cls_id = row.tolist()
                if score < self.conf or int(cls_id) != self.person_class_id:
                    continue
                ox1, oy1, ox2, oy2 = self._box_to_original(x1, y1, x2, y2)
                dets.append(Detection(
                    x1=ox1, y1=oy1, x2=ox2, y2=oy2,
                    score=float(score), class_id=int(cls_id),
                ))
        return self._nms(dets)

    def release(self):
        if hasattr(self, "_rknn") and self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self):
        self.release()
