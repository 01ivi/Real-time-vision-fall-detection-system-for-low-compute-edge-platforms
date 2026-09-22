"""PyTorch datasets for temporal fall classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from edgefall.data.manifest import ClipRecord, label_to_index, read_manifest
from edgefall.utils.deps import require_cv2, require_module, require_torch


class GeometryFeatureDataset:
    """Load precomputed geometry features from JSONL records.

    The preferred early baseline is a JSONL where each row includes `features`,
    a list shaped [T, geometry_dim]. This avoids coupling temporal training to a
    detector before the detector is ready.
    """

    def __init__(self, manifest_path: Union[str, Path], labels: list[str], clip_len: int, geometry_dim: int) -> None:
        torch = require_torch()
        self.torch = torch
        self.records = []
        self.labels = label_to_index(labels)
        self.clip_len = clip_len
        self.geometry_dim = geometry_dim

        import json

        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if "features" not in item:
                    raise ValueError("GeometryFeatureDataset expects each manifest row to include 'features'.")
                self.records.append(item)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        record = self.records[idx]
        features = record["features"]
        if len(features) < self.clip_len:
            pad = [features[-1] if features else [0.0] * self.geometry_dim] * (self.clip_len - len(features))
            features = features + pad
        features = features[: self.clip_len]
        x = self.torch.tensor(features, dtype=self.torch.float32)
        y = self.torch.tensor(self.labels[record["label"]], dtype=self.torch.long)
        return x, y


class VideoClipDataset:
    """Frame clip loader for future crop/image-model training."""

    def __init__(self, manifest_path: Union[str, Path], labels: list[str], clip_len: int, stride: int = 2) -> None:
        self.records: list[ClipRecord] = read_manifest(manifest_path)
        self.labels = label_to_index(labels)
        self.clip_len = clip_len
        self.stride = stride

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[list[Any], int]:
        cv2 = require_cv2()
        record = self.records[idx]
        capture = cv2.VideoCapture(record.video)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {record.video}")
        start = record.start_frame
        end = record.end_frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        frame_idx = start
        while len(frames) < self.clip_len:
            ok, frame = capture.read()
            if not ok:
                break
            if end is not None and frame_idx > end:
                break
            if (frame_idx - start) % self.stride == 0:
                frames.append(frame)
            frame_idx += 1
        capture.release()
        if not frames:
            np = require_module("numpy", "pip install numpy")
            frames = [np.zeros((288, 512, 3), dtype=np.uint8)]
        while len(frames) < self.clip_len:
            frames.append(frames[-1])
        return frames[: self.clip_len], self.labels[record.label]
