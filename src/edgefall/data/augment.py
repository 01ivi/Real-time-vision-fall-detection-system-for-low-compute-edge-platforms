"""Video-consistent augmentation primitives.

All random parameters are sampled once per clip and then applied to every frame,
which keeps the temporal signal coherent for action recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from edgefall.utils.deps import require_cv2, require_module


@dataclass
class ClipAugmentConfig:
    lowlight_prob: float = 0.25
    grayscale_prob: float = 0.20
    blur_prob: float = 0.15
    occlusion_prob: float = 0.20
    gamma_min: float = 0.55
    gamma_max: float = 1.40


class VideoConsistentAugment:
    def __init__(self, config: Optional[ClipAugmentConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or ClipAugmentConfig()
        self.rng = Random(seed)

    def __call__(self, frames: list[Any]) -> list[Any]:
        if not frames:
            return frames
        cv2 = require_cv2()
        np = require_module("numpy", "pip install numpy")
        cfg = self.config
        use_gray = self.rng.random() < cfg.grayscale_prob
        use_blur = self.rng.random() < cfg.blur_prob
        use_occ = self.rng.random() < cfg.occlusion_prob
        gamma = self.rng.uniform(cfg.gamma_min, cfg.gamma_max)
        if self.rng.random() < cfg.lowlight_prob:
            gamma = max(gamma, 1.25)

        h, w = frames[0].shape[:2]
        occ = None
        if use_occ:
            ow = int(w * self.rng.uniform(0.08, 0.22))
            oh = int(h * self.rng.uniform(0.08, 0.26))
            ox = self.rng.randint(0, max(0, w - ow))
            oy = self.rng.randint(0, max(0, h - oh))
            occ = (ox, oy, ow, oh)

        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        augmented = []
        for frame in frames:
            out = cv2.LUT(frame, table)
            if use_gray:
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            if use_blur:
                out = cv2.GaussianBlur(out, (5, 5), 0)
            if occ is not None:
                ox, oy, ow, oh = occ
                out[oy : oy + oh, ox : ox + ow] = 0
            augmented.append(out)
        return augmented
