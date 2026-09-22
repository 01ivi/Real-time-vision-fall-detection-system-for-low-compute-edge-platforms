"""Geometry features derived from tracked person boxes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1e-6)


def normalize_box(box: Box, image_width: int, image_height: int) -> Box:
    return Box(
        x1=box.x1 / max(image_width, 1),
        y1=box.y1 / max(image_height, 1),
        x2=box.x2 / max(image_width, 1),
        y2=box.y2 / max(image_height, 1),
    )


def geometry_sequence(boxes: list[Box], image_width: int, image_height: int) -> list[list[float]]:
    """Build per-frame geometry features.

    Features:
    cx, cy, w, h, aspect, delta_y, delta_aspect, velocity_y, acceleration_y, area
    """
    out: list[list[float]] = []
    prev_cy = 0.0
    prev_delta_y = 0.0
    prev_aspect = 0.0
    for idx, raw_box in enumerate(boxes):
        box = normalize_box(raw_box, image_width, image_height)
        delta_y = 0.0 if idx == 0 else box.cy - prev_cy
        velocity_y = delta_y
        acceleration_y = 0.0 if idx == 0 else delta_y - prev_delta_y
        delta_aspect = 0.0 if idx == 0 else box.aspect - prev_aspect
        out.append(
            [
                box.cx,
                box.cy,
                box.width,
                box.height,
                box.aspect,
                delta_y,
                delta_aspect,
                velocity_y,
                acceleration_y,
                box.width * box.height,
            ]
        )
        prev_cy = box.cy
        prev_delta_y = delta_y
        prev_aspect = box.aspect
    return out
