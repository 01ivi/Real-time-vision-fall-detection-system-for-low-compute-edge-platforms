"""Fall alarm state machine.

The classifier predicts action stages; this module converts noisy frame-level
probabilities into stable event alarms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FallState(str, Enum):
    NORMAL = "normal"
    UNSTABLE = "unstable"
    FALLING = "falling"
    FALLEN_CONFIRMING = "fallen_confirming"
    ALARM = "alarm"
    RECOVERED = "recovered"


@dataclass
class AlarmConfig:
    min_fall_transition_frames: int = 3
    allow_fall_transition_alarm: bool = False
    min_fall_transition_alarm_frames: int = 8
    fall_transition_score_threshold: float = 0.0
    min_fallen_seconds: float = 1.0
    allow_fallen_without_transition: bool = False
    min_fallen_without_transition_seconds: float = 1.0
    fallen_score_threshold: float = 0.7
    fps: float = 25.0
    descent_threshold: float = 0.08
    aspect_change_threshold: float = 0.25
    fall_transition_label: str = "fall"
    fallen_label: str = "fallen"
    recovery_labels: tuple[str, ...] = ("walk", "standing", "sitting", "sit_down")


@dataclass
class AlarmEvent:
    track_id: int
    state: FallState
    alarm: bool
    score: float
    reason: str
    label: str = ""
    positive_score: float = 0.0
    delta_y: float = 0.0
    delta_aspect: float = 0.0


class FallStateMachine:
    def __init__(self, config: Optional[AlarmConfig] = None) -> None:
        self.config = config or AlarmConfig()
        self.state = FallState.NORMAL
        self.transition_count = 0
        self.fallen_count = 0
        self.best_score = 0.0

    def update(
        self,
        track_id: int,
        label: str,
        score: float,
        delta_y: float = 0.0,
        delta_aspect: float = 0.0,
    ) -> AlarmEvent:
        cfg = self.config
        self.best_score = max(self.best_score, score)
        falling_motion = delta_y >= cfg.descent_threshold or abs(delta_aspect) >= cfg.aspect_change_threshold

        is_fall_transition = label in {cfg.fall_transition_label, "fall_transition"}
        if is_fall_transition and falling_motion:
            self.transition_count += 1
            self.state = FallState.FALLING if self.transition_count >= cfg.min_fall_transition_frames else FallState.UNSTABLE
            if (
                cfg.allow_fall_transition_alarm
                and self.transition_count >= max(1, cfg.min_fall_transition_alarm_frames)
                and score >= cfg.fall_transition_score_threshold
            ):
                self.state = FallState.ALARM
                return AlarmEvent(track_id, self.state, True, self.best_score, "fall transition persisted", label, delta_y, delta_aspect)
            return AlarmEvent(track_id, self.state, False, self.best_score, "fall transition accumulating", label, delta_y, delta_aspect)

        if label == cfg.fallen_label:
            if self.state in {FallState.FALLING, FallState.FALLEN_CONFIRMING, FallState.ALARM}:
                self.fallen_count += 1
                required = max(1, int(cfg.min_fallen_seconds * cfg.fps))
                if self.fallen_count >= required:
                    self.state = FallState.ALARM
                    return AlarmEvent(track_id, self.state, True, self.best_score, "fallen persisted after transition", label, delta_y, delta_aspect)
                self.state = FallState.FALLEN_CONFIRMING
                return AlarmEvent(track_id, self.state, False, self.best_score, "fallen confirmation pending", label, delta_y, delta_aspect)
            if cfg.allow_fallen_without_transition and score >= cfg.fallen_score_threshold:
                self.fallen_count += 1
                required = max(1, int(cfg.min_fallen_without_transition_seconds * cfg.fps))
                if self.fallen_count >= required:
                    self.state = FallState.ALARM
                    return AlarmEvent(track_id, self.state, True, self.best_score, "fallen persisted without transition", label, delta_y, delta_aspect)
                self.state = FallState.FALLEN_CONFIRMING
                return AlarmEvent(track_id, self.state, False, self.best_score, "fallen-only confirmation pending", label, delta_y, delta_aspect)
            self.state = FallState.UNSTABLE
            return AlarmEvent(track_id, self.state, False, self.best_score, "fallen without transition", label, delta_y, delta_aspect)

        if label in cfg.recovery_labels:
            if self.state == FallState.ALARM:
                self.state = FallState.RECOVERED
            else:
                self.state = FallState.NORMAL
            self.transition_count = 0
            self.fallen_count = 0
            self.best_score = 0.0
            return AlarmEvent(track_id, self.state, False, score, "recovery or normal action", label, delta_y, delta_aspect)

        self.transition_count = max(0, self.transition_count - 1)
        self.fallen_count = max(0, self.fallen_count - 1)
        self.state = FallState.UNSTABLE if self.transition_count else FallState.NORMAL
        return AlarmEvent(track_id, self.state, False, self.best_score, "no fall evidence", label, delta_y, delta_aspect)
