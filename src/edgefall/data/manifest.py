"""JSONL manifest parsing for video clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union


@dataclass(frozen=True)
class ClipRecord:
    video: str
    label: str
    domain: str
    start: Optional[float] = None
    end: Optional[float] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    subject_id: Optional[str] = None
    view: Optional[str] = None
    modality: str = "rgb"

    @classmethod
    def from_dict(cls, item: dict) -> "ClipRecord":
        if "video" not in item or "label" not in item:
            raise ValueError(f"Manifest item requires 'video' and 'label': {item}")
        return cls(
            video=str(item["video"]),
            label=str(item["label"]),
            domain=str(item.get("domain", "unknown")),
            start=None if item.get("start") in (None, "") else float(item["start"]),
            end=None if item.get("end") in (None, "") else float(item["end"]),
            start_frame=int(item.get("start_frame", 0)),
            end_frame=None if item.get("end_frame") in (None, "") else int(item["end_frame"]),
            subject_id=None if item.get("subject_id") in (None, "") else str(item["subject_id"]),
            view=None if item.get("view") in (None, "") else str(item["view"]),
            modality=str(item.get("modality", "rgb")),
        )

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "label": self.label,
            "domain": self.domain,
            "start": self.start,
            "end": self.end,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "subject_id": self.subject_id,
            "view": self.view,
            "modality": self.modality,
        }


def read_manifest(path: Union[str, Path]) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ClipRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Bad manifest line {line_no} in {path}: {line}") from exc
    return records


def write_manifest(records: Iterable[ClipRecord], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def label_to_index(labels: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(labels)}
