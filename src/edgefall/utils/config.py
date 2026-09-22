"""Config loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from edgefall.utils.deps import require_module


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    yaml = require_module("yaml", "pip install PyYAML")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
