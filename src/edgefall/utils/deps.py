"""Small dependency helpers with readable errors."""

from __future__ import annotations

import importlib
from typing import Any, Optional


def require_module(name: str, install_hint: Optional[str] = None) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        hint = install_hint or f"pip install {name}"
        raise RuntimeError(f"Missing optional dependency '{name}'. Install it with: {hint}") from exc


def require_torch() -> Any:
    return require_module("torch", "pip install -r requirements.txt")


def require_cv2() -> Any:
    return require_module("cv2", "pip install -r requirements.txt")
