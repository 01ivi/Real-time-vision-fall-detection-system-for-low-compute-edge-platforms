"""ONNX Runtime wrapper for the GeometryTCN temporal head.

Provides a torch.nn.Module-compatible interface so the existing EdgeFallPipeline
can use an INT8-quantised ONNX temporal model without code changes.
"""

from __future__ import annotations

import gzip
from typing import Optional

import numpy as np


class ONNXTemporalHead:
    """Thin wrapper that runs a GeometryTCN ONNX model via ONNX Runtime.

    The wrapper accepts torch tensors (to stay compatible with EdgeFallPipeline)
    but runs inference through ONNX Runtime with numpy arrays.
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[list[str]] = None,
    ) -> None:
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
            self._session = ort.InferenceSession(model_bytes, providers=providers)
        else:
            self._session = ort.InferenceSession(onnx_path, providers=providers)

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def __call__(self, x) -> "torch.Tensor":  # noqa: F821
        """Forward pass returning a torch Tensor (compatible with EdgeFallPipeline).

        Args:
            x: torch.Tensor of shape (batch, clip_len, geometry_dim).

        Returns:
            torch.Tensor of shape (batch, num_classes) — raw logits.
        """
        import torch

        if isinstance(x, torch.Tensor):
            inp = x.detach().cpu().numpy().astype(np.float32)
        else:
            inp = np.asarray(x, dtype=np.float32)

        outputs = self._session.run([self._output_name], {self._input_name: inp})
        logits = torch.from_numpy(outputs[0])
        return logits

    def eval(self):
        """No-op for ONNX Runtime models (always in eval mode)."""
        return self

    def to(self, _device):
        """No-op — ONNX Runtime manages device placement internally."""
        return self
