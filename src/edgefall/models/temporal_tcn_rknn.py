"""RKNN (rknn_toolkit_lite2) wrapper for GeometryTCN on RK3588 NPU.

Provides a torch.nn.Module-compatible interface so the existing EdgeFallPipeline
can use an RKNN INT8 temporal model without code changes.

Unlike the ONNX wrapper, this uses rknn_toolkit_lite2 which is the
lightweight runtime library that ships on the RK3588 board.

Prerequisites (on RK3588 board):
  sudo apt install rknn-toolkit-lite2
  # or: pip install rknn-toolkit-lite2==1.6.0  (check Rockchip docs for latest)
"""

from __future__ import annotations

from typing import Optional


class RKNNTemporalHead:
    """Thin wrapper that runs a GeometryTCN .rknn model on RK3588 NPU.

    The wrapper accepts torch tensors (to stay compatible with EdgeFallPipeline)
    but runs inference through rknn_toolkit_lite2 with numpy arrays.

    Usage::

        from edgefall.models.temporal_tcn_rknn import RKNNTemporalHead

        tcn = RKNNTemporalHead("temporal_head_int8_ln_addrelu_decomposed_cliprelu_b1_rk3588.rknn")
        # model = tcn.eval()  # no-op
        logits = tcn(geometry_tensor)  # shape (1, 16, 10) -> (1, 16)
    """

    def __init__(self, rknn_path: str) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise ImportError(
                "rknn-toolkit-lite2 not installed. "
                "On RK3588 board: pip install rknn-toolkit-lite2"
            )

        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {rknn_path} (code={ret})")

        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime (code={ret})")

    def __call__(self, x) -> "torch.Tensor":  # noqa: F821
        """Forward pass returning a torch Tensor.

        Args:
            x: torch.Tensor of shape (batch, clip_len, geometry_dim).

        Returns:
            torch.Tensor of shape (batch, num_classes) — raw logits.
        """
        import numpy as np
        import torch

        if isinstance(x, torch.Tensor):
            inp = x.detach().cpu().numpy().astype(np.float32)
        else:
            inp = np.asarray(x, dtype=np.float32)

        # rknn_toolkit_lite2 expects a list of numpy arrays, one per input
        outputs = self._rknn.inference(inputs=[inp])
        # outputs is a list of numpy arrays, one per output
        logits = torch.from_numpy(outputs[0])
        return logits

    def eval(self):
        """No-op — RKNN models are always in eval mode."""
        return self

    def to(self, _device):
        """No-op — RKNN runtime manages device placement on NPU."""
        return self

    def release(self):
        """Free NPU resources."""
        if hasattr(self, "_rknn") and self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self):
        self.release()
