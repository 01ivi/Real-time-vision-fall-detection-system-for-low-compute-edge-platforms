"""Export the temporal head to ONNX."""

from __future__ import annotations

import argparse

from edgefall.models.temporal_tcn import build_geometry_tcn
from edgefall.utils.deps import require_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    labels = checkpoint["labels"]
    clip_len = int(cfg["data"]["clip_len"])
    geometry_dim = int(cfg["data"]["geometry_dim"])
    model = build_geometry_tcn(
        geometry_dim=geometry_dim,
        num_classes=len(labels),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dummy = torch.zeros(1, clip_len, geometry_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["geometry"],
        output_names=["logits"],
        dynamic_axes={"geometry": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX temporal head: {args.output}")


if __name__ == "__main__":
    main()
