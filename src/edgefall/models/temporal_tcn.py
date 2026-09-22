"""Lightweight temporal models for action-stage classification."""

from __future__ import annotations

from edgefall.utils.deps import require_torch


def build_geometry_tcn(geometry_dim: int, num_classes: int, hidden_dim: int = 96, num_layers: int = 3, dropout: float = 0.15):
    torch = require_torch()
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(channels),
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(x + self.net(x))

    class GeometryTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(geometry_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
            blocks = [ResidualBlock(hidden_dim, dilation=2**idx) for idx in range(num_layers)]
            self.tcn = nn.Sequential(*blocks)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            # x: [B, T, D]
            x = self.input(x)
            x = x.transpose(1, 2)
            x = self.tcn(x)
            return self.classifier(x)

    return GeometryTCN()
