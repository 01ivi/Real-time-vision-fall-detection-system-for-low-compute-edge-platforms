"""Training-only domain generalization helpers."""

from __future__ import annotations

from edgefall.utils.deps import require_torch


def coral_loss(source, target):
    torch = require_torch()
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_tensor(0.0)
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cov_s = source.t().matmul(source) / (source.size(0) - 1)
    cov_t = target.t().matmul(target) / (target.size(0) - 1)
    return torch.mean((cov_s - cov_t) ** 2)


def mmd_rbf_loss(source, target, gamma: float = 1.0):
    torch = require_torch()

    def kernel(a, b):
        dist = torch.cdist(a, b) ** 2
        return torch.exp(-gamma * dist)

    return kernel(source, source).mean() + kernel(target, target).mean() - 2 * kernel(source, target).mean()


def gradient_reverse_layer(lambda_: float = 1.0):
    torch = require_torch()

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_ * grad_output

    return GradientReverse.apply
