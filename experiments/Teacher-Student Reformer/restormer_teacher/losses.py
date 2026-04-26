from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on finite differences (all RGB channels)."""
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def fft_mag_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on magnitude of real 2D FFT (per-channel). Uses CPU FFT on MPS to avoid backend warnings."""
    dev = pred.device
    pred = pred.float().contiguous()
    target = target.float().contiguous()
    if dev.type == "mps":
        pred = pred.cpu()
        target = target.cpu()
    fp = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
    ft = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
    loss = (fp.abs() - ft.abs()).abs().mean()
    return loss.to(dev)


class TeacherCompositeLoss(nn.Module):
    """0.5 Charbonnier + 0.25 L1 + 0.15 Edge + 0.1 FFT."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        c = charbonnier(pred, hr)
        l1 = F.l1_loss(pred, hr)
        e = edge_loss(pred, hr)
        f = fft_mag_loss(pred, hr)
        total = 0.50 * c + 0.25 * l1 + 0.15 * e + 0.10 * f
        parts = {
            "charbonnier": float(c.detach()),
            "l1": float(l1.detach()),
            "edge": float(e.detach()),
            "fft": float(f.detach()),
        }
        return total, parts


def residual_supervision_l1(pred: torch.Tensor, lr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """Reporting: L1((pred - lr), (hr - lr))."""
    return F.l1_loss(pred - lr, hr - lr)
