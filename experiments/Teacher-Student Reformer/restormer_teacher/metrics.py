from __future__ import annotations

import torch


def tensor_psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-batch PSNR (dB), same as tools/lab3_pipeline_lib.tensor_psnr."""
    mse = torch.mean((a.float().clamp(0.0, 1.0) - b.float().clamp(0.0, 1.0)) ** 2, dim=(-3, -2, -1))
    mse = mse.clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def mean_psnr_over_loader(
    pred: torch.Tensor, hr: torch.Tensor, device: torch.device
) -> float:
    """pred, hr [B,3,H,W] on same device."""
    return float(tensor_psnr(pred.to(device), hr.to(device)).mean().item())


def residual_l1_ratio(pred: torch.Tensor, lr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8) -> float:
    """mean |pred-lr| / (mean|hr-lr| + eps) over batch and spatial."""
    num = (pred - lr).abs().mean().item()
    den = (hr - lr).abs().mean().item() + eps
    return num / den
