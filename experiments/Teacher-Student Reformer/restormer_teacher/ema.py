from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of model weights (shadow copy)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: dict[str, Any] = {}
        self.backup: dict[str, Any] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            sh = self.shadow[name]
            if sh.device != param.device or sh.dtype != param.dtype:
                sh = sh.to(device=param.device, dtype=param.dtype)
                self.shadow[name] = sh
            new_average = (1.0 - self.decay) * param.data + self.decay * sh
            self.shadow[name] = new_average.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        """Copy shadow weights into model (destructive). Use backup/restore around eval."""
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.detach().clone()
            param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any], map_location: torch.device | None = None) -> None:
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        if map_location is not None:
            for k in self.shadow:
                self.shadow[k] = self.shadow[k].to(map_location)
