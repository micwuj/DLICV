"""Grad-CAM for CNN-style models (resnet18, convnext_tiny).

Forward hook captures activations of the target layer; backward hook captures
gradients. CAM = ReLU(sum_c(mean_hw(grad_c) * activation_c)).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet18":
        return model.layer4
    if model_name == "convnext_tiny":
        return model.stages[-1].blocks[-1].conv_dw
    raise ValueError(f"Grad-CAM target layer not configured for {model_name!r}")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[torch.Tensor, int]:
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        pred = int(out.argmax(dim=1).item())
        idx = pred if class_idx is None else int(class_idx)
        score = out[0, idx]
        score.backward()
        # gradients: (1, C, h, w); activations: (1, C, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        if was_training:
            self.model.train()
        return cam, pred
