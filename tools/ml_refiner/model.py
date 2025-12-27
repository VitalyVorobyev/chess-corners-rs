"""CNN for subpixel corner refinement with explicit spatial features."""

from __future__ import annotations

import torch
from torch import nn


class CornerRefinerNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        head_dim: int = 64,
        use_coordconv: bool = True,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.use_coordconv = bool(use_coordconv)
        in_ch = 1 + (2 if self.use_coordconv else 0)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coordconv:
            x = self._append_coords(x)
        x = self.backbone(x)
        return self.head(x)

    @staticmethod
    def _append_coords(x: torch.Tensor) -> torch.Tensor:
        """Append pixel-space (x, y) coordinate channels to preserve location."""
        batch, _, height, width = x.shape
        device = x.device
        dtype = x.dtype
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        x_coords = torch.linspace(-cx, cx, steps=width, device=device, dtype=dtype)
        y_coords = torch.linspace(-cy, cy, steps=height, device=device, dtype=dtype)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch, 1, height, width)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch, 1, height, width)
        return torch.cat((x, x_coords, y_coords), dim=1)
