"""CNN for subpixel corner refinement with explicit spatial features.

Two architectures are provided:

- ``CornerRefinerNet`` — the 5-layer CoordConv CNN with flatten+MLP head
  (~180K params at defaults). Ships as ``chess_refiner_v2.onnx`` /
  ``v3.onnx``. Shipped since v2.
- ``CornerRefinerNetLarge`` — 2–4× wider variant with optional
  GroupNorm, added to probe whether the small net has hit a capacity
  ceiling on the mixed tanh + hard-cells distribution. Topology
  matches the original so ONNX export is a drop-in swap.
"""

from __future__ import annotations

import torch
from torch import nn


class CornerRefinerNet(nn.Module):
    """Small CoordConv CNN. Topology unchanged since v1 so existing
    checkpoints (including ``chess_refiner_v2/v3.onnx``) load cleanly.
    """

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
            x = _append_coords(x)
        x = self.backbone(x)
        return self.head(x)


class CornerRefinerNetLarge(nn.Module):
    """Wider CoordConv CNN with GroupNorm for training stability.

    Defaults to ``base_channels=32`` and ``head_dim=128`` (~730K params,
    ~4× the small model). Topology and I/O contract match the small
    model — 1-channel patch in, 3 scalars out — so ``export_onnx.py``
    and the Rust inference path work as a drop-in swap.
    """

    def __init__(
        self,
        base_channels: int = 32,
        head_dim: int = 128,
        use_coordconv: bool = True,
        use_groupnorm: bool = True,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.use_coordconv = bool(use_coordconv)
        in_ch = 1 + (2 if self.use_coordconv else 0)

        def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1),
            ]
            if use_groupnorm:
                groups = min(8, cout)
                layers.append(nn.GroupNorm(groups, cout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.backbone = nn.Sequential(
            block(in_ch, c1),
            block(c1, c1),
            block(c1, c2, stride=2),
            block(c2, c2),
            block(c2, c3, stride=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coordconv:
            x = _append_coords(x)
        x = self.backbone(x)
        return self.head(x)


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


class CornerRefinerNetSoftArgmax(nn.Module):
    """Soft-argmax keypoint localiser for sub-pixel corners.

    Design, step by step:

    1. CoordConv on the raw patch — position-aware from the start.
    2. Backbone keeps the **full 21×21 spatial resolution**; no
       stride-2 downsamples. Capacity comes from width
       (``base_channels``) and depth, not from coarsening.
    3. A final 1×1 conv to a single-channel "corner-presence logit"
       heatmap, then pixel-shuffle upsample by ``heatmap_upsample``
       (default 4) to a 84×84 resolution for sub-pixel addressing.
    4. Softmax over the whole heatmap, then spatial expectation:
       ``x = Σ i · p(i, j)``, ``y = Σ j · p(i, j)``.
    5. Convert the expected pixel index back to an offset from the
       patch centre, so the output contract ``[dx, dy, conf_logit]``
       matches the legacy :class:`CornerRefinerNet` — drop-in replacement
       for the Rust inference path.

    The ``conf_logit`` head is a separate GAP-then-linear branch. It
    could also be derived from the peak heatmap value (``log max p``),
    but keeping it as a separate head preserves the training loss
    structure used by v2/v3.
    """

    def __init__(
        self,
        base_channels: int = 32,
        depth: int = 6,
        heatmap_upsample: int = 4,
        use_coordconv: bool = True,
        use_groupnorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_coordconv = bool(use_coordconv)
        self.patch_size = 21
        self.upsample = int(heatmap_upsample)
        self.heatmap_size = self.patch_size * self.upsample

        in_ch = 1 + (2 if self.use_coordconv else 0)

        def block(cin: int, cout: int, kernel: int = 3) -> nn.Sequential:
            pad = kernel // 2
            layers: list[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=kernel, padding=pad),
            ]
            if use_groupnorm:
                groups = min(8, cout)
                layers.append(nn.GroupNorm(groups, cout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        layers: list[nn.Module] = [block(in_ch, base_channels, kernel=5)]
        for _ in range(depth - 1):
            layers.append(block(base_channels, base_channels))
        self.backbone = nn.Sequential(*layers)

        # 1×1 projection to a single-channel heatmap, then pixel-shuffle
        # upsample by `upsample`² channels so output has 1 channel at
        # (P·U)×(P·U) resolution.
        shuf_c = self.upsample * self.upsample
        self.heatmap_proj = nn.Conv2d(base_channels, shuf_c, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(self.upsample)

        # Separate confidence branch: GAP + linear → 1 logit.
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 1),
        )

        # Precomputed index grids for the expectation, in patch-pixel
        # units. register_buffer makes them move with the model but not
        # trained. Indices are scaled so that the centre of the patch
        # corresponds to 0.
        self._build_index_grid()

    def _build_index_grid(self) -> None:
        hs = self.heatmap_size
        centre = (self.patch_size - 1) / 2.0
        # Each heatmap pixel spans 1/upsample input-patch pixels; its
        # centre sits at `(i + 0.5) / upsample - 0.5` in patch-pixel
        # coords. Subtract `centre` so the grid is zero at the patch
        # centre.
        coords = (torch.arange(hs, dtype=torch.float32) + 0.5) / self.upsample - 0.5
        coords = coords - centre
        # Shape: (1, 1, H, W) for broadcasting over batch.
        x_grid = coords.view(1, 1, 1, hs).clone()
        y_grid = coords.view(1, 1, hs, 1).clone()
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("y_grid", y_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_coordconv:
            x = _append_coords(x)
        feat = self.backbone(x)                          # (B, C, 21, 21)
        logits = self.heatmap_proj(feat)                 # (B, U², 21, 21)
        logits = self.pixel_shuffle(logits)              # (B, 1, 84, 84)
        # Softmax over spatial dims.
        b, _, h, w = logits.shape
        flat = logits.view(b, 1, h * w)
        probs = torch.softmax(flat, dim=-1).view(b, 1, h, w)
        # Spatial expectation → predicted (dx, dy) in patch-pixel units
        # offset from centre.
        dx = (probs * self.x_grid).sum(dim=(2, 3)).squeeze(1)
        dy = (probs * self.y_grid).sum(dim=(2, 3)).squeeze(1)
        conf_logit = self.conf_head(feat).squeeze(1)
        return torch.stack([dx, dy, conf_logit], dim=1)


def build_model(name: str) -> nn.Module:
    """Factory for the public architectures. Used by train/eval/export."""
    name = name.lower()
    if name in ("small", "corner_refiner_net", ""):
        return CornerRefinerNet()
    if name in ("large", "corner_refiner_net_large"):
        return CornerRefinerNetLarge()
    if name in ("soft_argmax", "soft-argmax", "sa"):
        return CornerRefinerNetSoftArgmax()
    raise ValueError(f"unknown model name {name!r}")
