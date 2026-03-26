import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        precompute_resolution: Optional[int] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        self._precompute_sizes = None
        if precompute_resolution is not None:
            self._precompute_sizes = [
                (precompute_resolution // 4, precompute_resolution // 4),
                (precompute_resolution // 8, precompute_resolution // 8),
                (precompute_resolution // 16, precompute_resolution // 16),
                (precompute_resolution // 32, precompute_resolution // 32),
            ]
            for size in self._precompute_sizes:
                tensors = torch.zeros((1, 1) + size)
                self(tensors)

    def _encode_xy(self, x, y):
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (torch.sin(pos_x[:, 0::2]), torch.cos(pos_x[:, 1::2])),
            dim=2,
        ).flatten(1)
        pos_y = torch.stack(
            (torch.sin(pos_y[:, 0::2]), torch.cos(pos_y[:, 1::2])),
            dim=2,
        ).flatten(1)
        return pos_x, pos_y

    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos.detach()

    encode = encode_boxes

    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos.detach()

    def forward(self, x: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """
        Args:
            x: Either a torch.Tensor (NCHW format) or a shape tuple (N, C, H, W)
        Returns:
            Position encoding in NCHW format
        """
        shape = x if isinstance(x, tuple) else x.shape
        batch, _, height, width = shape

        cache_key = (height, width)
        cur_device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached.device != cur_device:
                cached = cached.to(cur_device)
                self.cache[cache_key] = cached
            return cached[None].repeat(batch, 1, 1, 1)

        device = cur_device

        y_embed = (
            torch.arange(1, height + 1, dtype=torch.float32, device=device)
            .reshape(1, -1, 1)
        )
        y_embed = y_embed.expand(batch, height, width)
        x_embed = (
            torch.arange(1, width + 1, dtype=torch.float32, device=device)
            .reshape(1, 1, -1)
        )
        x_embed = x_embed.expand(batch, height, width)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (torch.sin(pos_x[:, :, :, 0::2]), torch.cos(pos_x[:, :, :, 1::2])),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (torch.sin(pos_y[:, :, :, 0::2]), torch.cos(pos_y[:, :, :, 1::2])),
            dim=4,
        ).flatten(3)
        # concat along last dim then permute to NCHW
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        if cache_key is not None:
            self.cache[cache_key] = pos[0]
        return pos.detach()
