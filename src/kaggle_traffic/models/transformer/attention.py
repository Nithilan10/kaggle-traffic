import math

import torch
from torch import nn


def sinusoidal_encoding(length: int, channels: int, device: torch.device) -> torch.Tensor:
    """Fixed position codes so attention can tell nearby stations and times apart."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    index = torch.arange(0, channels, 2, device=device, dtype=torch.float32)
    divisor = torch.exp(index * (-math.log(10000.0) / channels))
    encoding = torch.zeros(length, channels, device=device)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)
