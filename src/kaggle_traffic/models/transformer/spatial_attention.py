from torch import nn

from kaggle_traffic.models.transformer.attention import AttentionBlock


class SpatialAttention(nn.Module):
    """Attend across stations. Tokens are shaped (time, station, channels)."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.block = AttentionBlock(channels, heads)

    def forward(self, hidden):
        return self.block(hidden)
