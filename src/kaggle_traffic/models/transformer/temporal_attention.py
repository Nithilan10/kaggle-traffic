from torch import nn

from kaggle_traffic.models.transformer.attention import AttentionBlock


class TemporalAttention(nn.Module):
    """Attend across time. Tokens are shaped (time, station, channels)."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.block = AttentionBlock(channels, heads)

    def forward(self, hidden):
        ordered = hidden.transpose(0, 1)
        return self.block(ordered).transpose(0, 1)
