from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from kaggle_traffic.models.graph.gnn import (
    _features,
    _mean_std,
    _normalized_adjacency,
    _panel,
    _stations_by_milepost,
    _training_mask,
)

TRAFFIC_COLUMNS = ("speed_kmh", "flow_vph")
INPUT_COLUMNS = ("speed_kmh", "flow_vph", "occupancy")


class TrafficSTGNN(nn.Module):
    """Temporal convolution, then a graph convolution along the freeway, then time again.

    This is the STGCN block used for detector data: each station looks at nearby
    times, messages pass to neighboring stations, then time is mixed once more.
    """

    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.temporal_in = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.spatial = nn.Linear(hidden, hidden)
        self.temporal_out = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.readout = nn.Linear(hidden, len(TRAFFIC_COLUMNS))

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = _mix_time(self.temporal_in, features)
        hidden = torch.tanh(torch.einsum("ij,tjf->tif", adjacency, self.spatial(hidden)))
        hidden = _mix_time(self.temporal_out, hidden)
        return self.readout(hidden)


def pretrain_stgnn(
    frames,
    epochs: int = 2,
    hidden: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    mask_rate: float = 0.2,
) -> dict:
    """Learn weights from complete corridor days by hiding readings and restoring them.

    Each frame is one day on the same kind of freeway: stations ordered by milepost,
    five-minute speed, flow, and occupancy. The returned weights are the pretrained
    state, with no adjacency matrix stored, so they can be loaded on any day of this corridor.
    """
    prepared = [item for item in (_prepare(frame) for frame in frames) if item is not None]
    if not prepared:
        raise ValueError("matrix has no observed entries")

    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    in_features = _features(prepared[0][0], prepared[0][1], prepared[0][2]).shape[-1]
    model = TrafficSTGNN(in_features, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for panels, observed, stats, adjacency in prepared:
            _fit_step(model, optimizer, panels, observed, stats, adjacency, generator, mask_rate)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def stgnn_impute(
    df: pd.DataFrame,
    pretrained: Optional[dict] = None,
    finetune_epochs: int = 40,
    hidden: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    mask_rate: float = 0.2,
) -> pd.DataFrame:
    """Fill speed and flow from a spatiotemporal GNN, starting from pretrained weights.

    Fine-tuning uses only cells that still have readings. Held-out blanks stay out
    of the loss, and the original readings are written back after prediction.
    """
    prepared = _prepare(df)
    if prepared is None:
        raise ValueError("matrix has no observed entries")
    panels, observed, stats, adjacency = prepared
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    stations, _ = _stations_by_milepost(frame)
    times = pd.Index(frame["timestamp"].unique()).sort_values()

    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    in_features = _features(panels, observed, stats).shape[-1]
    model = TrafficSTGNN(in_features, hidden)
    if pretrained is not None:
        model.load_state_dict(pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(finetune_epochs):
        _fit_step(model, optimizer, panels, observed, stats, adjacency, generator, mask_rate)

    model.eval()
    with torch.no_grad():
        features = torch.tensor(_features(panels, observed, stats), dtype=torch.float32)
        adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
        predictions = model(features, adjacency_tensor).numpy()

    result = frame.copy()
    for channel, column in enumerate(TRAFFIC_COLUMNS):
        mean, scale = stats[column]
        filled = predictions[:, :, channel] * scale + mean
        filled = np.where(np.isfinite(panels[column]), panels[column], filled)
        lookup = pd.DataFrame(filled, index=times, columns=stations).stack()
        lookup.index = lookup.index.set_names(["timestamp", "station_id"])
        keys = pd.MultiIndex.from_arrays([result["timestamp"], result["station_id"].to_numpy()])
        result[column] = lookup.reindex(keys).to_numpy()
    return result


def _fit_step(model, optimizer, panels, observed, stats, adjacency, generator, mask_rate):
    hidden_targets = _training_mask(observed, mask_rate, generator)
    if not hidden_targets.any():
        return
    visible = observed & ~hidden_targets
    batch = torch.tensor(_features(panels, visible, stats), dtype=torch.float32)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    targets = np.stack(
        [(panels[column] - stats[column][0]) / stats[column][1] for column in TRAFFIC_COLUMNS],
        axis=-1,
    )
    prediction = model(batch, adjacency_tensor)
    loss = nn.functional.mse_loss(
        prediction[hidden_targets],
        torch.tensor(targets, dtype=torch.float32)[hidden_targets],
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _prepare(frame: pd.DataFrame):
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    stations, positions = _stations_by_milepost(working)
    times = pd.Index(working["timestamp"].unique()).sort_values()
    panels = {column: _panel(working, stations, times, column) for column in INPUT_COLUMNS}
    observed = np.isfinite(panels["speed_kmh"]) & np.isfinite(panels["flow_vph"])
    if not observed.any():
        return None
    stats = {column: _mean_std(panels[column]) for column in panels}
    adjacency = _normalized_adjacency(positions)
    return panels, observed, stats, adjacency


def _mix_time(convolution: nn.Conv1d, hidden: torch.Tensor) -> torch.Tensor:
    mixed = convolution(hidden.permute(1, 2, 0))
    return torch.tanh(mixed.permute(2, 0, 1))
