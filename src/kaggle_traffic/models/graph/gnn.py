import numpy as np
import pandas as pd
import torch
from torch import nn

TRAFFIC_COLUMNS = ("speed_kmh", "flow_vph")


class SpatialGCN(nn.Module):
    """Two graph-convolution layers on one freeway path, shared across time."""

    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.lift = nn.Linear(in_features, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, len(TRAFFIC_COLUMNS))

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self._mix(adjacency, self.lift(features)))
        hidden = torch.tanh(self._mix(adjacency, self.message(hidden)))
        return self.readout(hidden)

    @staticmethod
    def _mix(adjacency: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,tjf->tif", adjacency, hidden)


def gnn_impute(
    df: pd.DataFrame,
    epochs: int = 40,
    hidden: int = 16,
    learning_rate: float = 1e-2,
    seed: int = 0,
    mask_rate: float = 0.2,
) -> pd.DataFrame:
    """Fill speed and flow with a graph convolutional network.

    Stations ordered by milepost are the graph. Training hides a fraction of
    the cells that still have readings and asks the network to recover them
    from the remaining neighbors. Cells that were already blank stay out of
    the loss. Observed values are written back after prediction.
    """
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    stations, positions = _stations_by_milepost(frame)
    times = pd.Index(frame["timestamp"].unique()).sort_values()
    panels = {
        column: _panel(frame, stations, times, column)
        for column in ("speed_kmh", "flow_vph", "occupancy")
    }
    observed = np.isfinite(panels["speed_kmh"]) & np.isfinite(panels["flow_vph"])
    if not observed.any():
        raise ValueError("matrix has no observed entries")

    stats = {column: _mean_std(panels[column]) for column in panels}
    adjacency = _normalized_adjacency(positions)
    predictions = _train(
        panels,
        observed,
        stats,
        adjacency,
        epochs=epochs,
        hidden=hidden,
        learning_rate=learning_rate,
        seed=seed,
        mask_rate=mask_rate,
    )

    result = frame.copy()
    for channel, column in enumerate(TRAFFIC_COLUMNS):
        mean, std = stats[column]
        filled = predictions[:, :, channel] * std + mean
        filled = np.where(np.isfinite(panels[column]), panels[column], filled)
        lookup = pd.DataFrame(filled, index=times, columns=stations).stack()
        lookup.index = lookup.index.set_names(["timestamp", "station_id"])
        keys = pd.MultiIndex.from_arrays([result["timestamp"], result["station_id"].to_numpy()])
        result[column] = lookup.reindex(keys).to_numpy()
    return result


def _train(panels, observed, stats, adjacency, epochs, hidden, learning_rate, seed, mask_rate):
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    features = _features(panels, observed, stats)
    targets = np.stack(
        [(panels[column] - stats[column][0]) / stats[column][1] for column in TRAFFIC_COLUMNS],
        axis=-1,
    )
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    model = SpatialGCN(features.shape[-1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        hidden_targets = _training_mask(observed, mask_rate, generator)
        if not hidden_targets.any():
            continue
        visible = observed & ~hidden_targets
        batch = torch.tensor(_features(panels, visible, stats), dtype=torch.float32)
        prediction = model(batch, adjacency_tensor)
        loss = nn.functional.mse_loss(prediction[hidden_targets], target_tensor[hidden_targets])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        full = torch.tensor(features, dtype=torch.float32)
        return model(full, adjacency_tensor).numpy()


def _features(panels, visible, stats) -> np.ndarray:
    channels = []
    for column in ("speed_kmh", "flow_vph", "occupancy"):
        mean, std = stats[column]
        values = (panels[column] - mean) / std
        seen = visible if column in TRAFFIC_COLUMNS else np.isfinite(panels[column])
        channels.append(np.where(seen, values, 0.0))
        channels.append(seen.astype(np.float64))
    return np.stack(channels, axis=-1).astype(np.float32)


def _training_mask(observed: np.ndarray, mask_rate: float, generator: np.random.Generator) -> np.ndarray:
    hidden = np.zeros(observed.shape, dtype=bool)
    for time_index in range(observed.shape[0]):
        candidates = np.flatnonzero(observed[time_index])
        if candidates.size < 2:
            continue
        count = max(1, int(round(candidates.size * mask_rate)))
        count = min(count, candidates.size - 1)
        chosen = generator.choice(candidates, size=count, replace=False)
        hidden[time_index, chosen] = True
    return hidden


def _stations_by_milepost(frame: pd.DataFrame):
    milepost = frame.groupby("station_id", sort=False)["milepost"].mean().sort_values()
    return milepost.index.to_numpy(), milepost.to_numpy(dtype=float)


def _panel(frame, stations, times, column) -> np.ndarray:
    pivot = frame.pivot(index="timestamp", columns="station_id", values=column)
    return pivot.reindex(index=times, columns=stations).to_numpy(dtype=np.float64)


def _mean_std(values: np.ndarray):
    observed = values[np.isfinite(values)]
    if observed.size == 0:
        return 0.0, 1.0
    std = float(observed.std())
    if std < 1e-6:
        std = 1.0
    return float(observed.mean()), std


def _normalized_adjacency(positions: np.ndarray) -> np.ndarray:
    count = len(positions)
    adjacency = np.eye(count, dtype=np.float64)
    if count == 1:
        return adjacency
    distance = np.diff(positions.astype(np.float64))
    weight = 1.0 / np.maximum(distance, 1e-6)
    hops = np.arange(count - 1)
    adjacency[hops, hops + 1] = weight
    adjacency[hops + 1, hops] = weight
    degree = adjacency.sum(axis=1)
    scale = 1.0 / np.sqrt(degree)
    return scale[:, None] * adjacency * scale[None, :]
