import numpy as np
import pandas as pd
import torch
from torch import nn

from kaggle_traffic.models.transformer.attention import sinusoidal_encoding
from kaggle_traffic.models.transformer.spatial_attention import SpatialAttention
from kaggle_traffic.models.transformer.temporal_attention import TemporalAttention

TRAFFIC_COLUMNS = ("speed_kmh", "flow_vph")


class SpatiotemporalTransformer(nn.Module):
    """Spatial attention, then temporal attention, shared across the day."""

    def __init__(self, in_features: int, channels: int, heads: int):
        super().__init__()
        self.lift = nn.Linear(in_features, channels)
        self.spatial = SpatialAttention(channels, heads)
        self.temporal = TemporalAttention(channels, heads)
        self.readout = nn.Linear(channels, len(TRAFFIC_COLUMNS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.lift(features)
        time_count, station_count, channels = hidden.shape
        hidden = hidden + sinusoidal_encoding(time_count, channels, hidden.device)[:, None, :]
        hidden = hidden + sinusoidal_encoding(station_count, channels, hidden.device)[None, :, :]
        hidden = self.spatial(hidden)
        hidden = self.temporal(hidden)
        return self.readout(hidden)


def st_impute(
    df: pd.DataFrame,
    epochs: int = 30,
    channels: int = 32,
    heads: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 0,
    mask_rate: float = 0.2,
) -> pd.DataFrame:
    """Fill speed and flow with a spatiotemporal transformer.

    Each timestamp attends over stations, then each station attends over the
    day. Training hides some cells that still have readings. Cells that were
    already blank stay out of the loss, and observed values are written back.
    """
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    stations = _stations_by_milepost(frame)
    times = pd.Index(frame["timestamp"].unique()).sort_values()
    panels = {
        column: _panel(frame, stations, times, column)
        for column in ("speed_kmh", "flow_vph", "occupancy")
    }
    observed = np.isfinite(panels["speed_kmh"]) & np.isfinite(panels["flow_vph"])
    if not observed.any():
        raise ValueError("matrix has no observed entries")

    stats = {column: _mean_std(panels[column]) for column in panels}
    predictions = _train(
        panels,
        observed,
        stats,
        epochs=epochs,
        channels=channels,
        heads=heads,
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


def _train(panels, observed, stats, epochs, channels, heads, learning_rate, seed, mask_rate):
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    features = _features(panels, observed, stats)
    targets = np.stack(
        [(panels[column] - stats[column][0]) / stats[column][1] for column in TRAFFIC_COLUMNS],
        axis=-1,
    )
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    model = SpatiotemporalTransformer(features.shape[-1], channels, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        hidden_targets = _training_mask(observed, mask_rate, generator)
        if not hidden_targets.any():
            continue
        visible = observed & ~hidden_targets
        batch = torch.tensor(_features(panels, visible, stats), dtype=torch.float32)
        prediction = model(batch)
        loss = nn.functional.mse_loss(prediction[hidden_targets], target_tensor[hidden_targets])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float32)).numpy()


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
    return milepost.index.to_numpy()


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
