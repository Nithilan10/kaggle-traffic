from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_traffic.evaluation.metrics import rmse
from kaggle_traffic.models.ensemble import blend_frames, component_fills, ensemble_impute

SPEED_SCALE = 25.0
FLOW_SCALE = 600.0
SPEED_WEIGHT = 0.54
FLOW_WEIGHT = 0.46


def task1_cost(speed_true, speed_pred, flow_true, flow_pred, lanes) -> float:
    """Lower is better. Matches the Task 1 weights and per-lane flow scaling."""
    speed = rmse(speed_true, speed_pred) / SPEED_SCALE
    flow = rmse(flow_true / lanes, flow_pred / lanes) / FLOW_SCALE
    return SPEED_WEIGHT * speed + FLOW_WEIGHT * flow


def fit_weights(root: Path, panel: str = "D7_I10_E") -> tuple:
    """Pick blend weights on one training day, scored on the real masked cells."""
    truth_path, masked_path = _train_pair(root, panel)
    masked = pd.read_parquet(masked_path)
    truth = pd.read_parquet(truth_path, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"])
    lanes = pd.read_csv(root / "corridors" / panel / "network" / "fd_parameters.csv", usecols=["link_id", "lanes"])
    parts = component_fills(masked)
    candidates = {(index / 10, 1 - index / 10): f"svd {index / 10:.1f}" for index in range(11)}
    best_weights = (0.5, 0.5)
    best_cost = float("inf")
    for weights, name in candidates.items():
        filled = blend_frames(parts, weights)
        for column in ("speed_kmh", "flow_vph"):
            observed = masked[column].notna().to_numpy()
            filled[column] = np.where(observed, masked[column].to_numpy(dtype=float), filled[column].to_numpy(dtype=float))
        cost = _cost_against_truth(filled, masked, truth, lanes)
        print(f"{name:18} cost {cost:.4f} weights {tuple(round(w, 2) for w in weights)}")
        if cost < best_cost:
            best_cost = cost
            best_weights = weights
    print(f"chosen weights {tuple(round(w, 2) for w in best_weights)} cost {best_cost:.4f}")
    return best_weights


def build_task1_submission(root: Path, out_path: Path, weights) -> None:
    key = pd.read_csv(
        root / "submission_key.csv",
        usecols=["submission_id", "task", "split", "panel", "timestamp", "station_id", "mask_regime"],
        dtype={
            "submission_id": "int64",
            "task": "string",
            "split": "string",
            "panel": "string",
            "timestamp": "string",
            "station_id": "string",
            "mask_regime": "string",
        },
    )
    state = key.loc[key["task"] == "state"].copy()
    state["timestamp"] = pd.to_datetime(state["timestamp"], utc=True)
    state["date"] = state["timestamp"].dt.strftime("%Y-%m-%d")
    submission = pd.read_csv(root / "sample_submission.csv").set_index("submission_id")
    submission["speed_kmh"] = submission["speed_kmh"].astype(float)
    submission["flow_vph"] = submission["flow_vph"].astype(float)
    groups = state.groupby(["panel", "split", "date", "mask_regime"], sort=False)
    print(f"state files {groups.ngroups}", flush=True)
    for index, ((panel, split, date, regime), group) in enumerate(groups, start=1):
        path = _masked_path(root, panel, split, regime, date)
        frame = pd.read_parquet(path, columns=["timestamp", "station_id", "milepost", "speed_kmh", "flow_vph", "occupancy"])
        filled = ensemble_impute(frame, weights=weights)
        filled["timestamp"] = pd.to_datetime(filled["timestamp"], utc=True)
        lookup = filled.set_index(["timestamp", "station_id"])[list(("speed_kmh", "flow_vph"))]
        keys = pd.MultiIndex.from_arrays([group["timestamp"].to_numpy(), group["station_id"].to_numpy()])
        predicted = lookup.reindex(keys)
        speed = _fill_missing(predicted["speed_kmh"].to_numpy(dtype=float))
        flow = _fill_missing(predicted["flow_vph"].to_numpy(dtype=float))
        submission.loc[group["submission_id"].to_numpy(), "speed_kmh"] = speed
        submission.loc[group["submission_id"].to_numpy(), "flow_vph"] = flow
        if index % 25 == 0 or index == groups.ngroups:
            print(f"{index}/{groups.ngroups} {panel} {split} {date}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.reset_index().to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def _cost_against_truth(filled, masked, truth, lanes) -> float:
    merged = masked[["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]].copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged["speed_hat"] = filled["speed_kmh"].to_numpy()
    merged["flow_hat"] = filled["flow_vph"].to_numpy()
    answers = truth.copy()
    answers["timestamp"] = pd.to_datetime(answers["timestamp"], utc=True)
    merged = merged.merge(
        answers[["timestamp", "station_id", "speed_kmh", "flow_vph"]],
        on=["timestamp", "station_id"],
        suffixes=("", "_true"),
    )
    holes = merged["speed_kmh"].isna() & merged["speed_kmh_true"].notna() & (merged["is_score_eligible"] == 1)
    scored = merged.loc[holes].merge(lanes, on="link_id", how="left")
    lane_count = scored["lanes"].fillna(1.0).to_numpy(dtype=float)
    return task1_cost(
        scored["speed_kmh_true"].to_numpy(dtype=float),
        scored["speed_hat"].to_numpy(dtype=float),
        scored["flow_vph_true"].to_numpy(dtype=float),
        scored["flow_hat"].to_numpy(dtype=float),
        lane_count,
    )


def _train_pair(root: Path, panel: str):
    for truth_path in sorted((root / "corridors" / panel / "train" / "mainline_states").rglob("*.parquet")):
        date = truth_path.stem.removeprefix("synthetic_mainline_").replace("_", "-")
        for regime in ("R1", "R2", "R3"):
            masked_path = _masked_path(root, panel, "train", regime, date)
            if masked_path.exists():
                return truth_path, masked_path
    raise FileNotFoundError(f"no train pair for {panel}")


def _masked_path(root: Path, panel: str, split: str, regime: str, date: str) -> Path:
    name = f"synthetic_mainline_{date.replace('-', '_')}.parquet"
    return root / "corridors" / panel / split / "mainline_states_masked" / f"mask_regime={regime}" / name


def _fill_missing(values: np.ndarray) -> np.ndarray:
    if np.isfinite(values).all():
        return values
    fallback = np.nanmean(values) if np.isfinite(values).any() else 0.0
    return np.where(np.isfinite(values), values, fallback)


if __name__ == "__main__":
    root = Path("data/raw/kaggle_public")
    weights = fit_weights(root)
    build_task1_submission(root, Path("data/processed/task1_submission.csv"), weights)
