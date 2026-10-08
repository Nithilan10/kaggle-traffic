
from pathlib import Path
import os
import pandas as pd
from kaggle_traffic.data.inspect import inspect
from kaggle_traffic.data.unzip import unzip_file
from kaggle_traffic.evaluation.synthetic_mask import mask
from kaggle_traffic.evaluation.metrics import rmse
from kaggle_traffic.models.linear_interpolation.temporal import temporal_interpolation
from kaggle_traffic.models.linear_interpolation.spatial import spatial_interpolation
from kaggle_traffic.models.linear_interpolation.spatiotemporal import spatiotemporal_interpolation
from kaggle_traffic.models.low_rank.svd.matrix import build_matrix
from kaggle_traffic.models.low_rank.svd.svd import svd_impute
from kaggle_traffic.models.low_rank.dmd import dmd_impute
from kaggle_traffic.models.low_rank.matrix_completion import matrix_complete

if __name__ == "__main__":

    def interpolation():
    #unzip files
        for file in os.listdir("/Users/selvakumarrangasamy/Documents/GitHub/kaggle-traffic/data/raw/kaggle_public/corridors/D7_I10_E/private/mainline_states_masked/mask_regime=R2"):
            if file.endswith(".parquet"):
                df = inspect(Path(f"/Users/selvakumarrangasamy/Documents/GitHub/kaggle-traffic/data/raw/kaggle_public/corridors/D7_I10_E/private/mainline_states_masked/mask_regime=R2/{file}"))
                missing_mask = df["speed_kmh"].isna()

                print("\nMISSING BEFORE INTERPOLATION:")
                print(df.loc[missing_mask, ["timestamp", "station_id", "speed_kmh", "flow_vph"]])
                masked_df, true_values, masked_indices = mask(df)

                temp_df = temporal_interpolation(masked_df.copy())
                print("\nAFTER TEMPORAL INTERPOLATION:")
                print(temp_df.loc[missing_mask, ["timestamp", "station_id", "speed_kmh", "flow_vph"]])
                print(temp_df.isna().sum())

                comparison = true_values.copy()

                comparison["predicted_speed"] = temp_df.loc[
                    masked_indices, "speed_kmh"
                ]

                comparison["predicted_flow"] = temp_df.loc[
                    masked_indices, "flow_vph"
                ]

                print(comparison.head(20))

                tempflow_rmse = rmse(comparison["flow_vph"], comparison["predicted_flow"])
                tempspeed_rmse = rmse(comparison["speed_kmh"], comparison["predicted_speed"])
                print(f"Flow RMSE: {tempflow_rmse}")
                print(f"Speed RMSE: {tempspeed_rmse}")

                spatial_df = spatial_interpolation(masked_df.copy())
                print("\nAFTER SPATIAL INTERPOLATION:")
                print(spatial_df.loc[missing_mask, ["timestamp", "station_id", "speed_kmh", "flow_vph"]])
                print(spatial_df.isna().sum())

                comparison = true_values.copy()

                comparison["predicted_speed"] = spatial_df.loc[
                    masked_indices, "speed_kmh"
                ]

                comparison["predicted_flow"] = spatial_df.loc[
                    masked_indices, "flow_vph"
                ]

                print(comparison.head(20))

                spatflow_rmse = rmse(comparison["flow_vph"], comparison["predicted_flow"])
                spatpeed_rmse = rmse(comparison["speed_kmh"], comparison["predicted_speed"])
                print(f"Flow RMSE: {spatflow_rmse}")
                print(f"Speed RMSE: {spatpeed_rmse}")

                best_flow_rmse = float('inf')
                best_speed_rmse = float('inf')
                best_flow_weight = 0
                best_speed_weight = 0
                for i in range(101):
                    temporal_weight = i / 100
                    spatiotemporal_df = spatiotemporal_interpolation(temporal_interpolation, spatial_interpolation, masked_df.copy(), temporal_weight)

                    comparison = true_values.copy()

                    comparison["predicted_speed"] = spatiotemporal_df.loc[
                        masked_indices, "speed_kmh"
                    ]

                    comparison["predicted_flow"] = spatiotemporal_df.loc[
                        masked_indices, "flow_vph"
                    ]

                    stflow_rmse = rmse(comparison["flow_vph"], comparison["predicted_flow"])
                    stspeed_rmse = rmse(comparison["speed_kmh"], comparison["predicted_speed"])
                    if stflow_rmse < best_flow_rmse:
                        best_flow_rmse = stflow_rmse
                        best_flow_weight = temporal_weight
                    if stspeed_rmse < best_speed_rmse:
                        best_speed_rmse = stspeed_rmse
                        best_speed_weight = temporal_weight
                print(f"Best RMSE - Flow: {best_flow_rmse}, Speed: {best_speed_rmse}")
                print(f"Best Weight - Flow: {best_flow_weight}, Speed: {best_speed_weight}")

    def predictions_for(frame, indices, completed):
        stations = frame.loc[indices, "station_id"].to_numpy()
        times = pd.to_datetime(frame.loc[indices, "timestamp"], utc=True)
        row_pos = pd.Index(completed.index).get_indexer(stations)
        col_pos = pd.DatetimeIndex(completed.columns).get_indexer(times)
        return completed.to_numpy()[row_pos, col_pos]

    def low_rank():
        ranks = (2, 5, 10, 15, 20)
        data_dir = Path("/Users/selvakumarrangasamy/Documents/GitHub/kaggle-traffic/data/raw/kaggle_public/corridors/D7_I10_E/private/mainline_states_masked/mask_regime=R2")
        for file in sorted(os.listdir(data_dir)):
            if not file.endswith(".parquet"):
                continue
            df = inspect(data_dir / file)
            masked_df, true_values, masked_indices = mask(df)
            speed_matrix = build_matrix(masked_df, "speed_kmh")
            flow_matrix = build_matrix(masked_df, "flow_vph")

            print("\nLAG SHAPES:")
            print(speed_matrix.iloc[:, :-1].shape, speed_matrix.iloc[:, 1:].shape)
            print(flow_matrix.iloc[:, :-1].shape, flow_matrix.iloc[:, 1:].shape)

            best_svd_speed = (float("inf"), None)
            best_svd_flow = (float("inf"), None)
            best_dmd_speed = (float("inf"), None)
            best_dmd_flow = (float("inf"), None)

            for rank in ranks:
                svd_speed = svd_impute(speed_matrix, rank=rank)
                svd_flow = svd_impute(flow_matrix, rank=rank)
                dmd_speed = dmd_impute(speed_matrix, rank=rank, initial=svd_speed)
                dmd_flow = dmd_impute(flow_matrix, rank=rank, initial=svd_flow)

                svd_speed_rmse = rmse(true_values["speed_kmh"].to_numpy(), predictions_for(masked_df, masked_indices, svd_speed))
                svd_flow_rmse = rmse(true_values["flow_vph"].to_numpy(), predictions_for(masked_df, masked_indices, svd_flow))
                dmd_speed_rmse = rmse(true_values["speed_kmh"].to_numpy(), predictions_for(masked_df, masked_indices, dmd_speed))
                dmd_flow_rmse = rmse(true_values["flow_vph"].to_numpy(), predictions_for(masked_df, masked_indices, dmd_flow))

                print(f"\nRANK {rank}")
                print(f"SVD  speed RMSE: {svd_speed_rmse:.4f}  flow RMSE: {svd_flow_rmse:.4f}")
                print(f"DMD  speed RMSE: {dmd_speed_rmse:.4f}  flow RMSE: {dmd_flow_rmse:.4f}")

                if svd_speed_rmse < best_svd_speed[0]:
                    best_svd_speed = (svd_speed_rmse, rank)
                if svd_flow_rmse < best_svd_flow[0]:
                    best_svd_flow = (svd_flow_rmse, rank)
                if dmd_speed_rmse < best_dmd_speed[0]:
                    best_dmd_speed = (dmd_speed_rmse, rank)
                if dmd_flow_rmse < best_dmd_flow[0]:
                    best_dmd_flow = (dmd_flow_rmse, rank)

            print(f"\nBest SVD speed: rank {best_svd_speed[1]} RMSE {best_svd_speed[0]:.4f}")
            print(f"Best SVD flow:  rank {best_svd_flow[1]} RMSE {best_svd_flow[0]:.4f}")
            print(f"Best DMD speed: rank {best_dmd_speed[1]} RMSE {best_dmd_speed[0]:.4f}")
            print(f"Best DMD flow:  rank {best_dmd_flow[1]} RMSE {best_dmd_flow[0]:.4f}")

            shrinkages = (0.01, 0.05, 0.1, 0.2, 0.4)
            best_mc_speed = (float("inf"), None)
            best_mc_flow = (float("inf"), None)
            for shrinkage in shrinkages:
                mc_speed = matrix_complete(speed_matrix, shrinkage=shrinkage)
                mc_flow = matrix_complete(flow_matrix, shrinkage=shrinkage)
                mc_speed_rmse = rmse(true_values["speed_kmh"].to_numpy(), predictions_for(masked_df, masked_indices, mc_speed))
                mc_flow_rmse = rmse(true_values["flow_vph"].to_numpy(), predictions_for(masked_df, masked_indices, mc_flow))
                print(
                    f"\nSHRINKAGE {shrinkage:.2f}  "
                    f"speed rank {mc_speed.attrs['rank']}  flow rank {mc_flow.attrs['rank']}"
                )
                print(f"MC   speed RMSE: {mc_speed_rmse:.4f}  flow RMSE: {mc_flow_rmse:.4f}")
                if mc_speed_rmse < best_mc_speed[0]:
                    best_mc_speed = (mc_speed_rmse, shrinkage)
                if mc_flow_rmse < best_mc_flow[0]:
                    best_mc_flow = (mc_flow_rmse, shrinkage)

            print(f"\nBest MC speed: shrinkage {best_mc_speed[1]} RMSE {best_mc_speed[0]:.4f}")
            print(f"Best MC flow:  shrinkage {best_mc_flow[1]} RMSE {best_mc_flow[0]:.4f}")

    low_rank()
