import pandas as pd

def temporal_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["station_id","timestamp"])

    traffic_columns = ["speed_kmh", "flow_vph", "occupancy"]
    df[traffic_columns] = df.groupby("station_id")[traffic_columns].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
    return df


