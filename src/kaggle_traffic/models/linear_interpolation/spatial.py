import pandas as pd

def spatial_interpolation(df):
    df = df.copy()  
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["milepost", "timestamp"])

    traffic_columns = ["speed_kmh", "flow_vph", "occupancy"]

    df[traffic_columns] = df.groupby("timestamp")[traffic_columns].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
    return df


