import pandas as pd

def build_matrix(df, value_column):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    matrix = df.pivot(index="station_id", columns="timestamp", values=value_column)
    matrix = matrix.sort_index(axis=1)

    return matrix