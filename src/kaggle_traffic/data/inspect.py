import pandas as pd
from pathlib import Path


def inspect(file: Path):
    df = pd.read_parquet(file)

    print("Shape:", df.shape)

    print("\nUnique stations:", df["station_id"].nunique())
    print("Unique links:", df["link_id"].nunique())
    print("Unique timestamps:", df["timestamp"].nunique())

    print("\nMissing states:")
    print(df["is_missing"].value_counts())

    print("\nObserved states:")
    print(df["is_observed"].value_counts())

    print("\nScore eligible:")
    print(df["is_score_eligible"].value_counts())

    print("\nImputed:")
    print(df["is_imputed"].value_counts())

    print("\nMissing rows:")
    print(
        df.loc[
            df["is_missing"] == 1,
            [
                "timestamp",
                "station_id",
                "milepost",
                "speed_kmh",
                "flow_vph",
                "occupancy",
                "is_observed",
                "is_imputed",
                "is_score_eligible",
                "is_missing",
            ],
        ].head(20)
    )

    return df