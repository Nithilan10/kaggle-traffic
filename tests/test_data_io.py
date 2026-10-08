import zipfile
from pathlib import Path

import pandas as pd

from kaggle_traffic.data.inspect import inspect
from kaggle_traffic.data.unzip import unzip_file


def _states():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2030-06-01", periods=4, freq="5min", tz="UTC"),
            "station_id": ["s1", "s1", "s2", "s2"],
            "link_id": ["l1", "l1", "l2", "l2"],
            "milepost": [0.0, 0.0, 1.5, 1.5],
            "speed_kmh": [100.0, None, 80.0, 90.0],
            "flow_vph": [1000.0, None, 1200.0, 1100.0],
            "occupancy": [0.1, None, 0.2, 0.15],
            "is_observed": [1, 0, 1, 1],
            "is_imputed": [0, 1, 0, 0],
            "is_score_eligible": [1, 0, 1, 1],
            "is_missing": [0, 1, 0, 0],
        }
    )


def test_unzip_file_extracts_into_a_new_directory(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("corridor")
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source, arcname="note.txt")

    destination = tmp_path / "out" / "nested"
    unzip_file(Path(archive), Path(destination))

    assert (destination / "note.txt").read_text() == "corridor"


def test_inspect_returns_the_frame_and_reports_its_shape(tmp_path, capsys):
    frame = _states()
    path = tmp_path / "states.parquet"
    frame.to_parquet(path)

    loaded = inspect(path)
    printed = capsys.readouterr().out

    pd.testing.assert_frame_equal(loaded, frame)
    assert "Shape: (4, 11)" in printed
    assert "Unique stations: 2" in printed
    assert "Unique timestamps: 4" in printed
