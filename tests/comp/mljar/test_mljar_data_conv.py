from tabular_orchestrated.mljar.mljar import MLJARTraining

import hypothesis as hp
import pandas as pd
import pytest
from hypothesis.extra.pandas import column, data_frames
from pandas_pyarrow import convert_to_pyarrow


@hp.given(
    df=data_frames(
        columns=[
            column("A", dtype=int),
            # column("B", dtype=str),
            column(
                "C",
                dtype=float,
            ),
            column("D", dtype="float64"),
            column("E", dtype="int64"),
            column("F", dtype="int32"),
            column("G", dtype="int16"),
            column("H", dtype="int8"),
            column("I", dtype="float32"),
            column("J", dtype="float16"),
        ]
    )
)
@hp.settings(suppress_health_check=[hp.HealthCheck.too_slow])
def test_internal_feature_prep_hypothesis(df):
    target_column = "A"
    pyarrow_df = convert_to_pyarrow(df)
    result_df = MLJARTraining.internal_feature_prep(pyarrow_df, target_column)
    for c in result_df.columns:
        assert "Int" not in repr(result_df[c].dtype)
        assert "Float" not in repr(result_df[c].dtype)
    hp.assume(result_df["J"].isna().sum() > 0)
    for c in result_df.columns:
        assert "Int" not in repr(result_df[c].dtype)
        assert "Float" not in repr(result_df[c].dtype)


@pytest.fixture
def example_df():
    df = pd.DataFrame({
        # "A": [1, 2, 3, 4, 5],
        # "B": ["a", "b", "c", "d", "e"],
        # "C": [1.1, 2.2, 3.3, 4.4, 5.5],
        "A": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
        "B": pd.Series(["a", "b", "c", "d", "e"], dtype="str"),
        "C": pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], dtype="float64"),
        "D": pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], dtype="float16"),
        "E": pd.Series([1, 2, 3, 4, 5], dtype="int16"),
        "F": pd.Series([0.1, 0.2, 0.3, 0.4, None], dtype="float32"),
        "G": pd.Series([], dtype="float16"),
        "H": pd.Series([1], dtype="int8"),
    })
    return df


def test_internal_feature_prep_numeric_target(example_df):
    target_column = "A"
    pyarrow_example_df = convert_to_pyarrow(example_df)
    result_df = MLJARTraining.internal_feature_prep(pyarrow_example_df, target_column)
    for c in result_df.columns:
        assert "Int" not in repr(result_df[c].dtype)
        assert "Float" not in repr(result_df[c].dtype)


def test_example_df(dataset_examples_folder):
    ds_path = dataset_examples_folder / "natality.parquet"

    nat_df = pd.read_parquet(ds_path)
    target_column = "weight_pounds"
    MLJARTraining.internal_feature_prep(nat_df, target_column)
