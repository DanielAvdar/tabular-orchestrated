import dataclasses as dc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_orchestrator import artifacts
from pandas_pyarrow import convert_to_pyarrow
from sklearn import datasets

from tabular_orchestrated.components import DataSplitter


@dc.dataclass
class DummyDataset:
    data: np.array
    target: np.array
    feature_names: list


def example_df():
    right_here = Path(__file__).parent
    dataset_examples_folder = right_here.parent / "dataset_examples"
    ds_path = dataset_examples_folder / "natality.parquet"
    target_col = "weight_pounds"
    df = pd.read_parquet(ds_path)
    features_array = df[df.columns.difference([target_col])].values
    target_array = df[target_col].values
    feature_names = df.columns.difference([target_col]).tolist()
    return DummyDataset(data=features_array, target=target_array, feature_names=feature_names)


dataset_importers = [
    example_df,
    datasets.load_iris,
    datasets.load_diabetes,
    datasets.load_breast_cancer,
    datasets.load_wine,
    datasets.load_digits,
]


@pytest.fixture(scope="session")
def test_directory() -> Path:
    folder_path = Path(__file__).parent
    assert folder_path.exists()
    return folder_path


@pytest.fixture(scope="session")
def model_params() -> dict:
    return dict(
        exclude_columns=[],
        target_column="target",
    )


@pytest.fixture(scope="session")
def tmp_files_folder(test_cleaner, test_directory) -> Path:
    folder_path = test_directory / "tmp_folder_for_tests"
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir()
    test_cleaner(lambda: shutil.rmtree(folder_path))
    return folder_path


@pytest.fixture(params=dataset_importers, scope="session")
def get_df_example(request, tmp_files_folder) -> artifacts.Dataset:
    # dataset_func,ds_name = request.param
    dataset_func = request.param
    ds_name = dataset_func.__name__

    def func(x):
        return tmp_files_folder / x

    folder_path = func(ds_name)
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir()

    data = dataset_func()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    ds = artifacts.Dataset(uri=(folder_path / ds_name).as_posix(), name=ds_name)
    adf = convert_to_pyarrow(df)
    adf.to_parquet(ds.path + ".parquet", engine="pyarrow")

    return ds


@pytest.fixture(scope="session")
def split_op(get_df_example: artifacts.Dataset) -> DataSplitter:
    ds = get_df_example

    tmp_files_folder = Path(ds.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    split_op = DataSplitter(
        dataset=ds,
        train_dataset=artifacts.Dataset(uri=func("train")),
        test_dataset=artifacts.Dataset(uri=func("test")),
        random_state=1,
    )
    split_op.execute()
    return split_op
