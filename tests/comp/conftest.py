import shutil
from pathlib import Path

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DCDataComp, DCFullComp, DCTrainTestComp
from tabular_orchestrated.mljar import MLJARTraining

import pandas as pd
import pytest
from ml_orchestrator import artifacts
from pandas_pyarrow import convert_to_pyarrow
from sklearn import datasets

dataset_importers = [
    datasets.load_iris,
    datasets.load_diabetes,
    datasets.load_breast_cancer,
    datasets.load_wine,
]


@pytest.fixture(scope="session")
def test_cleaner():
    funcs = []

    def add_func(func):
        funcs.append(func)

    yield add_func
    for func in funcs:
        func()


@pytest.fixture(scope="session")
def test_directory() -> Path:
    folder_path = Path(__file__).parent
    assert folder_path.exists()
    return folder_path


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
    ds = artifacts.Dataset(uri=(folder_path / ds_name).as_posix())
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
    )
    split_op.execute()
    return split_op


@pytest.fixture(scope="session")
def deepchecks_op(
    get_df_example: artifacts.Dataset, split_op: DataSplitter, mljar_training_op: MLJARTraining
) -> DCFullComp:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    deepchecks_op = DCFullComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        report=artifacts.HTML(uri=func("deepchecks.html")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks")),
    )
    deepchecks_op.execute()
    return deepchecks_op


@pytest.fixture(scope="session")
def deepchecks_data_op(get_df_example: artifacts.Dataset, split_op: DataSplitter) -> DCDataComp:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    deepchecks_data_op = DCDataComp(
        dataset=split_op.train_dataset,
        report=artifacts.HTML(uri=func("deepchecks_data.html")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_data")),
    )
    deepchecks_data_op.execute()
    return deepchecks_data_op


@pytest.fixture(scope="session")
def deepchecks_train_test_op(get_df_example: artifacts.Dataset, split_op: DataSplitter) -> DCTrainTestComp:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    deepchecks_train_test_op = DCTrainTestComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        report=artifacts.HTML(uri=func("deepchecks_train_test.html")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_train_test")),
    )
    deepchecks_train_test_op.execute()
    return deepchecks_train_test_op
