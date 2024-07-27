from pathlib import Path

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DCDataComp, DCFullComp, DCModelComp, DCTrainTestComp
from tabular_orchestrated.mljar import EvaluateMLJAR, MLJARTraining

import pandas as pd
import pytest
from ml_orchestrator import artifacts
from pandas_pyarrow import convert_to_pyarrow
from sklearn import datasets


@pytest.fixture
def get_df_example(tmp_files_folder) -> artifacts.Dataset:
    def func(x):
        return (tmp_files_folder / x).as_posix()

    data = datasets.load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    ds = artifacts.Dataset(uri=func("iris"))
    adf = convert_to_pyarrow(df)
    adf.to_parquet(ds.path + ".parquet", engine="pyarrow")

    return ds


def test_mljar(tmp_files_folder: Path, get_df_example: artifacts.Dataset) -> None:
    ds = get_df_example

    def func(x):
        return (tmp_files_folder / x).as_posix()

    split_op = DataSplitter(
        dataset=ds,
        train_dataset=artifacts.Dataset(uri=func("train")),
        test_dataset=artifacts.Dataset(uri=func("test")),
    )
    split_op.execute()
    assert Path(func("train.parquet")).exists()
    assert Path(func("test.parquet")).exists()
    mljar_training_op = MLJARTraining(
        dataset=split_op.train_dataset,
        model=artifacts.Model(uri=func("model")),
    )
    mljar_training_op.execute()
    assert Path(func("model.pkl")).exists()
    eval_mljar_op = EvaluateMLJAR(
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        metrics=artifacts.Metrics(uri=func("metrics")),
        report=artifacts.HTML(uri=func("report.html")),
    )
    eval_mljar_op.execute()
    assert Path(func("report.html")).exists()
    deepchecks_op = DCFullComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        report=artifacts.HTML(uri=func("deepchecks.html")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks")),
    )
    deepchecks_op.execute()
    assert Path(func("deepchecks.html")).exists()

    deepchecks_data_op = DCDataComp(
        dataset=split_op.train_dataset,
        # test_dataset=split_op.test_dataset,
        report=artifacts.HTML(uri=func("deepchecks_data.html")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_data")),
    )
    deepchecks_data_op.execute()
    assert Path(func("deepchecks_data.html")).exists()
    deepchecks_train_test_op = DCTrainTestComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        report=artifacts.HTML(uri=func("deepchecks_train_test.html")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_train_test")),
    )
    deepchecks_train_test_op.execute()
    assert Path(func("deepchecks_train_test.html")).exists()

    deepchecks_model_op = DCModelComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        report=artifacts.HTML(uri=func("deepchecks_model.html")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_model")),
    )
    deepchecks_model_op.execute()
    assert Path(func("deepchecks_model.html")).exists()
