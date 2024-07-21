from pathlib import Path

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DCFullComp
from tabular_orchestrated.mljar import EvaluateMLJAR, MLJARTraining

import pandas as pd
import pytest
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts.artifacts import Metrics
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
    mljar_training_op = MLJARTraining(
        dataset=split_op.train_dataset,
        model=artifacts.Model(uri=func("model")),
    )
    eval_mljar_op = EvaluateMLJAR(
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        metrics=artifacts.Metrics(uri=func("metrics")),
        report=artifacts.HTML(uri=func("report")),
    )

    deepchecks_op = DCFullComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        report=artifacts.HTML(uri=func("deepchecks")),
        failed_checks=Metrics(uri=func("failed_checks")),
    )
    split_op.execute()
    mljar_training_op.execute()
    eval_mljar_op.execute()
    deepchecks_op.execute()
    print(eval_mljar_op.metrics)
    assert True
