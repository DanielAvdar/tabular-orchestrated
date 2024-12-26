import shutil
from pathlib import Path

import pandas as pd
import pytest
from ml_orchestrator import artifacts
from pandas_pyarrow import convert_to_pyarrow
from sklearn import datasets

from tabular_orchestrated.mljar.mljar import EvaluateMLJAR, MLJARTraining

dataset_importers = [
    datasets.load_iris,
    datasets.load_diabetes,
    datasets.load_breast_cancer,
    datasets.load_wine,
]


@pytest.fixture(params=dataset_importers, scope="session")
def get_df_example_mljar(request, tmp_files_folder) -> artifacts.Dataset:
    # dataset_func,ds_name = request.param
    dataset_func = request.param
    ds_name = dataset_func.__name__

    def func(x):
        return tmp_files_folder / ("mljar" + x)

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
def mljar_training_op(
    get_df_example_mljar: artifacts.Dataset,
    model_params: dict,
) -> MLJARTraining:
    tmp_files_folder = Path(get_df_example_mljar.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    mljar_training_op = MLJARTraining(
        dataset=get_df_example_mljar, model=artifacts.Model(uri=func("model")), **model_params
    )
    mljar_training_op.execute()
    return mljar_training_op


@pytest.fixture(scope="session")
def eval_mljar_op(
    get_df_example_mljar: artifacts.Dataset,
    mljar_training_op: MLJARTraining,
    model_params: dict,
) -> EvaluateMLJAR:
    tmp_files_folder = Path(get_df_example_mljar.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    eval_mljar_op = EvaluateMLJAR(
        test_dataset=get_df_example_mljar,
        model=mljar_training_op.model,
        metrics=artifacts.Metrics(uri=func("metrics")),
        report=artifacts.HTML(uri=func("report")),
        **model_params,
    )
    eval_mljar_op.execute()
    return eval_mljar_op
