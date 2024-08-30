from pathlib import Path

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DCModelComp
from tabular_orchestrated.mljar.mljar import EvaluateMLJAR, MLJARTraining
from tabular_orchestrated.mljar.mljar_deepchecks import MljarDCModelComp

import deepchecks
import pytest
from ml_orchestrator import artifacts


@pytest.fixture(scope="session")
def mljar_training_op(get_df_example: artifacts.Dataset, split_op: DataSplitter) -> MLJARTraining:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    mljar_training_op = MLJARTraining(
        dataset=split_op.train_dataset,
        model=artifacts.Model(uri=func("model")),
    )
    mljar_training_op.execute()
    return mljar_training_op


@pytest.fixture(scope="session")
def eval_mljar_op(
    get_df_example: artifacts.Dataset, split_op: DataSplitter, mljar_training_op: MLJARTraining
) -> EvaluateMLJAR:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    eval_mljar_op = EvaluateMLJAR(
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        metrics=artifacts.Metrics(uri=func("metrics")),
        report=artifacts.HTML(uri=func("report")),
    )
    eval_mljar_op.execute()
    return eval_mljar_op


@pytest.fixture(scope="session")
def deepchecks_model_op(
    get_df_example: artifacts.Dataset, split_op: DataSplitter, mljar_training_op: MLJARTraining
) -> DCModelComp:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    deepchecks_model_op = MljarDCModelComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        model=mljar_training_op.model,
        report=artifacts.HTML(uri=func("deepchecks_model")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_model")),
    )
    try:
        deepchecks_model_op.execute()
    except deepchecks.core.errors.DatasetValidationError:
        return None
    return deepchecks_model_op
