from pathlib import Path

import deepchecks
import pytest
from ml_orchestrator import artifacts

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.dc import DCDataComp, DCTrainTestComp
from tabular_orchestrated.dc.dc_model_v2 import _DCModelCompV2


@pytest.fixture(scope="session")
def deepchecks_data_op(
    get_df_example: artifacts.Dataset,
    split_op: DataSplitter,
    model_params: dict,
) -> DCDataComp:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    deepchecks_data_op = DCDataComp(
        dataset=split_op.train_dataset,
        report=artifacts.HTML(uri=func("deepchecks_data")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_data")),
        **model_params,
    )
    deepchecks_data_op.execute()
    return deepchecks_data_op


@pytest.fixture(scope="session")
def deepchecks_train_test_op(
    get_df_example: artifacts.Dataset,
    split_op: DataSplitter,
    model_params: dict,
) -> DCTrainTestComp:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    deepchecks_train_test_op = DCTrainTestComp(
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        report=artifacts.HTML(uri=func("deepchecks_train_test")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_train_test")),
        **model_params,
    )
    try:
        deepchecks_train_test_op.execute()
    except deepchecks.core.errors.DatasetValidationError:
        return None
    return deepchecks_train_test_op


@pytest.fixture(scope="session")
def deepchecks_model_v2_op(
    get_df_example: artifacts.Dataset,
    split_op: DataSplitter,
    model_params: dict,
) -> _DCModelCompV2:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    deepchecks_train_test_op = _DCModelCompV2(
        pred_column="target",
        train_dataset=split_op.train_dataset,
        test_dataset=split_op.test_dataset,
        report=artifacts.HTML(uri=func("deepchecks_train_test")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_train_test")),
        **model_params,
    )
    try:
        deepchecks_train_test_op.execute()
    except deepchecks.core.errors.DatasetValidationError as e:
        print(e)
        return None
    return deepchecks_train_test_op
