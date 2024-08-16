from pathlib import Path

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DCDataComp, DCTrainTestComp

from ml_orchestrator import artifacts

#
# def test_mljar(get_df_example: artifacts.Dataset, split_op: DataSplitter,
#                deepchecks_op: DCFullComp, deepchecks_data_op: DCDataComp,
#                deepchecks_train_test_op: DCTrainTestComp, ) -> None:
#     tmp_files_folder = Path(get_df_example.uri).parent
#
#     def func(x):
#         return (tmp_files_folder / x).as_posix()
#
#     assert Path(func("train.parquet")).exists()
#     assert Path(func("test.parquet")).exists()
#     assert Path(func("deepchecks.html")).exists()
#     assert Path(func("deepchecks_data.html")).exists()
#     assert Path(func("deepchecks_train_test.html")).exists()


def test_split(get_df_example: artifacts.Dataset, split_op: DataSplitter) -> None:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("train.parquet")).exists()
    assert Path(func("test.parquet")).exists()


def test_data_deepchecks(
    get_df_example: artifacts.Dataset,
    split_op: DataSplitter,
    deepchecks_data_op: DCDataComp,
) -> None:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("deepchecks_data.html")).exists()


def test_split_deepchecks(
    get_df_example: artifacts.Dataset,
    deepchecks_train_test_op: DCTrainTestComp,
) -> None:
    if deepchecks_train_test_op is None:
        assert True
        return
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("deepchecks_train_test.html")).exists()
