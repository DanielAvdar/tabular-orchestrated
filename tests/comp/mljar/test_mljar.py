from pathlib import Path

from tabular_orchestrated.deepchecks import DCModelComp
from tabular_orchestrated.mljar.mljar import EvaluateMLJAR, MLJARTraining

from ml_orchestrator import artifacts
from sklearn import datasets

dataset_importers = [
    # (datasets.load_iris,'iris'),
    # (datasets.load_wine,'wine'),
    # (datasets.load_diabetes,'diabetes'),
    # (datasets.load_breast_cancer,'breast_cancer'),
    datasets.load_diabetes,
    datasets.load_breast_cancer,
    datasets.load_iris,
    datasets.load_wine,
]


# def test_mljar(get_df_example: artifacts.Dataset, split_op: DataSplitter, mljar_training_op: MLJARTraining,
#                eval_mljar_op: EvaluateMLJAR, deepchecks_op: DCFullComp, deepchecks_data_op: DCDataComp,
#                deepchecks_train_test_op: DCTrainTestComp, deepchecks_model_op: DCModelComp) -> None:
#     tmp_files_folder = Path(get_df_example.uri)
#
#     def func(x):
#         return (tmp_files_folder / x).as_posix()
#
#     assert Path(func("train.parquet")).exists()
#     assert Path(func("test.parquet")).exists()
#     assert Path(func("model.pkl")).exists()
#     assert Path(func("report.html")).exists()
#     assert Path(func("deepchecks.html")).exists()
#     assert Path(func("deepchecks_data.html")).exists()
#     assert Path(func("deepchecks_train_test.html")).exists()
#     assert Path(func("deepchecks_model.html")).exists()


def test_train_mljar(get_df_example: artifacts.Dataset, mljar_training_op: MLJARTraining) -> None:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("model.pkl")).exists()


def test_eval_mljar(get_df_example: artifacts.Dataset, eval_mljar_op: EvaluateMLJAR) -> None:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("report.html")).exists()


def test_mljar_deepchecks(get_df_example: artifacts.Dataset, deepchecks_model_op: DCModelComp) -> None:
    if deepchecks_model_op is None:
        assert True
        return
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("deepchecks_model.html")).exists()
