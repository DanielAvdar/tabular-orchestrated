from pathlib import Path

from ml_orchestrator import artifacts
from sklearn import datasets

from tabular_orchestrated.mljar.mljar import EvaluateMLJAR, MLJARTraining

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
