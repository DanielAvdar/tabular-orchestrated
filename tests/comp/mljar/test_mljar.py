from pathlib import Path

from ml_orchestrator import artifacts

from tabular_orchestrated.mljar.mljar import EvaluateMLJAR, MLJARTraining


def test_train_mljar(get_df_example_mljar: artifacts.Dataset, mljar_training_op: MLJARTraining) -> None:
    tmp_files_folder = Path(get_df_example_mljar.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("model.pkl")).exists()


def test_eval_mljar(get_df_example_mljar: artifacts.Dataset, eval_mljar_op: EvaluateMLJAR) -> None:
    tmp_files_folder = Path(get_df_example_mljar.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    assert Path(func("report.html")).exists()
