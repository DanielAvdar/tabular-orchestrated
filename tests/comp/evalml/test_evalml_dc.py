from pathlib import Path

import pytest
from ml_orchestrator import artifacts

from tabular_orchestrated.dc import DCModelComp
from tabular_orchestrated.evalml import (
    EvalMLSelectPipeline,
)


@pytest.fixture(scope="session")
def evalml_dc_op(
    evalml_select_pipeline_op: EvalMLSelectPipeline, get_df_example: artifacts.Dataset, model_params: dict
) -> DCModelComp:
    tmp_files_folder = Path(evalml_select_pipeline_op.automl.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    dc_model_op = DCModelComp(
        model=evalml_select_pipeline_op.model,
        test_dataset=get_df_example,
        train_dataset=get_df_example,
        report=artifacts.HTML(uri=func("report")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks")),
        **model_params,
    )
    dc_model_op.execute()
    return dc_model_op


def test_model_deepchecks(evalml_dc_op: DCModelComp) -> None:
    assert Path(evalml_dc_op.report.uri).exists()
