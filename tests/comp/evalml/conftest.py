from pathlib import Path

import pytest
from ml_orchestrator import artifacts

from tabular_orchestrated.evalml import (
    EvalMLPredict,
    EvalMLSearch,
    EvalMLSelectPipeline,
)


@pytest.fixture(scope="session")
def evalml_search_op(get_df_example: artifacts.Dataset, model_params: dict) -> EvalMLSearch:
    tmp_files_folder = Path(get_df_example.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    search_op = EvalMLSearch(
        dataset=get_df_example, automl=artifacts.Model(uri=func("automl")), search_params={}, **model_params
    )
    search_op.execute()
    return search_op


@pytest.fixture(scope="session")
def evalml_select_pipeline_op(evalml_search_op: EvalMLSearch, model_params: dict) -> EvalMLSelectPipeline:
    tmp_files_folder = Path(evalml_search_op.dataset.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    select_op = EvalMLSelectPipeline(
        automl=evalml_search_op.automl, model=artifacts.Model(uri=func("model")), pipeline_id=-1, **model_params
    )
    select_op.execute()
    return select_op


@pytest.fixture(scope="session")
def evalml_predict_op(
    evalml_select_pipeline_op: EvalMLSelectPipeline, get_df_example: artifacts.Dataset, model_params: dict
) -> EvalMLPredict:
    tmp_files_folder = Path(evalml_select_pipeline_op.automl.uri).parent

    def func(x):
        return (tmp_files_folder / x).as_posix()

    predict_op = EvalMLPredict(
        model=evalml_select_pipeline_op.model,
        test_dataset=get_df_example,
        predictions=artifacts.Dataset(uri=func("predictions")),
        pred_column="pred_column",
        proba_column=None,
        **model_params,
    )
    predict_op.execute()
    return predict_op


def test_evalml_search_op(evalml_search_op: EvalMLSearch):
    assert Path(evalml_search_op.automl.uri + ".pkl").exists()


def test_evalml_select_pipeline_op(evalml_select_pipeline_op: EvalMLSelectPipeline):
    assert Path(evalml_select_pipeline_op.model.uri + ".pkl").exists()


# def test_evalml_evaluate_pipeline_op(evalml_evaluate_pipeline_op: EvalMLEvaluatePipeline):
#     assert Path(evalml_evaluate_pipeline_op.metrics.uri+

# def test_evalml_predict_op(evalml_predict_op: EvalMLPredict):
#     assert Path(evalml_predict_op.predictions.uri+'.parquet').exists()
