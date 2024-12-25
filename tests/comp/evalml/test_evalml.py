from pathlib import Path

from tabular_orchestrated.evalml import EvalMLPredict, EvalMLSearch, EvalMLSelectPipeline


def test_evalml_search_op(evalml_search_op: EvalMLSearch):
    assert Path(evalml_search_op.automl.uri + ".pkl").exists()


def test_evalml_select_pipeline_op(evalml_select_pipeline_op: EvalMLSelectPipeline):
    assert Path(evalml_select_pipeline_op.model.uri + ".pkl").exists()


def test_evalml_predict_op(evalml_predict_op: EvalMLPredict):
    assert Path(evalml_predict_op.predictions.uri + ".parquet").exists()
