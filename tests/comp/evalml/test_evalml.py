from pathlib import Path

from tabular_orchestrated.evalml import EvalMLPredict, EvalMLSearch, EvalMLSelectPipeline


def test_evalml_search_op(evalml_search_op: EvalMLSearch):
    assert Path(evalml_search_op.automl.uri + ".pkl").exists()
    automl_metadata = evalml_search_op.automl.metadata
    for k, v in automl_metadata.items():
        assert isinstance(v, (int, float, bool, str)), f"Type {type(v)} invalid in {k}"
        assert isinstance(k, str), f"Type {type(k)} is not a valid metric name"


def test_evalml_select_pipeline_op(evalml_select_pipeline_op: EvalMLSelectPipeline):
    assert Path(evalml_select_pipeline_op.model.uri + ".pkl").exists()
    model_metadata = evalml_select_pipeline_op.model.metadata
    for k, v in model_metadata.items():
        assert isinstance(v, (int, float, bool, str)), f"Type {type(v)} invalid in {k}"
        assert isinstance(k, str), f"Type {type(k)} is not a valid metric name"


def test_evalml_predict_op(evalml_predict_op: EvalMLPredict):
    assert Path(evalml_predict_op.predictions.uri + ".parquet").exists()
