from pathlib import Path

from tabular_orchestrated.evalml import (
    EvalMLAnalysis,
    EvalMLAnalysisV2,
    EvalMLPredict,
    EvalMLSearch,
    EvalMLSelectPipeline,
)


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
    pred_artifact = evalml_predict_op.predictions
    model_input = evalml_predict_op.model

    assert Path(pred_artifact.uri + ".parquet").exists()
    problem_type = model_input.metadata["problem_type"]
    pred_df = evalml_predict_op.load_df(pred_artifact)
    assert evalml_predict_op.pred_column in pred_df.columns, "Prediction column not found"
    if problem_type != "regression":
        proba_cols = [col for col in pred_df.columns if evalml_predict_op.proba_column_prefix in col]
        assert proba_cols, "Probability column not found"


def test_evalml_analysis_op(evalml_analysis_op: EvalMLAnalysis):
    assert Path(evalml_analysis_op.analysis.uri).exists
    metrics = evalml_analysis_op.metrics.metadata
    analysis_metadata = evalml_analysis_op.analysis.metadata
    assert metrics, "Metrics are empty"
    assert analysis_metadata["number of charts"] > 0, "No charts were generated"


def test_evalml_analysis_v2_op(evalml_analysis_v2_op: EvalMLAnalysisV2):
    assert Path(evalml_analysis_v2_op.analysis.uri).exists
    metrics = evalml_analysis_v2_op.metrics.metadata
    analysis_metadata = evalml_analysis_v2_op.analysis.metadata
    assert metrics, "Metrics are empty"
    assert analysis_metadata["number of charts"] > 0, "No charts were generated"
