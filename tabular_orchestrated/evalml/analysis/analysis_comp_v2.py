import dataclasses

import pandas as pd
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output

from tabular_orchestrated.evalml.analysis.utils import EvalMLAnalysisUtils
from tabular_orchestrated.evalml.evalml import EvalMLComp


@dataclasses.dataclass
class EvalMLAnalysisV2(EvalMLComp):
    predictions: Input[artifacts.Dataset]
    analysis: Output[artifacts.HTML]
    metrics: Output[artifacts.Metrics]
    pred_column: str = "pred_column"
    proba_column_prefix: str = "proba_column"

    @property
    def problem_type(self) -> str:
        return self.predictions.metadata["problem_type"]

    def execute(self) -> None:
        predictions_df = self.load_df(self.predictions)
        self.analyze(predictions_df)

    def analyze(self, predictions_df):
        y_pred = predictions_df[self.pred_column]
        y_pred_proba = None
        if self.detect_problem_type(predictions_df[self.target_column]) != "regression":
            y_pred_proba = EvalMLAnalysisUtils.get_proba(predictions_df, self.proba_column_prefix)

        labels = predictions_df[self.target_column]
        str_charts = self.create_charts(labels, y_pred, y_pred_proba)
        metrics = EvalMLAnalysisUtils.create_metrics(labels, y_pred, y_pred_proba)
        for metric_name, metric_value in metrics.items():
            self.metrics.log_metric(metric_name, metric_value)
        html_str = "<br>".join(str_charts)
        self.save_html(self.analysis, html_str)
        self.analysis.metadata["number of charts"] = len(str_charts)

    def create_charts(self, labels: pd.Series, y_pred, y_pred_proba) -> list[str]:
        charts = EvalMLAnalysisUtils.create_metric_charts(labels, y_pred, y_pred_proba)

        str_charts = [chart.to_html() for chart in charts]
        return str_charts
