import dataclasses

import pandas as pd
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output

from tabular_orchestrated.evalml.analysis.utils import EvalMLAnalysisUtils, EvalMLModel
from tabular_orchestrated.evalml.evalml import EvalMLComp


@dataclasses.dataclass
class EvalMLAnalysis(EvalMLComp):
    model: Input[artifacts.Model]
    test_dataset: Input[artifacts.Dataset]
    analysis: Output[artifacts.HTML]
    metrics: Output[artifacts.Metrics]

    @property
    def problem_type(self) -> str:
        return self.model.metadata["problem_type"]

    def execute(self) -> None:
        model = self.load_model(self.model)
        test_df = self.load_df(self.test_dataset)
        self.analyze(model, test_df)

    def analyze(self, model: EvalMLModel, test_df: pd.DataFrame) -> None:
        y_pred = model.predict(test_df[self.model_columns(test_df)])
        y_pred_proba = (
            model.predict_proba(test_df[self.model_columns(test_df)]) if self.problem_type != "regression" else None
        )
        labels = test_df[self.target_column]
        str_charts = self.create_charts(model, labels, y_pred, y_pred_proba)
        metrics = EvalMLAnalysisUtils.create_metrics(labels, y_pred, y_pred_proba)
        for metric_name, metric_value in metrics.items():
            self.metrics.log_metric(metric_name, metric_value)
        html_str = "<br>".join(str_charts)
        self.save_html(self.analysis, html_str)
        self.analysis.metadata["number of charts"] = len(str_charts)

    def create_charts(
        self, model: EvalMLModel, labels: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame | None
    ) -> list[str]:
        charts = EvalMLAnalysisUtils.create_metric_charts(labels, y_pred, y_pred_proba)

        charts.append(model.graph_feature_importance())
        str_charts = [chart.to_html() for chart in charts]
        return str_charts
