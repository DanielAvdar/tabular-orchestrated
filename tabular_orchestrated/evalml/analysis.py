import dataclasses
from typing import Union

import pandas as pd
from evalml.model_understanding import (
    confusion_matrix,
    graph_confusion_matrix,
    graph_precision_recall_curve,
    graph_prediction_vs_actual,
    graph_roc_curve,
)
from evalml.objectives import (
    AUC,
    F1,
    R2,
    AUCWeighted,
    LogLossBinary,
    LogLossMulticlass,
    MCCMulticlass,
    Precision,
    Recall,
    RootMeanSquaredError,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    PipelineBase,
    RegressionPipeline,
)
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output

from tabular_orchestrated.evalml.evalml import EvalMLComp

EvalMLModel = Union[PipelineBase, BinaryClassificationPipeline, RegressionPipeline, MulticlassClassificationPipeline]


@dataclasses.dataclass
class EvalMLAnalysis(EvalMLComp):
    model: Input[artifacts.Model]
    test_dataset: Input[artifacts.Dataset]
    analysis: Output[artifacts.HTML]
    metrics: Output[artifacts.Metrics]

    def execute(self) -> None:
        model = self.load_model(self.model)
        test_df = self.load_df(self.test_dataset)
        self.create_charts(model, test_df)
        self.create_metrics(model, test_df)

    def create_charts(self, model: EvalMLModel, test_df: pd.DataFrame) -> None:
        charts = []
        labels = test_df[self.target_column]
        fimp = model.graph_feature_importance()
        charts.append(fimp)
        y_pred = model.predict(test_df[self.model_columns(test_df)])
        problem_type = self.model.metadata["problem_type"]
        if problem_type == "binary":
            y_pred_proba = model.predict_proba(test_df[self.model_columns(test_df)])
            roc = graph_roc_curve(test_df[self.target_column], y_pred_proba)
            prc = graph_precision_recall_curve(test_df[self.target_column], y_pred_proba)
            confusion = graph_confusion_matrix(test_df[self.target_column], y_pred)
            charts.extend([roc, prc, confusion])
        elif problem_type == "regression":
            pred_vs_true = graph_prediction_vs_actual(labels, y_pred, outlier_threshold=50)
            charts.append(pred_vs_true)
        elif problem_type == "multiclass":
            confusion = confusion_matrix(labels, y_pred)
            charts.append(confusion)
        str_charts = [chart.to_html() for chart in charts]
        html_str = "<br>".join(str_charts)
        self.save_html(self.analysis, html_str)

    def create_metrics(self, model: EvalMLModel, test_df: pd.DataFrame) -> None:
        y_pred = model.predict(test_df[self.model_columns(test_df)])
        problem_type = self.model.metadata["problem_type"]
        y_pred_proba = None
        if problem_type == "binary" or problem_type == "multiclass":
            y_pred_proba = model.predict_proba(test_df[self.model_columns(test_df)])

        if problem_type == "binary":
            auc = AUC().score(test_df[self.target_column], y_pred)
            f1 = F1().score(test_df[self.target_column], y_pred)
            logloss = LogLossBinary().score(test_df[self.target_column], y_pred_proba)
            precision = Precision().score(test_df[self.target_column], y_pred)
            recall = Recall().score(test_df[self.target_column], y_pred)
            self.metrics.log_metric("AUC", auc)
            self.metrics.log_metric("F1", f1)
            self.metrics.log_metric("Log Loss", logloss)
            self.metrics.log_metric("Precision", precision)
            self.metrics.log_metric("Recall", recall)
        elif problem_type == "regression":
            r2 = R2().score(test_df[self.target_column], y_pred)
            rmse = RootMeanSquaredError().score(test_df[self.target_column], y_pred)
            self.metrics.log_metric("R2", r2)
            self.metrics.log_metric("RMSE", rmse)
        elif problem_type == "multiclass":
            logloss = LogLossMulticlass().score(test_df[self.target_column], y_pred_proba)
            auc = AUCWeighted().score(test_df[self.target_column], y_pred_proba)
            MCCMulticlass().score(test_df[self.target_column], y_pred)
            self.metrics.log_metric("Log Loss", logloss)
