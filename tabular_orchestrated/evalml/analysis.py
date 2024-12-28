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
    MAE,
    MAPE,
    MSE,
    R2,
    SMAPE,
    AccuracyBinary,
    AccuracyMulticlass,
    AUCMacro,
    AUCMicro,
    AUCWeighted,
    BalancedAccuracyBinary,
    BalancedAccuracyMulticlass,
    ExpVariance,
    F1Macro,
    F1Micro,
    F1Weighted,
    Gini,
    LogLossBinary,
    LogLossMulticlass,
    MaxError,
    MCCBinary,
    MCCMulticlass,
    MeanSquaredLogError,
    MedianAE,
    Precision,
    PrecisionWeighted,
    Recall,
    RecallMacro,
    RecallMicro,
    RecallWeighted,
    RootMeanSquaredError,
    RootMeanSquaredLogError,
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
        target_series = test_df[self.target_column]
        if problem_type == "binary" or problem_type == "multiclass":
            y_pred_proba = model.predict_proba(test_df[self.model_columns(test_df)])

        metrics = {}
        if problem_type == "binary":
            metrics = self.binary_metrics(target_series, y_pred, y_pred_proba)
        elif problem_type == "regression":
            metrics = self.regression_metrics(target_series, y_pred)
        elif problem_type == "multiclass":
            metrics = self.multiclass_metrics(target_series, y_pred, y_pred_proba)

        for metric_name, metric_value in metrics.items():
            self.metrics.log_metric(metric_name, metric_value)

    def multiclass_metrics(self, target_series, y_pred, y_pred_proba):
        return {
            "Log Loss": LogLossMulticlass().score(target_series, y_pred_proba),
            "AUC": AUCWeighted().score(target_series, y_pred_proba),
            "AUC Micro": AUCMicro().score(target_series, y_pred_proba),
            "AUC Macro": AUCMacro().score(target_series, y_pred_proba),
            "MCC Multiclass": MCCMulticlass().score(target_series, y_pred),
            "Precision": PrecisionWeighted().score(target_series, y_pred),
            "Recall": RecallWeighted().score(target_series, y_pred),
            "Recall Micro": RecallMicro().score(target_series, y_pred),
            "Recall Macro": RecallMacro().score(target_series, y_pred),
            "Accuracy Multiclass": AccuracyMulticlass().score(target_series, y_pred),
            "Balanced Accuracy Multiclass": BalancedAccuracyMulticlass().score(target_series, y_pred),
            "F1 Weighted": F1Weighted().score(target_series, y_pred),
            "F1 Macro": F1Macro().score(target_series, y_pred),
            "F1 Micro": F1Micro().score(target_series, y_pred),
        }

    def regression_metrics(self, target_series, y_pred):
        return {
            "R2": R2().score(target_series, y_pred),
            "RMSE": RootMeanSquaredError().score(target_series, y_pred),
            "MAE": MAE().score(target_series, y_pred),
            "MSE": MSE().score(target_series, y_pred),
            "MAPE": MAPE().score(target_series, y_pred),
            "SMAPE": SMAPE().score(target_series, y_pred),
            "MSLE": MeanSquaredLogError().score(target_series, y_pred),
            "RMSLE": RootMeanSquaredLogError().score(target_series, y_pred),
            "Max Error": MaxError().score(target_series, y_pred),
            "Exp Var": ExpVariance().score(target_series, y_pred),
            "Median Absolute Error": MedianAE().score(target_series, y_pred),
        }

    def binary_metrics(self, target_series, y_pred, y_pred_proba):
        return {
            "AUC": AUC().score(target_series, y_pred),
            "F1": F1().score(target_series, y_pred),
            "Log Loss": LogLossBinary().score(target_series, y_pred),
            "Precision": Precision().score(target_series, y_pred),
            "Recall": Recall().score(target_series, y_pred),
            "Balanced Accuracy": BalancedAccuracyBinary().score(target_series, y_pred),
            "Accuracy Binary": AccuracyBinary().score(target_series, y_pred),
            "MCC Binary": MCCBinary().score(target_series, y_pred),
            "Gini": Gini().score(target_series, y_pred),
        }
