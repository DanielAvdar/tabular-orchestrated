import dataclasses
from typing import Union

# Set default Plotly template to dark mode
import pandas as pd
import plotly.io as plt_io
from evalml.model_understanding import (
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
    PrecisionMacro,
    PrecisionMicro,
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
from evalml.problem_types import detect_problem_type
from plotly.graph_objs import Figure

plt_io.templates.default = "plotly_dark"

EvalMLModel = Union[PipelineBase, BinaryClassificationPipeline, RegressionPipeline, MulticlassClassificationPipeline]


@dataclasses.dataclass
class EvalMLAnalysisUtils:
    @staticmethod
    def detect_problem_type(y: pd.Series) -> str:
        pt = detect_problem_type(y)
        return str(pt)

    @classmethod
    def get_proba_cols(cls, df: pd.DataFrame, column_prefix: str) -> list[str]:
        get_proba_cols = [col for col in df.columns if col.startswith(column_prefix)]
        return get_proba_cols

    @classmethod
    def get_proba(cls, df: pd.DataFrame, column_prefix: str) -> pd.DataFrame:
        get_proba_cols = cls.get_proba_cols(df, column_prefix)
        return df[get_proba_cols]

    @classmethod
    def create_metrics(cls, labels: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame) -> dict[str, float]:
        metrics = {}
        problem_type = cls.detect_problem_type(labels)
        if problem_type == "binary":
            metrics = cls.binary_metrics(labels, y_pred, y_pred_proba)
        elif problem_type == "regression":
            metrics = cls.regression_metrics(labels, y_pred)
        elif problem_type == "multiclass":
            metrics = cls.multiclass_metrics(labels, y_pred, y_pred_proba)

        return metrics

    @staticmethod
    def compute_metrics(metrics_dict: dict, metric_name: str, scorer, *args):
        try:
            metrics_dict[metric_name] = scorer.score(*args)
        except Exception as e:
            metrics_dict[metric_name] = f"Error: {str(e)}"

    @staticmethod
    def multiclass_metrics(target_series: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame) -> dict[str, float]:
        metrics = {}
        scorers = {
            "Log Loss": LogLossMulticlass(),
            "AUC": AUCWeighted(),
            "AUC Micro": AUCMicro(),
            "AUC Macro": AUCMacro(),
            "MCC Multiclass": MCCMulticlass(),
            "Precision": PrecisionWeighted(),
            "Recall": RecallWeighted(),
            "Recall Micro": RecallMicro(),
            "Recall Macro": RecallMacro(),
            "Accuracy Multiclass": AccuracyMulticlass(),
            "Balanced Accuracy Multiclass": BalancedAccuracyMulticlass(),
            "F1 Weighted": F1Weighted(),
            "F1 Macro": F1Macro(),
            "F1 Micro": F1Micro(),
            "Precision Weighted": PrecisionWeighted(),
            "Precision Macro": PrecisionMacro(),
            "Precision Micro": PrecisionMicro(),
        }
        for name, scorer in scorers.items():
            args = (target_series, y_pred_proba if "AUC" in name or "Log Loss" in name else y_pred)
            EvalMLAnalysisUtils.compute_metrics(metrics, name, scorer, *args)
        return metrics

    @staticmethod
    def regression_metrics(target_series: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        metrics = {}
        scorers = {
            "R2": R2(),
            "RMSE": RootMeanSquaredError(),
            "MAE": MAE(),
            "MSE": MSE(),
            "MAPE": MAPE(),
            "SMAPE": SMAPE(),
            "MSLE": MeanSquaredLogError(),
            "RMSLE": RootMeanSquaredLogError(),
            "Max Error": MaxError(),
            "Exp Var": ExpVariance(),
            "Median Absolute Error": MedianAE(),
        }
        for name, scorer in scorers.items():
            EvalMLAnalysisUtils.compute_metrics(metrics, name, scorer, target_series, y_pred)
        return metrics

    @staticmethod
    def binary_metrics(target_series: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame) -> dict[str, float]:
        metrics = {}
        scorers = {
            "AUC": AUC(),
            "F1": F1(),
            "Log Loss": LogLossBinary(),
            "Precision": Precision(),
            "Recall": Recall(),
            "Balanced Accuracy": BalancedAccuracyBinary(),
            "Accuracy Binary": AccuracyBinary(),
            "MCC Binary": MCCBinary(),
            "Gini": Gini(),
        }
        for name, scorer in scorers.items():
            args = (target_series, y_pred_proba if name == "Log Loss" else y_pred)
            EvalMLAnalysisUtils.compute_metrics(metrics, name, scorer, *args)
        return metrics

    @classmethod
    def create_metric_charts(cls, labels: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame) -> list[Figure]:
        charts = []
        problem_type = cls.detect_problem_type(labels)
        if problem_type == "binary":
            roc = graph_roc_curve(labels, y_pred_proba)
            prc = graph_precision_recall_curve(labels, y_pred_proba)
            confusion = graph_confusion_matrix(labels, y_pred)
            charts.extend([
                roc,
                prc,
                confusion,
            ])
        elif problem_type == "regression":
            pred_vs_true = graph_prediction_vs_actual(labels, y_pred, outlier_threshold=50)

            charts.extend([pred_vs_true])
        elif problem_type == "multiclass":
            confusion = graph_confusion_matrix(labels, y_pred)
            charts.extend([
                confusion,
            ])

        return charts
