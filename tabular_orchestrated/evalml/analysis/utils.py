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
    def multiclass_metrics(target_series: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame) -> dict[str, float]:
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
            "Precision Weighted": PrecisionWeighted().score(target_series, y_pred),
            "Precision Macro": PrecisionMacro().score(target_series, y_pred),
            "Precision Micro": PrecisionMicro().score(target_series, y_pred),
        }

    @staticmethod
    def regression_metrics(target_series: pd.Series, y_pred: pd.Series) -> dict[str, float]:
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

    @staticmethod
    def binary_metrics(target_series: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame) -> dict[str, float]:
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
