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

        if problem_type == "binary":
            self.binary_metrics(target_series, y_pred, y_pred_proba)
        elif problem_type == "regression":
            self.regression_metrics(target_series, y_pred)
        elif problem_type == "multiclass":
            self.multiclass_metrics(target_series, y_pred, y_pred_proba)

    def multiclass_metrics(self, target_series, y_pred, y_pred_proba):
        logloss = LogLossMulticlass().score(target_series, y_pred_proba)
        auc = AUCWeighted().score(target_series, y_pred_proba)
        auc_micro = AUCMicro().score(target_series, y_pred_proba)
        auc_macro = AUCMacro().score(target_series, y_pred_proba)
        mccmulticlass = MCCMulticlass().score(target_series, y_pred)
        precision = PrecisionWeighted().score(target_series, y_pred)
        recall = RecallWeighted().score(target_series, y_pred)
        recall_micro = RecallMicro().score(target_series, y_pred)
        recall_macro = RecallMacro().score(target_series, y_pred)
        accuracy_multiclass = AccuracyMulticlass().score(target_series, y_pred)
        balanced_accuracy_multiclass = BalancedAccuracyMulticlass().score(target_series, y_pred)
        f1weighted = F1Weighted().score(target_series, y_pred)
        f1macro = F1Macro().score(target_series, y_pred)
        f1micro = F1Micro().score(target_series, y_pred)

        self.metrics.log_metric("Log Loss", logloss)
        self.metrics.log_metric("AUC", auc)
        self.metrics.log_metric("AUC Micro", auc_micro)
        self.metrics.log_metric("AUC Macro", auc_macro)
        self.metrics.log_metric("MCC Multiclass", mccmulticlass)
        self.metrics.log_metric("Precision", precision)
        self.metrics.log_metric("Recall", recall)
        self.metrics.log_metric("Recall Micro", recall_micro)
        self.metrics.log_metric("Recall Macro", recall_macro)
        self.metrics.log_metric("Accuracy Multiclass", accuracy_multiclass)
        self.metrics.log_metric("Balanced Accuracy Multiclass", balanced_accuracy_multiclass)
        self.metrics.log_metric("F1 Weighted", f1weighted)
        self.metrics.log_metric("F1 Macro", f1macro)
        self.metrics.log_metric("F1 Micro", f1micro)

    def regression_metrics(self, target_series, y_pred):
        r2 = R2().score(target_series, y_pred)
        rmse = RootMeanSquaredError().score(target_series, y_pred)
        mae = MAE().score(target_series, y_pred)
        mse = MSE().score(target_series, y_pred)
        mape = MAPE().score(target_series, y_pred)
        smape = SMAPE().score(target_series, y_pred)
        msle = MeanSquaredLogError().score(target_series, y_pred)
        rmsle = RootMeanSquaredLogError().score(target_series, y_pred)
        max_error = MaxError().score(target_series, y_pred)
        exp_var = ExpVariance().score(target_series, y_pred)
        median_ae = MedianAE().score(target_series, y_pred)

        self.metrics.log_metric("R2", r2)
        self.metrics.log_metric("RMSE", rmse)
        self.metrics.log_metric("MAE", mae)
        self.metrics.log_metric("MSE", mse)
        self.metrics.log_metric("MAPE", mape)
        self.metrics.log_metric("SMAPE", smape)
        self.metrics.log_metric("MSLE", msle)
        self.metrics.log_metric("RMSLE", rmsle)
        self.metrics.log_metric("Max Error", max_error)
        self.metrics.log_metric("Exp Var", exp_var)
        self.metrics.log_metric("Median Absolute Error", median_ae)

    def binary_metrics(self, target_series, y_pred, y_pred_proba):
        auc = AUC().score(target_series, y_pred)
        f1 = F1().score(target_series, y_pred)
        logloss = LogLossBinary().score(target_series, y_pred)
        precision = Precision().score(target_series, y_pred)
        recall = Recall().score(target_series, y_pred)
        balanced_accuracy = BalancedAccuracyBinary().score(target_series, y_pred)
        accuracy_binary = AccuracyBinary().score(target_series, y_pred)
        mccbinary = MCCBinary().score(target_series, y_pred)
        gini = Gini().score(target_series, y_pred)

        self.metrics.log_metric("AUC", auc)
        self.metrics.log_metric("F1", f1)
        self.metrics.log_metric("Log Loss", logloss)
        self.metrics.log_metric("Precision", precision)
        self.metrics.log_metric("Recall", recall)
        self.metrics.log_metric("Balanced Accuracy", balanced_accuracy)
        self.metrics.log_metric("Accuracy Binary", accuracy_binary)
        self.metrics.log_metric("MCC Binary", mccbinary)
        self.metrics.log_metric("Gini", gini)
