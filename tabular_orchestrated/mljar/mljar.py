import dataclasses
from pathlib import Path
from typing import Dict, List, Union

from tabular_orchestrated.tab_comp import ModelComp

import pandas as pd
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype
from pandas_pyarrow import convert_to_numpy
from supervised import AutoML


@dataclasses.dataclass
class MLJARTraining(ModelComp):
    dataset: Input[artifacts.Dataset] = None
    model: Output[artifacts.Model] = None
    mljar_automl_params: Dict = dataclasses.field(
        default_factory=lambda: dict(
            total_time_limit=12 * 60 * 60,
            algorithms=[
                "Linear",
                "Random Forest",
                "Extra Trees",
                "LightGBM",
                "Xgboost",
                "CatBoost",
            ],
            train_ensemble=True,
            eval_metric="auto",
            validation_strategy={"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True},
            explain_level=2,
        )
    )

    def execute(self) -> None:
        df: DataFrame = self.load_df(self.dataset)
        model = self.train_model(df)
        self.save_model(model, self.model)

    @staticmethod
    def internal_feature_prep(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        for c in data.columns:
            if repr(data[c].dtype).startswith("halffloat"):
                data[c] = data[c].astype("double[pyarrow]")

        data = convert_to_numpy(data)
        if not is_numeric_dtype(data[target_column]):
            data[target_column] = data[target_column].astype("category").cat.codes

        for c in data.columns:
            if "Int" not in repr(data[c].dtype) and "Float" not in repr(data[c].dtype):
                continue
            if "Int" in repr(data[c].dtype) and data[c].isna().any():
                data[c] = data[c].astype("float64")
                continue
            type_str = str(data[c].dtype).lower()
            data[c] = data[c].astype(type_str)  # type: ignore
            # data[c] = data[c].values  # type: ignore

        return data

    def train_model(self, df: DataFrame) -> AutoML:
        automl = AutoML(results_path=self.get_mljar_path.as_posix(), **self.mljar_automl_params)
        mljar_df = self.internal_feature_prep(df, self.target_column)
        x = mljar_df[mljar_df.columns.difference(self.exclude_columns + [self.target_column])]
        y = mljar_df[self.target_column]
        automl.fit(x, y)
        return automl

    @property
    def get_mljar_path(self) -> Path:
        path = Path(self.model.path).parent
        folder = path / "mljar"
        if not folder.exists():
            folder.mkdir()
        return folder

    @property
    def extra_packages(self) -> List[str]:
        return ["mljar"]


@dataclasses.dataclass
class EvaluateMLJAR(ModelComp):
    test_dataset: Input[artifacts.Dataset] = None
    model: Input[artifacts.Model] = None
    metrics: Output[artifacts.Metrics] = None
    report: Output[artifacts.HTML] = None

    @property
    def extra_packages(self) -> List[str]:
        return ["mljar"]

    def execute(self) -> None:
        test_df = self.load_df(self.test_dataset)
        model = self.load_model(self.model)
        regulated_df = MLJARTraining.internal_feature_prep(test_df, self.target_column)
        metrics = self.evaluate_model(regulated_df, model)
        self.create_report(model)
        for m in metrics:
            self.metrics.log_metric(
                m,
                metrics[m],
            )

    def evaluate_model(self, test_df: DataFrame, model: AutoML) -> Dict[str, Union[float, str, bool, int]]:
        metrics: Dict[str, Union[float, str, bool, int]] = dict()
        x = test_df[test_df.columns.difference([self.target_column])]
        y = test_df[self.target_column]
        metrics["score"] = model.score(X=x, y=y)
        return metrics

    def create_report(self, model: AutoML) -> None:
        report = model.report()
        self.save_html(self.report, report.data)
