import dataclasses
from pathlib import Path
from typing import Dict, Union

from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output
from pandas import DataFrame
from supervised import AutoML

from tabular_orchestrated.tab_comp import ModelComp


@dataclasses.dataclass
class MLJARTraining(ModelComp):
    extra_packages = ["mljar"]
    dataset: Input[artifacts.Dataset]
    model: Output[artifacts.Model]
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

    def train_model(self, df: DataFrame) -> AutoML:
        automl = AutoML(results_path=self.get_mljar_path.as_posix(), **self.mljar_automl_params)

        x = df[df.columns.difference(self.exclude_columns + [self.target_column])]
        y = df[self.target_column]
        automl.fit(x, y)
        return automl

    @property
    def get_mljar_path(self) -> Path:
        path = Path(self.model.path).parent
        folder = path / "mljar"
        if not folder.exists():
            folder.mkdir()
        return folder


@dataclasses.dataclass
class EvaluateMLJAR(ModelComp):
    extra_packages = ["mljar"]

    test_dataset: Input[artifacts.Dataset]
    model: Input[artifacts.Model]
    metrics: Output[artifacts.Metrics]
    report: Output[artifacts.HTML]

    def execute(self) -> None:
        test_df = self.load_df(self.test_dataset)
        model = self.load_model(self.model)

        metrics = self.evaluate_model(test_df, model)
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
