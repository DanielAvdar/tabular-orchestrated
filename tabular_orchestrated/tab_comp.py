import dataclasses
import pickle
from typing import Any, Dict, List, Union

import pandas as pd
from ml_orchestrator import MetaComponent
from ml_orchestrator.artifacts import HTML, Dataset, Metrics, Model
from ml_orchestrator.env_params import EnvironmentParams


@dataclasses.dataclass
class TabComponent(MetaComponent):
    @property
    def env(self) -> EnvironmentParams:
        return EnvironmentParams(
            packages_to_install=[
                self.self_package,
            ],
            base_image="python:3.11",
        )

    @property
    def self_package(self) -> str:
        package_name = self.self_package_name
        extras = ""
        if self.extra_packages:
            extras = f"[{','.join(self.extra_packages)}]"
        return f"{package_name}{extras}=={self.self_package_version}"

    @property
    def extra_packages(self) -> List[str]:
        return []

    @property
    def self_package_version(self) -> str:
        return "{version('tabular-orchestrated')}"

    @property
    def self_package_name(self) -> str:
        return "tabular-orchestrated"

    @staticmethod
    def save_df(df: pd.DataFrame, dataset: Dataset) -> None:
        path = dataset.path
        df.to_parquet(path + ".parquet", engine="pyarrow")

    @staticmethod
    def load_df(dataset: Dataset) -> pd.DataFrame:
        return pd.read_parquet(dataset.path + ".parquet", engine="pyarrow")

    @staticmethod
    def save_model(model: Any, model_path: Model) -> None:
        with open(model_path.path + ".pkl", "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(model_path: Model) -> Any:
        with open(model_path.path + ".pkl", "rb") as f:
            return pickle.load(f)

    def save_html(self, html: HTML, path: str) -> None:
        with open(html.path, "w") as f:
            f.write(path)

    def save_metrics(self, artifact: Metrics, metrics: Dict[str, Union[float, str, bool, int]]) -> None:
        for k, v in metrics.items():
            print(f"Saving metric {k}: {v}")
            artifact.log_metric(k, v)


@dataclasses.dataclass
class ModelComp(TabComponent):
    exclude_columns: List[str] = dataclasses.field(default_factory=lambda: [])
    target_column: str = "target"
