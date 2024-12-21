import dataclasses
import pickle
from typing import Any, Dict, List, Union

import pandas as pd
from ml_orchestrator.artifacts import HTML, Dataset, Metrics, Model
from ml_orchestrator.env_params import EnvironmentParams
from ml_orchestrator.meta_comp import MetaComponentV2


@dataclasses.dataclass
class TabComponent(MetaComponentV2):
    self_package_name = "tabular-orchestrated"
    extra_packages = []  # type: List[str]
    self_package_version = "{version('tabular-orchestrated')}"

    @classmethod
    def env(cls) -> EnvironmentParams:
        return EnvironmentParams(
            packages_to_install=[
                cls.self_package(),
            ],
            base_image="python:3.11",
        )

    @classmethod
    def self_package(cls) -> str:
        package_name = cls.self_package_name
        extras = ""
        if cls.extra_packages:
            extras = f"[{','.join(cls.extra_packages)}]"
        return f"{package_name}{extras}=={cls.self_package_version}"

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

    @property
    def excluded_columns(self) -> List[str]:
        return self.exclude_columns
