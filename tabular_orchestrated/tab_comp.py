import dataclasses
import pickle
from typing import Any, Dict, List, Union

import pandas as pd
from ml_orchestrator.artifacts import HTML, Dataset, Metrics, Model
from ml_orchestrator.env_params import EnvironmentParams
from ml_orchestrator.meta_comp import MetaComponentV2
from pandas_pyarrow import convert_to_numpy, convert_to_pyarrow


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
        dataset.metadata["number_of_rows"] = len(df)
        dataset.metadata["number_of_columns"] = len(df.columns)
        dataset.metadata["original_unique_dtypes"] = str(set(df.dtypes))
        arrow_df = convert_to_pyarrow(df)
        dataset.metadata["pyarrow_unique_dtypes"] = str(set(arrow_df.dtypes))
        arrow_df.to_parquet(path + ".parquet", engine="pyarrow")

    @classmethod
    def load_df(cls, dataset: Dataset) -> pd.DataFrame:
        return pd.read_parquet(dataset.path + ".parquet", engine="pyarrow")

    def save_html(self, html: HTML, str_html: str) -> None:
        html.metadata["length"] = len(str_html)
        with open(html.path, "w", encoding="utf-8") as f:
            f.write(str_html)

    def save_metrics(self, artifact: Metrics, metrics: Dict[str, Union[float, str, bool, int]]) -> None:
        for k, v in metrics.items():
            print(f"Saving metric {k}: {v}")
            artifact.log_metric(k, v)


@dataclasses.dataclass
class ModelComp(TabComponent):
    exclude_columns: List[str]
    target_column: str

    @property
    def _excluded_columns(self) -> List[str]:
        return self.exclude_columns + [self.target_column]

    def model_columns(self, df: pd.DataFrame) -> List[str]:
        return df.columns.difference(self._excluded_columns).sort_values().tolist()

    @classmethod
    def load_df(cls, dataset: Dataset) -> pd.DataFrame:
        df = super().load_df(dataset)
        return convert_to_numpy(df)

    @classmethod
    def save_model(cls, model: Any, model_path: Model) -> None:
        model_path.metadata["model_type"] = str(type(model))
        with open(model_path.path + ".pkl", "wb") as f:
            pickle.dump(model, f)
        metadata = cls.log_metadata(model)
        for key, value in metadata.items():
            model_path.metadata[key] = value

    @classmethod
    def load_model(cls, model_path: Model) -> Any:
        with open(model_path.path + ".pkl", "rb") as f:
            return pickle.load(f)

    @classmethod
    def log_metadata(cls, model: Any) -> dict[str, Union[str, bool, int, float]]:
        # Log attributes of automl of types str, bool, int, and float into metadata
        metadata = {}
        for attr in dir(model):
            if not attr.startswith("_"):  # Skip private and protected attributes
                value = getattr(model, attr, None)
                if isinstance(value, (str, bool, int, float)):
                    metadata[attr] = value
        return metadata
