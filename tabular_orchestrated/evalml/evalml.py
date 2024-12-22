import dataclasses

import pandas as pd
from ml_orchestrator.artifacts import Dataset
from ml_orchestrator.env_params import EnvironmentParams

from tabular_orchestrated.tab_comp import ModelComp


@dataclasses.dataclass
class EvalMLComp(ModelComp):
    extra_packages = ["evalml"]

    @classmethod
    def env(cls) -> EnvironmentParams:
        env = super().env()
        env.base_image = "python:3.10"
        return env

    def load_df(self, dataset: Dataset) -> pd.DataFrame:
        df = super().load_df(dataset)
        return self.internal_feature_prep(df)
