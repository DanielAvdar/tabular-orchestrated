import dataclasses

import pandas as pd
from ml_orchestrator.artifacts import Dataset

from tabular_orchestrated.tab_comp import ModelComp


@dataclasses.dataclass
class EvalMLComp(ModelComp):
    extra_packages = ["evalml"]

    @staticmethod
    def load_df(dataset: Dataset) -> pd.DataFrame:
        df = super().load_df(dataset)
        return EvalMLComp.internal_feature_prep(df)
