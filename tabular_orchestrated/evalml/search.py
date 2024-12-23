import dataclasses

import pandas as pd
from evalml import AutoMLSearch
from evalml.problem_types import detect_problem_type
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output
from pandas import DataFrame

from tabular_orchestrated.evalml.evalml import EvalMLComp

pd.options.plotting.backend = "plotly"

# search_params = dict(
#     n_jobs=0,
#     # allowed_model_families=[
#     #     ModelFamily.LINEAR_MODEL,
#     #     ModelFamily.RANDOM_FOREST,
#     #     ModelFamily.EXTRA_TREES,
#     #     ModelFamily.XGBOOST,
#     #     ModelFamily.CATBOOST,
#     # ],
# )


@dataclasses.dataclass
class EvalMLSearch(EvalMLComp):
    extra_packages = ["evalml"]
    dataset: Input[artifacts.Dataset]
    automl: Output[artifacts.Model]
    search_params: dict = dataclasses.field(default_factory=dict)

    def create_search(self, df: DataFrame) -> AutoMLSearch:
        x = df[df.columns.difference([self.target_column] + self.excluded_columns)]
        y = df[self.target_column]
        problem_type = self.problem_type(y)
        self.automl.metadata["problem_type"] = problem_type
        search_params = self.search_params
        search_params["problem_type"] = problem_type
        return AutoMLSearch(X_train=x, y_train=y, **search_params)

    def execute(
        self,
    ) -> None:
        df: DataFrame = self.load_df(self.dataset)
        automl = self.create_search(df)

        automl.search(interactive_plot=False)

        self.save_model(automl, self.automl)

    def problem_type(self, y: pd.Series) -> str:
        return detect_problem_type(y)
