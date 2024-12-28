import dataclasses

import pandas as pd
from evalml.problem_types import detect_problem_type

from tabular_orchestrated.tab_comp import ModelComp


@dataclasses.dataclass
class EvalMLComp(ModelComp):
    extra_packages = ["evalml"]

    @staticmethod
    def detect_problem_type(y: pd.Series) -> str:
        pt = detect_problem_type(y)
        return str(pt)
