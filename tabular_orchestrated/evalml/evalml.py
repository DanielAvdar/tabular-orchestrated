import dataclasses

from tabular_orchestrated.tab_comp import ModelComp


@dataclasses.dataclass
class EvalMLComp(ModelComp):
    extra_packages = ["evalml"]
