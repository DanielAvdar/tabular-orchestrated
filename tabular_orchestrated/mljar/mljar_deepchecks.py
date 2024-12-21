import dataclasses
from typing import List

from tabular_orchestrated.dc.dc_model import DCFullComp, DCModelComp


@dataclasses.dataclass
class MljarDCModelComp(DCModelComp):
    @property
    def extra_packages(self) -> List[str]:
        return super().extra_packages + ["mljar"]


@dataclasses.dataclass
class MljarDCFullComp(DCFullComp):
    @property
    def extra_packages(self) -> List[str]:
        return super().extra_packages + ["mljar"]
