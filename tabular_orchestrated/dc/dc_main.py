import dataclasses
from abc import ABC, abstractmethod
from typing import Any

from deepchecks.tabular import Dataset as DC_Dataset
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Output
from pandas import DataFrame
from pandas_pyarrow import convert_to_numpy

from tabular_orchestrated.tab_comp import ModelComp


@dataclasses.dataclass
class DCMetaComp(ModelComp, ABC):
    extra_packages = ["deepchecks"]
    report: Output[artifacts.HTML]
    failed_checks: Output[artifacts.Metrics]

    @property
    @abstractmethod
    def as_widget(self) -> bool: ...

    def prepare_dataset(self, df: DataFrame) -> DC_Dataset:
        converted_df = convert_to_numpy(df)
        return DC_Dataset(df=converted_df, label=self.target_column, features=self.model_columns(converted_df))

    def summarize_results(self, suite_result: Any) -> dict:
        suite_list = [t.header for t in suite_result.get_not_passed_checks() if t.header is not None]
        suite_dict = {t: str(suite_result) for t in suite_list}
        return suite_dict

    @abstractmethod
    def prepare_suite(self) -> Any:
        pass

    def execute(self) -> None:
        suite_result = self.prepare_suite()
        metrics = self.summarize_results(suite_result)
        suite_result.save_as_html(self.report.path, as_widget=self.as_widget, connected=self.as_widget)
        self.save_metrics(self.failed_checks, metrics)
