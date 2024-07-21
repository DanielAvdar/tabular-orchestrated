import dataclasses
from abc import ABC, abstractmethod
from typing import Any

from tabular_orchestrated.tab_comp import ModelComp

from deepchecks.tabular import Dataset as DC_Dataset
from deepchecks.tabular.suites import data_integrity, full_suite, model_evaluation, train_test_validation
from ml_orchestrator import EnvironmentParams, artifacts
from ml_orchestrator.artifacts import Input, Output
from pandas import DataFrame


@dataclasses.dataclass
class DCMetaComp(ModelComp, ABC):
    report: Output[artifacts.HTML] = None
    failed_checks: Output[artifacts.Metrics] = None

    @property
    def env(self) -> EnvironmentParams:
        env = super().env
        env.packages_to_install = ["deepchecks==0.18.1", "mljar-supervised==1.1.6"] + env.packages_to_install
        return env

    def transform_dataframe(self, df: DataFrame) -> DC_Dataset:
        converted_df = df.convert_dtypes(dtype_backend="numpy_nullable")
        return DC_Dataset(df=converted_df, label=self.target_column)

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
        suite_result.save_as_html(self.report.path)
        self.save_metrics(self.failed_checks, metrics)


@dataclasses.dataclass
class DCDataComp(DCMetaComp):
    dataset: Input[artifacts.Dataset] = None

    def prepare_suite(self) -> Any:
        data = self.load_df(self.dataset)
        suite = data_integrity()
        dc_data = self.transform_dataframe(data)
        return suite.run(dc_data)


@dataclasses.dataclass
class DCTrainTestComp(DCMetaComp):
    train_dataset: Input[artifacts.Dataset] = None
    test_dataset: Input[artifacts.Dataset] = None

    def prepare_suite(self) -> Any:
        train_data = self.load_df(self.train_dataset)
        test_data = self.load_df(self.test_dataset)
        suite = train_test_validation()
        dc_train_dataset = self.transform_dataframe(train_data)
        dc_test_dataset = self.transform_dataframe(test_data)
        return suite.run(dc_train_dataset, dc_test_dataset)


@dataclasses.dataclass
class DCModelComp(DCTrainTestComp):
    model: Input[artifacts.Model] = None

    def prepare_suite(self) -> Any:
        model = self.load_model(self.model)
        train_data = self.load_df(self.train_dataset)
        test_data = self.load_df(self.test_dataset)
        suite = model_evaluation()
        dc_train_dataset = self.transform_dataframe(train_data)
        dc_test_dataset = self.transform_dataframe(test_data)
        return suite.run(dc_train_dataset, dc_test_dataset, model=model)


@dataclasses.dataclass
class DCFullComp(DCModelComp):
    def prepare_suite(self) -> Any:
        train_data = self.load_df(self.train_dataset)
        test_data = self.load_df(self.test_dataset)
        model = self.load_model(self.model)
        suite = full_suite()
        dc_train_dataset = self.transform_dataframe(train_data)
        dc_test_dataset = self.transform_dataframe(test_data)
        return suite.run(dc_train_dataset, dc_test_dataset, model)
