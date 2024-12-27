import dataclasses
from typing import Any

from deepchecks.tabular.suites import data_integrity, train_test_validation
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input

from tabular_orchestrated.dc.dc_main import DCMetaComp


@dataclasses.dataclass
class DCDataComp(DCMetaComp):
    dataset: Input[artifacts.Dataset]
    as_widget: bool = True

    def prepare_suite(self) -> Any:
        data = self.load_df(self.dataset)
        suite = data_integrity()
        dc_data = self.prepare_dataset(data)
        return suite.run(dc_data)


@dataclasses.dataclass
class DCTrainTestComp(DCMetaComp):
    train_dataset: Input[artifacts.Dataset]
    test_dataset: Input[artifacts.Dataset]
    as_widget: bool = True

    def prepare_suite(self) -> Any:
        train_data = self.load_df(self.train_dataset)
        test_data = self.load_df(self.test_dataset)
        suite = train_test_validation()
        dc_train_dataset = self.prepare_dataset(train_data)
        dc_test_dataset = self.prepare_dataset(test_data)
        return suite.run(dc_train_dataset, dc_test_dataset)
