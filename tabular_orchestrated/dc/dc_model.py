import dataclasses
from typing import Any

from deepchecks.tabular.suites import full_suite, model_evaluation
from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input

from tabular_orchestrated.dc.dc_data import DCTrainTestComp


@dataclasses.dataclass
class DCModelComp(DCTrainTestComp):
    model: Input[artifacts.Model]

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
