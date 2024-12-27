import dataclasses
from typing import Any, List

from deepchecks.tabular.suites import model_evaluation

from tabular_orchestrated.dc import DCTrainTestComp


@dataclasses.dataclass
class _DCModelCompV2(DCTrainTestComp):
    pred_column: str = "pred_column"
    proba_column: str = "proba_column"
    as_widget: bool = True

    def prepare_suite(self) -> Any:
        train_data = self.load_df(self.train_dataset)
        test_data = self.load_df(self.test_dataset)
        suite = model_evaluation()
        dc_train_dataset = self.prepare_dataset(train_data)
        dc_test_dataset = self.prepare_dataset(test_data)
        y_pred_train = train_data[self.pred_column].values
        y_pred_test = test_data[self.pred_column].values
        proba = dict(
            y_proba_train=train_data[self.proba_column].values if self.proba_column in train_data.columns else None,
            y_proba_test=test_data[self.proba_column].values if self.proba_column in test_data.columns else None,
        )

        return suite.run(
            dc_train_dataset,
            dc_test_dataset,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            **proba,
        )


@dataclasses.dataclass
class DCModelCompV2(_DCModelCompV2):
    @property
    def _excluded_columns(self) -> List[str]:
        ex_cols = super()._excluded_columns + [self.pred_column]
        if self.proba_column:
            ex_cols.append(self.proba_column)
        return ex_cols
