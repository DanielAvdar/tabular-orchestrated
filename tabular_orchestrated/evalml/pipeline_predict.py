import dataclasses

from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output

from tabular_orchestrated.evalml.evalml import EvalMLComp


@dataclasses.dataclass
class EvalMLPredict(EvalMLComp):
    model: Input[artifacts.Model]
    test_dataset: Input[artifacts.Dataset]
    predictions: Output[artifacts.Dataset]
    pred_column: str = "pred_column"
    proba_column_prefix: str = "proba_column"

    def execute(self) -> None:
        model = self.load_model(self.model)
        test_df = self.load_df(self.test_dataset)
        predictions = model.predict(test_df[self.model_columns(test_df)])
        if self.model.metadata["problem_type"] != "regression":
            proba = model.predict_proba(test_df[self.model_columns(test_df)])
            test_df[[f"{self.proba_column_prefix}_{col}" for col in proba.columns]] = proba
        test_df[self.pred_column] = predictions
        self.save_df(test_df, self.predictions)
