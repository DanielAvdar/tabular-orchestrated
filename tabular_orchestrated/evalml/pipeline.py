import dataclasses

from ml_orchestrator import artifacts
from ml_orchestrator.artifacts import Input, Output

from tabular_orchestrated.evalml.evalml import EvalMLComp


@dataclasses.dataclass
class EvalMLSelectPipeline(EvalMLComp):
    automl: Input[artifacts.Model] = None
    model: Output[artifacts.Model] = None
    pipeline_id: int = -1

    def execute(self) -> None:
        automl = self.load_model(self.automl)
        pipeline = automl.get_pipeline(self.pipeline_id) if self.pipeline_id != -1 else automl.best_pipeline
        self.save_model(pipeline, self.model)


@dataclasses.dataclass
class EvalMLPredict(EvalMLComp):
    model: Input[artifacts.Model]
    test_dataset: Input[artifacts.Dataset]
    predictions: Output[artifacts.Dataset]
    pred_column: str = "pred_column"
    proba_column: str = None

    def execute(self) -> None:
        model = self.load_model(self.model)
        test_df = self.load_df(self.test_dataset)
        predictions = model.predict(test_df[test_df.columns.difference([self.target_column] + self.excluded_columns)])
        predictions[self.pred_column] = predictions
        if self.proba_column:
            proba = model.predict_proba(
                test_df[test_df.columns.difference([self.target_column] + self.excluded_columns)]
            )
            predictions[self.proba_column] = proba
        self.save_df(predictions, self.predictions)
