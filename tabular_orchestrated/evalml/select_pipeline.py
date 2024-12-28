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
        self.model.metadata["problem_type"] = str(automl.problem_type)
