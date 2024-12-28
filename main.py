from pathlib import Path

from ml_orchestrator import ComponentParser

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.dc import DCDataComp, DCModelComp, DCModelCompV2, DCTrainTestComp
from tabular_orchestrated.evalml import (
    EvalMLAnalysis,
    EvalMLAnalysisV2,
    EvalMLFineTune,
    EvalMLPredict,
    EvalMLSearch,
    EvalMLSelectPipeline,
)
from tabular_orchestrated.mljar.mljar import EvaluateMLJAR, MLJARTraining


def parse_components(file_path: str) -> None:
    parser = ComponentParser()
    comp_list = [
        MLJARTraining,
        DataSplitter,
        EvaluateMLJAR,
        DCTrainTestComp,
        DCDataComp,
        DCModelCompV2,
        DCModelComp,
        EvalMLPredict,
        EvalMLSearch,
        EvalMLSelectPipeline,
        EvalMLAnalysis,
        EvalMLAnalysisV2,
        EvalMLFineTune,
    ]
    parser.parse_components_to_file(comp_list, file_path)


if __name__ == "__main__":
    # file_path = Path(__file__).parent / "components.py"
    file_path = Path(__file__).parent / "tabular_orchestrated_kfp" / "__init__.py"
    parse_components(file_path.as_posix())
