from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DCDataComp, DCFullComp, DCModelComp, DCTrainTestComp
from tabular_orchestrated.mljar import EvaluateMLJAR, MLJARTraining

from ml_orchestrator import ComponentParser


def parse_components(file_path: str) -> None:
    parser = ComponentParser()
    comp_list = [
        DataSplitter(),
        MLJARTraining(),
        EvaluateMLJAR(),
        DCFullComp(),
        DCTrainTestComp(),
        DCModelComp(),
        DCDataComp(),
    ]
    parser.parse_components_to_file(comp_list, file_path)
