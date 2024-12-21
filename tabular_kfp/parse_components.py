from ml_orchestrator import ComponentParser

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.dc.dc_data import DCDataComp, DCTrainTestComp
from tabular_orchestrated.dc.dc_model_v2 import DCModelCompV2
from tabular_orchestrated.mljar.mljar import EvaluateMLJAR, MLJARTraining
from tabular_orchestrated.mljar.mljar_deepchecks import MljarDCFullComp, MljarDCModelComp


def parse_components(file_path: str) -> None:
    parser = ComponentParser()
    comp_list = [
        MLJARTraining(),
        DataSplitter(),
        EvaluateMLJAR(),
        DCTrainTestComp(),
        DCDataComp(),
        DCModelCompV2(),
        # DCModelComp(),
        # DCFullComp(),
        MljarDCModelComp(),
        MljarDCFullComp(),
    ]
    parser.parse_components_to_file(comp_list, file_path)
