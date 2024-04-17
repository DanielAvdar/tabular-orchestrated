from pathlib import Path

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DeepChecksFullComp
from tabular_orchestrated.mljar import EvaluateMLJAR, MLJARTraining

from ml_orchestrator import ComponentParser


def parse_components(file_path: str) -> None:
    parser = ComponentParser()
    comp_list = [DataSplitter(), MLJARTraining(), EvaluateMLJAR(), DeepChecksFullComp()]
    parser.parse_components_to_file(comp_list, file_path)


def test_component_parser(test_directory: Path) -> None:
    path = test_directory.parent / "tabular_kfp" / "components.py"
    parse_components(path.as_posix())
