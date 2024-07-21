from pathlib import Path

from tabular_orchestrated.components import DataSplitter
from tabular_orchestrated.deepchecks import DCFullComp
from tabular_orchestrated.mljar import EvaluateMLJAR, MLJARTraining

from ml_orchestrator import ComponentParser


def parse_components(file_path: str) -> None:
    parser = ComponentParser()
    comp_list = [DataSplitter(), MLJARTraining(), EvaluateMLJAR(), DCFullComp()]
    parser.parse_components_to_file(comp_list, file_path)


if __name__ == "__main__":
    file_path = Path(__file__).parent / "components.py"
    parse_components(file_path.as_posix())
