from pathlib import Path

from tabular_kfp.parse_components import parse_components


def test_component_parser(test_directory: Path) -> None:
    path = test_directory.parent / "tabular_kfp" / "components.py"
    parse_components(path.as_posix())
