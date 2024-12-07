from pathlib import Path

from tabular_kfp.parse_components import parse_components

if __name__ == "__main__":
    # file_path = Path(__file__).parent / "components.py"
    file_path = Path(__file__).parent / "tabular_orchestrated_kfp" / "__init__.py"
    parse_components(file_path.as_posix())
