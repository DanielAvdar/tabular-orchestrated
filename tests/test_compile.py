import pytest
from kfp import compiler

from tabular_orchestrated_kfp import (
    datasplitter,
    dcdatacomp,
    dctraintestcomp,
    evalmlpredict,
    evalmlsearch,
    evalmlselectpipeline,
    evaluatemljar,
    mljartraining,
)

components = [
    datasplitter,
    dcdatacomp,
    dctraintestcomp,
    evaluatemljar,
    mljartraining,
    evalmlsearch,
    evalmlselectpipeline,
    evalmlpredict,
]


@pytest.mark.parametrize("component", components)
def test_compile_core(tmp_files_folder, component):
    pipeline_path = tmp_files_folder / "comp.yaml"
    compiler.Compiler().compile(component, pipeline_path.as_posix())
