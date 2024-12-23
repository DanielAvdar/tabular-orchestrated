# from kfp import compiler
#
# from tabular_kfp.pipeline import mljar_pipeline
# from tabular_kfp.pipeline_gcp import mljar_gcp_pipeline
#
#
# def test_compile_core(tmp_files_folder):
#     pipeline_path = tmp_files_folder / "mljar_pipeline.yaml"
#     compiler.Compiler().compile(mljar_pipeline, pipeline_path.as_posix())
#
#
# def test_compile_gcp(tmp_files_folder):
#     pipeline_path = tmp_files_folder / "mljar_gcp_pipeline.yaml"
#     compiler.Compiler().compile(mljar_gcp_pipeline, pipeline_path.as_posix())
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
    mljardcmodelcomp,
    mljartraining,
)

components = [
    datasplitter,
    dcdatacomp,
    dctraintestcomp,
    evaluatemljar,
    mljardcmodelcomp,
    mljartraining,
    evalmlsearch,
    evalmlselectpipeline,
    evalmlpredict,
]


@pytest.mark.parametrize("component", components)
def test_compile_core(tmp_files_folder, component):
    pipeline_path = tmp_files_folder / "comp.yaml"
    compiler.Compiler().compile(component, pipeline_path.as_posix())
