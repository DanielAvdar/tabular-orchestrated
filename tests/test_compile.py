from tabular_kfp.pipeline import mljar_pipeline
from tabular_kfp.pipeline_gcp import mljar_gcp_pipeline

from kfp import compiler


def test_compile_core(tmp_files_folder):
    pipeline_path = tmp_files_folder / "mljar_pipeline.yaml"
    compiler.Compiler().compile(mljar_pipeline, pipeline_path.as_posix())


def test_compile_gcp(tmp_files_folder):
    pipeline_path = tmp_files_folder / "mljar_gcp_pipeline.yaml"
    compiler.Compiler().compile(mljar_gcp_pipeline, pipeline_path.as_posix())
