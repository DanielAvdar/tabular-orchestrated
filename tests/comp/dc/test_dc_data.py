from pathlib import Path

import ml_orchestrator.artifacts as artifacts
import numpy as np
import pandas as pd
import pytest

from tabular_orchestrated.dc import DCDataComp


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({
            "target": [1, 2, 3, 4],
            "B": [5, 6, 7, 8],
            "C": [9, 10, 11, 12],
        }),
        pd.DataFrame({
            "target": [0.1, -np.inf, 0, 0],
            "B": [5, 6, 7, 8],
            "C": [0.1, -np.Inf, 0, 0],
        }),
    ],
)
def test_dc_data_op(data: pd.DataFrame, tmp_files_folder, model_params) -> None:
    def func(x):
        return (tmp_files_folder / x).as_posix()

    dataset = artifacts.Dataset(uri=func("dc_data_check"))
    DCDataComp.save_df(data, dataset)
    deepchecks_data_op = DCDataComp(
        dataset=dataset,
        report=artifacts.HTML(uri=func("dc_data")),
        failed_checks=artifacts.Metrics(uri=func("failed_checks_data")),
        **model_params,
    )
    deepchecks_data_op.execute()
    assert Path(deepchecks_data_op.report.uri).exists()
