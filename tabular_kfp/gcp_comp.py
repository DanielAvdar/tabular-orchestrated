# flake8: noqa: F403, F405, B006
from typing import *

from kfp.dsl import *


@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas-pyarrow[bigquery]",
    ],
)
def bq_read(
    query: str,
    project_id: str,
    dataset: Output[Dataset],
):
    import pandas as pd
    from pandas_pyarrow import convert_to_pyarrow

    df = pd.read_gbq(query=query, project_id=project_id)
    df = convert_to_pyarrow(df)
    df.to_parquet(dataset.path + ".parquet", engine="pyarrow")
