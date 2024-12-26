from kfp.dsl import Dataset, Output, component


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
    adf = convert_to_pyarrow(df)
    adf.to_parquet(dataset.path + ".parquet", engine="pyarrow")
    dataset.metadata["original dtypes"] = [repr(f) for f in set(df.dtypes)]
    dataset.metadata["original dtypes str"] = [str(f) for f in set(df.dtypes)]
    dataset.metadata["pyarrow dtypes"] = [repr(f) for f in set(adf.dtypes)]
    dataset.metadata["pyarrow dtypes str"] = [str(f) for f in set(adf.dtypes)]
    dataset.metadata["number of rows"] = len(adf)
    dataset.metadata["number of columns"] = len(adf.columns)
