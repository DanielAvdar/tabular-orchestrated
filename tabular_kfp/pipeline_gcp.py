# flake8: noqa: F403, F405, B006

from tabular_kfp.gcp_comp import bq_read
from tabular_kfp.pipeline import mljar_pipeline

from kfp.dsl import pipeline


@pipeline(name="MLJAR Pipeline", description="A pipeline that performs data splitting, MLJAR training, and evaluation.")
def mljar_gcp_pipeline(
    query: str,
    target_column: str,
    project_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
    mljar_automl_params: dict = {
        "total_time_limit": 43200,
        "algorithms": ["Linear", "Random Forest", "Extra Trees", "LightGBM", "Xgboost", "CatBoost"],
        "train_ensemble": True,
        "eval_metric": "auto",
        "validation_strategy": {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True},
        "explain_level": 2,
    },
    exclude_columns: list = [],
):
    data = bq_read(
        query=query,
        project_id=project_id,
    ).set_display_name("Read BigQuery")
    mljar_pipeline(
        dataset=data.outputs["dataset"],
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        mljar_automl_params=mljar_automl_params,
        exclude_columns=exclude_columns,
    )
