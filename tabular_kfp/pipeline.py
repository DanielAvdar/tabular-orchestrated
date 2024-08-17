# flake8: noqa: F403, F405, B006


from kfp.dsl import Dataset, Input, Model, pipeline


@pipeline(name="MLJAR Pipeline", description="A pipeline that performs data splitting, MLJAR training, and evaluation.")
def mljar_pipeline(
    dataset: Input[Dataset],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    mljar_automl_params: dict = {
        "total_time_limit": 43200,
        "algorithms": ["LightGBM", "Xgboost", "CatBoost"],
        "train_ensemble": True,
        "eval_metric": "auto",
        "validation_strategy": {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True},
        "explain_level": 2,
    },
    exclude_columns: list = [],
) -> Model:
    from tabular_kfp.components import datasplitter, evaluatemljar, mljardcfullcomp, mljartraining

    datasplitter_comp = datasplitter(
        dataset=dataset,
        test_size=test_size,
        random_state=random_state,
    ).set_display_name("Data Splitter")

    mljartraining_comp = mljartraining(
        exclude_columns=exclude_columns,
        target_column=target_column,
        dataset=datasplitter_comp.outputs["train_dataset"],
        mljar_automl_params=mljar_automl_params,
    ).set_display_name("MLJAR Training")

    evaluatemljar(
        exclude_columns=exclude_columns,
        target_column=target_column,
        test_dataset=datasplitter_comp.outputs["test_dataset"],
        model=mljartraining_comp.outputs["model"],
    ).set_display_name("Evaluate MLJAR")

    mljardcfullcomp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        train_dataset=datasplitter_comp.outputs["train_dataset"],
        test_dataset=datasplitter_comp.outputs["test_dataset"],
        model=mljartraining_comp.outputs["model"],
    ).set_display_name("DeepChecks General")
    return mljartraining_comp.outputs["model"]
