# flake8: noqa: F403, F405, B006
from importlib.metadata import version
from typing import *

from kfp.dsl import *


@component(
    base_image="python:3.11",
    packages_to_install=[
        "scikit-learn",
        "pandas-pyarrow",
        f"tabular-orchestrated=={version('tabular-orchestrated')}",
    ],
)
def datasplitter(
    dataset: Input[Dataset] = None,
    train_dataset: Output[Dataset] = None,
    test_dataset: Output[Dataset] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
):
    from tabular_orchestrated.components import DataSplitter

    comp = DataSplitter(
        dataset=dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[
        "mljar-supervised==1.1.6",
        "pandas-pyarrow",
        f"tabular-orchestrated=={version('tabular-orchestrated')}",
    ],
)
def mljartraining(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    dataset: Input[Dataset] = None,
    model: Output[Model] = None,
    mljar_automl_params: Dict = {
        "total_time_limit": 43200,
        "algorithms": ["Linear", "Random Forest", "Extra Trees", "LightGBM", "Xgboost", "CatBoost"],
        "train_ensemble": True,
        "eval_metric": "auto",
        "validation_strategy": {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True},
        "explain_level": 2,
    },
):
    from tabular_orchestrated.mljar import MLJARTraining

    comp = MLJARTraining(
        exclude_columns=exclude_columns,
        target_column=target_column,
        dataset=dataset,
        model=model,
        mljar_automl_params=mljar_automl_params,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[
        "mljar-supervised==1.1.6",
        "pandas-pyarrow",
        f"tabular-orchestrated=={version('tabular-orchestrated')}",
    ],
)
def evaluatemljar(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    test_dataset: Input[Dataset] = None,
    model: Input[Model] = None,
    metrics: Output[Metrics] = None,
    report: Output[HTML] = None,
):
    from tabular_orchestrated.mljar import EvaluateMLJAR

    comp = EvaluateMLJAR(
        exclude_columns=exclude_columns,
        target_column=target_column,
        test_dataset=test_dataset,
        model=model,
        metrics=metrics,
        report=report,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[
        "deepchecks==0.18.1",
        "mljar-supervised==1.1.6",
        "pandas-pyarrow",
        f"tabular-orchestrated=={version('tabular-orchestrated')}",
    ],
)
def deepchecksfullcomp(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    report: Output[HTML] = None,
    failed_checks: Output[Metrics] = None,
    train_dataset: Input[Dataset] = None,
    test_dataset: Input[Dataset] = None,
    model: Input[Model] = None,
):
    from tabular_orchestrated.deepchecks import DeepChecksFullComp

    comp = DeepChecksFullComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
    comp.execute()


# shuffle: shuffle
# metrics: metrics
# random_state: random_state
# target_column: target_column
# mljar_automl_params: mljar_automl_params
# report: report
# failed_checks: failed_checks
# dataset: dataset
# test_dataset: test_dataset
# test_size: test_size
# train_dataset: train_dataset
# model: model
# exclude_columns: exclude_columns
