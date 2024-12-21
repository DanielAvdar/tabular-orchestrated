# flake8: noqa: F403, F405, B006
from importlib.metadata import version
from typing import *

from kfp.dsl import *


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[mljar]=={version('tabular-orchestrated')}"]
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
    from tabular_orchestrated.mljar.mljar import MLJARTraining

    comp = MLJARTraining(
        exclude_columns=exclude_columns,
        target_column=target_column,
        dataset=dataset,
        model=model,
        mljar_automl_params=mljar_automl_params,
    )
    comp.execute()


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[spliter]=={version('tabular-orchestrated')}"]
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
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[mljar]=={version('tabular-orchestrated')}"]
)
def evaluatemljar(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    test_dataset: Input[Dataset] = None,
    model: Input[Model] = None,
    metrics: Output[Metrics] = None,
    report: Output[HTML] = None,
):
    from tabular_orchestrated.mljar.mljar import EvaluateMLJAR

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
    packages_to_install=[f"tabular-orchestrated[deepchecks]=={version('tabular-orchestrated')}"],
)
def dctraintestcomp(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    report: Output[HTML] = None,
    failed_checks: Output[Metrics] = None,
    train_dataset: Input[Dataset] = None,
    test_dataset: Input[Dataset] = None,
):
    from tabular_orchestrated.dc.dc_data import DCTrainTestComp

    comp = DCTrainTestComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[f"tabular-orchestrated[deepchecks]=={version('tabular-orchestrated')}"],
)
def dcdatacomp(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    report: Output[HTML] = None,
    failed_checks: Output[Metrics] = None,
    dataset: Input[Dataset] = None,
):
    from tabular_orchestrated.dc.dc_data import DCDataComp

    comp = DCDataComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        dataset=dataset,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[f"tabular-orchestrated[deepchecks]=={version('tabular-orchestrated')}"],
)
def dcmodelcompv2(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    report: Output[HTML] = None,
    failed_checks: Output[Metrics] = None,
    train_dataset: Input[Dataset] = None,
    test_dataset: Input[Dataset] = None,
    pred_column: str = "pred_column",
    proba_column: str = "None",
):
    from tabular_orchestrated.dc.dc_model_v2 import DCModelCompV2

    comp = DCModelCompV2(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pred_column=pred_column,
        proba_column=proba_column,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[f"tabular-orchestrated[deepchecks,mljar]=={version('tabular-orchestrated')}"],
)
def mljardcmodelcomp(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    report: Output[HTML] = None,
    failed_checks: Output[Metrics] = None,
    train_dataset: Input[Dataset] = None,
    test_dataset: Input[Dataset] = None,
    model: Input[Model] = None,
):
    from tabular_orchestrated.mljar.mljar_deepchecks import MljarDCModelComp

    comp = MljarDCModelComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[f"tabular-orchestrated[deepchecks,mljar]=={version('tabular-orchestrated')}"],
)
def mljardcfullcomp(
    exclude_columns: List[str] = [],
    target_column: str = "target",
    report: Output[HTML] = None,
    failed_checks: Output[Metrics] = None,
    train_dataset: Input[Dataset] = None,
    test_dataset: Input[Dataset] = None,
    model: Input[Model] = None,
):
    from tabular_orchestrated.mljar.mljar_deepchecks import MljarDCFullComp

    comp = MljarDCFullComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
    comp.execute()
