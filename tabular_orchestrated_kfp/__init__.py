# flake8: noqa: F403, F405, B006
from importlib.metadata import version
from typing import *

from kfp.dsl import *


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[mljar]=={version('tabular-orchestrated')}"]
)
def mljartraining(
    exclude_columns: List[str],
    target_column: str,
    dataset: Input[Dataset],
    model: Output[Model],
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
    exclude_columns: List[str],
    target_column: str,
    test_dataset: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
    report: Output[HTML],
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
    exclude_columns: List[str],
    target_column: str,
    report: Output[HTML],
    failed_checks: Output[Metrics],
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    as_widget: bool = True,
):
    from tabular_orchestrated.dc.dc_data import DCTrainTestComp

    comp = DCTrainTestComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        as_widget=as_widget,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[f"tabular-orchestrated[deepchecks]=={version('tabular-orchestrated')}"],
)
def dcdatacomp(
    exclude_columns: List[str],
    target_column: str,
    report: Output[HTML],
    failed_checks: Output[Metrics],
    dataset: Input[Dataset],
    as_widget: bool = True,
):
    from tabular_orchestrated.dc.dc_data import DCDataComp

    comp = DCDataComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        dataset=dataset,
        as_widget=as_widget,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[f"tabular-orchestrated[deepchecks]=={version('tabular-orchestrated')}"],
)
def dcmodelcompv2(
    exclude_columns: List[str],
    target_column: str,
    report: Output[HTML],
    failed_checks: Output[Metrics],
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    as_widget: bool = True,
    pred_column: str = "pred_column",
    proba_column: str = "proba_column",
):
    from tabular_orchestrated.dc.dc_model_v2 import DCModelCompV2

    comp = DCModelCompV2(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        as_widget=as_widget,
        pred_column=pred_column,
        proba_column=proba_column,
    )
    comp.execute()


@component(
    base_image="python:3.11",
    packages_to_install=[f"tabular-orchestrated[deepchecks,evalml]=={version('tabular-orchestrated')}"],
)
def dcmodelcomp(
    exclude_columns: List[str],
    target_column: str,
    report: Output[HTML],
    failed_checks: Output[Metrics],
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    as_widget: bool = True,
    model: Input[Model] = None,
):
    from tabular_orchestrated.dc.dc_model import DCModelComp

    comp = DCModelComp(
        exclude_columns=exclude_columns,
        target_column=target_column,
        report=report,
        failed_checks=failed_checks,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        as_widget=as_widget,
        model=model,
    )
    comp.execute()


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[evalml]=={version('tabular-orchestrated')}"]
)
def evalmlpredict(
    exclude_columns: List[str],
    target_column: str,
    model: Input[Model],
    test_dataset: Input[Dataset],
    predictions: Output[Dataset],
    pred_column: str = "pred_column",
    proba_column_prefix: str = "proba_column",
):
    from tabular_orchestrated.evalml.pipeline_predict import EvalMLPredict

    comp = EvalMLPredict(
        exclude_columns=exclude_columns,
        target_column=target_column,
        model=model,
        test_dataset=test_dataset,
        predictions=predictions,
        pred_column=pred_column,
        proba_column_prefix=proba_column_prefix,
    )
    comp.execute()


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[evalml]=={version('tabular-orchestrated')}"]
)
def evalmlsearch(
    exclude_columns: List[str],
    target_column: str,
    dataset: Input[Dataset],
    automl: Output[Model],
    search_params: dict = {},
):
    from tabular_orchestrated.evalml.search import EvalMLSearch

    comp = EvalMLSearch(
        exclude_columns=exclude_columns,
        target_column=target_column,
        dataset=dataset,
        automl=automl,
        search_params=search_params,
    )
    comp.execute()


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[evalml]=={version('tabular-orchestrated')}"]
)
def evalmlselectpipeline(
    exclude_columns: List[str],
    target_column: str,
    automl: Input[Model] = None,
    model: Output[Model] = None,
    pipeline_id: int = -1,
):
    from tabular_orchestrated.evalml.select_pipeline import EvalMLSelectPipeline

    comp = EvalMLSelectPipeline(
        exclude_columns=exclude_columns,
        target_column=target_column,
        automl=automl,
        model=model,
        pipeline_id=pipeline_id,
    )
    comp.execute()


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[evalml]=={version('tabular-orchestrated')}"]
)
def evalmlanalysis(
    exclude_columns: List[str],
    target_column: str,
    model: Input[Model],
    test_dataset: Input[Dataset],
    analysis: Output[HTML],
    metrics: Output[Metrics],
):
    from tabular_orchestrated.evalml.analysis.analysis_comp import EvalMLAnalysis

    comp = EvalMLAnalysis(
        exclude_columns=exclude_columns,
        target_column=target_column,
        model=model,
        test_dataset=test_dataset,
        analysis=analysis,
        metrics=metrics,
    )
    comp.execute()


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[evalml]=={version('tabular-orchestrated')}"]
)
def evalmlanalysisv2(
    exclude_columns: List[str],
    target_column: str,
    predictions: Input[Dataset],
    analysis: Output[HTML],
    metrics: Output[Metrics],
    pred_column: str = "pred_column",
    proba_column_prefix: str = "proba_column",
):
    from tabular_orchestrated.evalml.analysis.analysis_comp_v2 import EvalMLAnalysisV2

    comp = EvalMLAnalysisV2(
        exclude_columns=exclude_columns,
        target_column=target_column,
        predictions=predictions,
        analysis=analysis,
        metrics=metrics,
        pred_column=pred_column,
        proba_column_prefix=proba_column_prefix,
    )
    comp.execute()


@component(
    base_image="python:3.11", packages_to_install=[f"tabular-orchestrated[evalml]=={version('tabular-orchestrated')}"]
)
def evalmlfinetune(
    exclude_columns: List[str],
    target_column: str,
    model: Input[Model],
    train_dataset: Input[Dataset],
    fine_tuned_model: Output[Model],
):
    from tabular_orchestrated.evalml.pipeline_predict import EvalMLFineTune

    comp = EvalMLFineTune(
        exclude_columns=exclude_columns,
        target_column=target_column,
        model=model,
        train_dataset=train_dataset,
        fine_tuned_model=fine_tuned_model,
    )
    comp.execute()
