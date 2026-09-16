
from typing import Any

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1-score": f1_score(y, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probabilities),
    }


def build_model_pipelines(
    preprocessor: ColumnTransformer,
    random_state: int = 42,
    model_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Pipeline]:
    model_config = model_config or {}
    logistic_config = model_config.get("logistic_regression", {})
    knn_config = model_config.get("knn", {})
    tree_config = model_config.get("decision_tree", {})

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=logistic_config.get("max_iter", 2000),
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "K-Nearest Neighbours": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=knn_config.get("n_neighbors", 5)
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=random_state,
                        **tree_config,
                    ),
                ),
            ]
        ),
    }


def train_baseline_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    fitted_models = {}
    result_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        for dataset_name, X, y in (
            ("Training", X_train, y_train),
            ("Validation", X_validation, y_validation),
        ):
            result_rows.append(
                {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    **evaluate_model(model, X, y),
                }
            )

    return fitted_models, pd.DataFrame(result_rows)


def get_parameter_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"],
        },
        "K-Nearest Neighbours": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "Decision Tree": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [3, 5, 7, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__class_weight": [None, "balanced"],
        },
    }


def tune_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    random_state: int = 42,
    parameter_grids: dict[str, dict[str, list[Any]]] | None = None,
    cv_folds: int = 5,
    scoring: str = "f1",
    n_jobs: int = -1,
) -> tuple[dict[str, GridSearchCV], dict[str, Pipeline], pd.DataFrame]:
    
    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    parameter_grids = parameter_grids or get_parameter_grids()
    searches = {}
    tuned_models = {}
    result_rows = []

    for model_name, model in models.items():
        search = GridSearchCV(
            estimator=model,
            param_grid=parameter_grids[model_name],
            scoring=scoring,
            cv=cross_validation,
            n_jobs=n_jobs,
            return_train_score=True,
            refit=True,
        )
        search.fit(X_train, y_train)
        searches[model_name] = search
        tuned_models[model_name] = search.best_estimator_

        metrics = evaluate_model(
            search.best_estimator_, X_validation, y_validation
        )
        result_rows.append(
            {
                "Model": model_name,
                "Best CV F1-score": search.best_score_,
                "Mean CV Train F1-score": search.cv_results_[
                    "mean_train_score"
                ][search.best_index_],
                **{f"Validation {key}": value for key, value in metrics.items()},
            }
        )

    results = pd.DataFrame(result_rows).sort_values(
        "Validation F1-score", ascending=False
    )
    return searches, tuned_models, results.reset_index(drop=True)


def compare_resampling_strategies(
    tuned_knn_pipeline: Pipeline,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    random_state: int = 42,
) -> tuple[dict[str, ImbalancedPipeline], pd.DataFrame]:
    strategies = {
        "No Resampling": "passthrough",
        "Random Oversampling": RandomOverSampler(random_state=random_state),
        "Random Undersampling": RandomUnderSampler(random_state=random_state),
    }
    classifier = tuned_knn_pipeline.named_steps["classifier"]
    fitted_models = {}
    result_rows = []

    for strategy_name, sampler in strategies.items():
        model = ImbalancedPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("sampler", sampler),
                ("classifier", clone(classifier)),
            ]
        )
        model.fit(X_train, y_train)
        fitted_models[strategy_name] = model
        result_rows.append(
            {
                "Resampling Strategy": strategy_name,
                **{
                    f"Validation {key}": value
                    for key, value in evaluate_model(
                        model, X_validation, y_validation
                    ).items()
                },
            }
        )

    results = pd.DataFrame(result_rows).sort_values(
        "Validation F1-score", ascending=False
    )
    return fitted_models, results.reset_index(drop=True)


def build_final_pipeline(
    tuned_knn_pipeline: Pipeline,
    preprocessor: ColumnTransformer,
    resampling_strategy: str,
    random_state: int = 42,
) -> ImbalancedPipeline:
    if resampling_strategy == "Random Oversampling":
        sampler: Any = RandomOverSampler(random_state=random_state)
    elif resampling_strategy == "Random Undersampling":
        sampler = RandomUnderSampler(random_state=random_state)
    elif resampling_strategy == "No Resampling":
        sampler = "passthrough"
    else:
        raise ValueError(f"Unknown resampling strategy: {resampling_strategy}")

    return ImbalancedPipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("sampler", sampler),
            (
                "classifier",
                clone(tuned_knn_pipeline.named_steps["classifier"]),
            ),
        ]
    )
