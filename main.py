
import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data_preparation import prepare_data
from src.model_training import (
    build_final_pipeline,
    build_model_pipelines,
    compare_resampling_strategies,
    evaluate_model,
    train_baseline_models,
    tune_models,
)


DEFAULT_CONFIG_PATH = Path("src/config.yaml")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the patient survival classifier."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main(config_path: Path) -> None:
    config = load_config(config_path)
    data_config = config["data"]
    split_config = config["split"]
    cv_config = config["cross_validation"]
    random_state = split_config["random_state"]

    cleaned_data, splits, preprocessor = prepare_data(
        data_config["file_path"],
        validation_size=split_config["validation_size"],
        test_size=split_config["test_size"],
        random_state=random_state,
    )

    print(f"Cleaned dataset shape: {cleaned_data.shape}")
    print("Training rows:", len(splits.X_train))
    print("Validation rows:", len(splits.X_validation))
    print("Test rows:", len(splits.X_test))

    model_pipelines = build_model_pipelines(
        preprocessor,
        random_state=random_state,
        model_config=config.get("models"),
    )
    _, baseline_results = train_baseline_models(
        model_pipelines,
        splits.X_train,
        splits.y_train,
        splits.X_validation,
        splits.y_validation,
    )
    print("\nBaseline results")
    print(baseline_results.round(4).to_string(index=False))

    searches, tuned_models, tuning_results = tune_models(
        model_pipelines,
        splits.X_train,
        splits.y_train,
        splits.X_validation,
        splits.y_validation,
        random_state=random_state,
        parameter_grids=config.get("hyperparameters"),
        cv_folds=cv_config["folds"],
        scoring=cv_config["scoring"],
        n_jobs=cv_config["n_jobs"],
    )
    print("\nBest hyperparameters")
    for model_name, search in searches.items():
        print(f"{model_name}: {search.best_params_}")
    print("\nTuned validation results")
    print(tuning_results.round(4).to_string(index=False))

    _, resampling_results = compare_resampling_strategies(
        tuned_models["K-Nearest Neighbours"],
        preprocessor,
        splits.X_train,
        splits.y_train,
        splits.X_validation,
        splits.y_validation,
        random_state=random_state,
    )
    print("\nKNN resampling comparison")
    print(resampling_results.round(4).to_string(index=False))

    selected_strategy = resampling_results.iloc[0][
        "Resampling Strategy"
    ]
    final_pipeline = build_final_pipeline(
        tuned_models["K-Nearest Neighbours"],
        preprocessor,
        selected_strategy,
        random_state=random_state,
    )

    X_development = pd.concat(
        [splits.X_train, splits.X_validation], ignore_index=True
    )
    y_development = pd.concat(
        [splits.y_train, splits.y_validation], ignore_index=True
    )
    final_pipeline.fit(X_development, y_development)

    test_metrics = evaluate_model(
        final_pipeline,
        splits.X_test,
        splits.y_test,
    )
    print(f"\nSelected resampling strategy: {selected_strategy}")
    print("Final test results")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments.config_path)
