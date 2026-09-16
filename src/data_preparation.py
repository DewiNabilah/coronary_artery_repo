
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Survive"
REQUIRED_COLUMNS = {
    "ID",
    "Survive",
    "Gender",
    "Smoke",
    "Diabetes",
    "Age",
    "Ejection Fraction",
    "Sodium",
    "Creatinine",
    "Platelets",
    "Creatine phosphokinase",
    "Blood Pressure",
    "Hemoglobin",
    "Height",
    "Weight",
    "Favorite color",
}


@dataclass
class DatasetSplits:

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Read the coronary artery disease CSV file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = pd.read_csv(file_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()

    # Assumption: Yes/1 means survived and No/0 means did not survive.
    target_mapping = {"0": 0, "No": 0, "1": 1, "Yes": 1}
    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN].astype(str).str.strip().map(target_mapping)
    )
    if cleaned[TARGET_COLUMN].isna().any():
        invalid = data.loc[
            cleaned[TARGET_COLUMN].isna(), TARGET_COLUMN
        ].unique()
        raise ValueError(f"Unrecognised target labels: {invalid.tolist()}")
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    cleaned["Gender"] = cleaned["Gender"].astype(str).str.strip().str.title()
    cleaned["Smoke"] = cleaned["Smoke"].astype(str).str.strip().str.title()
    cleaned["Diabetes"] = cleaned["Diabetes"].astype(str).str.strip()
    cleaned["Ejection Fraction"] = (
        cleaned["Ejection Fraction"]
        .astype(str)
        .str.strip()
        .replace({"L": "Low", "N": "Normal"})
    )

    # Negative ages are treated as synthetic sign errors.
    cleaned["Age"] = cleaned["Age"].abs()

    # ID is an identifier; favourite colour is an unsupported synthetic feature.
    cleaned = cleaned.drop(columns=["ID", "Favorite color"])
    return cleaned


def split_data(
    data: pd.DataFrame,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplits:

    if validation_size <= 0 or test_size <= 0:
        raise ValueError("Validation and test sizes must both be positive.")
    temporary_size = validation_size + test_size
    if temporary_size >= 1:
        raise ValueError("Validation and test sizes must sum to less than 1.")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_temporary, y_train, y_temporary = train_test_split(
        X,
        y,
        test_size=temporary_size,
        random_state=random_state,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temporary,
        y_temporary,
        test_size=test_size / temporary_size,
        random_state=random_state,
        stratify=y_temporary,
    )

    return DatasetSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing steps using feature types from training data only."""
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def prepare_data(
    file_path: str | Path,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DatasetSplits, ColumnTransformer]:

    cleaned_data = clean_data(load_data(file_path))
    splits = split_data(
        cleaned_data,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = build_preprocessor(splits.X_train)
    return cleaned_data, splits, preprocessor
