import warnings
import importlib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils.multiclass import type_of_target

from schemas import TaskType

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    task_type: TaskType,
    target_column: Optional[str],
    ordinal_columns: list[str] = [],
    nominal_columns: list[str] = [],
) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
    """
    Returns
    -------
    X_transformed : np.ndarray
    y             : np.ndarray | None   (None for clustering)
    preprocessor  : fitted ColumnTransformer  (saved alongside the model)
    """
    df = df.copy()
    df.dropna(how="all", inplace=True)  # remove rows that are entirely NaN

    if task_type in (TaskType.classification, TaskType.regression):
        return _supervised_preprocessing(df, task_type, target_column, ordinal_columns, nominal_columns)
    else:
        return _clustering_preprocessing(df, ordinal_columns, nominal_columns)


# ──────────────────────────────────────────────────────────────────────────────
# Supervised preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _supervised_preprocessing(df, task_type, target_column, ordinal_columns, nominal_columns):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    # Encode target for classification
    if task_type == TaskType.classification:
        if y.dtype == object or str(y.dtype) == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_column)
        else:
            y = y.astype(int)
    else:
        y = y.astype(float)

    # Drop high-cardinality text columns (e.g. free-text, IDs) that won't encode well
    for col in X.select_dtypes(include=object).columns:
        if X[col].nunique() > 50:
            X.drop(columns=[col], inplace=True)

    preprocessor = _build_column_transformer(X, ordinal_columns, nominal_columns)
    X_transformed = preprocessor.fit_transform(X)

    # Handle class imbalance (classification only)
    if task_type == TaskType.classification:
        X_transformed, y = _handle_imbalance(X_transformed, y)

    return X_transformed, y.values, preprocessor


# ──────────────────────────────────────────────────────────────────────────────
# Clustering preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _clustering_preprocessing(df, ordinal_columns, nominal_columns):
    X = df.copy()

    # Drop obviously non-informative columns (high-cardinality strings)
    for col in X.select_dtypes(include=object).columns:
        if X[col].nunique() > 50:
            X.drop(columns=[col], inplace=True)

    preprocessor = _build_column_transformer(X, ordinal_columns, nominal_columns)
    X_transformed = preprocessor.fit_transform(X)

    return X_transformed, None, preprocessor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_column_transformer(X, ordinal_cols, nominal_cols):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    transformers = []

    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if ordinal_cols:
        ord_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("ord", ord_pipe, ordinal_cols))

    if nominal_cols:
        nom_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("nom", nom_pipe, nominal_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _handle_imbalance(X: np.ndarray, y: pd.Series):
    """Apply SMOTE or RandomOverSampler if the dataset is imbalanced."""
    ttype = type_of_target(y)
    if ttype not in ("binary", "multiclass"):
        return X, y

    counts     = y.value_counts()
    min_count  = counts.min()
    max_count  = counts.max()
    ratio      = min_count / max_count

    if ratio >= 0.5:          # already balanced enough
        return X, y

    try:
        oversampling = importlib.import_module("imblearn.over_sampling")
        SMOTE = oversampling.SMOTE
        RandomOverSampler = oversampling.RandomOverSampler

        if min_count >= 6:
            sampler = SMOTE(random_state=42)
        else:
            sampler = RandomOverSampler(random_state=42)

        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, pd.Series(y_res)

    except Exception:
        return X, y
