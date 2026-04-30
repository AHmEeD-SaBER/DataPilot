import warnings
import numpy as np
from typing import Any, Optional, Tuple

from schemas import TaskType

warnings.filterwarnings("ignore")


_N_ITER = 20
_CV_FOLDS = 5


def train_and_evaluate(
    X: np.ndarray,
    y: Optional[np.ndarray],
    task_type: TaskType,
) -> Tuple[Any, str, dict, dict]:
    """
    Returns
    -------
    best_model   : fitted estimator (refitted on full data)
    best_name    : str  – name of the winning algorithm
    comparison   : dict – lightweight metrics for every candidate
    best_metrics : dict – full metrics for the winning model
    """
    if task_type == TaskType.classification:
        return _classification(X, y)
    if task_type == TaskType.regression:
        return _regression(X, y)
    return _clustering(X)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _tune(estimator, param_dist: dict, X_train, y_train, scoring: str):
    from sklearn.model_selection import RandomizedSearchCV

    n_iter = min(_N_ITER, max(1, len(X_train) // _CV_FOLDS))
    cv = min(_CV_FOLDS, len(X_train) // 2)

    try:
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception:
        estimator.fit(X_train, y_train)
        return estimator


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────


def _classification(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    unique_classes = np.unique(y)
    avg = "binary" if len(unique_classes) == 2 else "weighted"

    stratify = y if np.bincount(y).min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # ── Candidate models + hyperparameter search spaces ───────────────────────
    candidates = {
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False],
            },
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
            {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "newton-cg"],
            },
        ),
        "GradientBoostingClassifier": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 1.0],
                "min_samples_split": [2, 5, 10],
            },
        ),
    }

    comparison: dict = {}
    best_name: str = ""
    best_f1: float = -1.0
    best_model = None
    best_metrics = None

    for name, (estimator, param_dist) in candidates.items():
        tuned = _tune(estimator, param_dist, X_train, y_train, scoring="f1_weighted")
        y_pred = tuned.predict(X_test)

        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        prec = round(
            float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4
        )
        rec = round(
            float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4
        )
        f1 = round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4)

        comparison[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "best_params": (tuned.get_params() if hasattr(tuned, "get_params") else {}),
        }

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = tuned
            cm = confusion_matrix(y_test, y_pred)
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm.tolist(),
                "best_params": tuned.get_params()
                if hasattr(tuned, "get_params")
                else {},
            }

    # Refit winner on full data
    best_model.fit(X, y)
    return best_model, best_name, comparison, best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Regression
# ──────────────────────────────────────────────────────────────────────────────


def _regression(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        ),
        "RidgeRegression": (
            Ridge(),
            {
                "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "solver": ["auto", "svd", "cholesky", "lsqr"],
            },
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 1.0],
                "min_samples_split": [2, 5, 10],
            },
        ),
    }

    comparison: dict = {}
    best_name: str = ""
    best_mae: float = float("inf")
    best_model = None
    best_metrics = None

    for name, (estimator, param_dist) in candidates.items():
        tuned = _tune(
            estimator, param_dist, X_train, y_train, scoring="neg_mean_absolute_error"
        )
        y_pred = tuned.predict(X_test)

        mae = round(float(mean_absolute_error(y_test, y_pred)), 4)
        mse = round(float(mean_squared_error(y_test, y_pred)), 4)
        r2 = round(float(r2_score(y_test, y_pred)), 4)

        comparison[name] = {
            "mae": mae,
            "mse": mse,
            "r2_score": r2,
            "best_params": tuned.get_params() if hasattr(tuned, "get_params") else {},
        }

        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = tuned
            best_metrics = {
                "mae": mae,
                "mse": mse,
                "r2_score": r2,
                "best_params": tuned.get_params()
                if hasattr(tuned, "get_params")
                else {},
            }

    best_model.fit(X, y)
    return best_model, best_name, comparison, best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────────────


def _clustering(X):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    best_k = _find_best_k(X)

    candidates = {
        "KMeans": KMeans(n_clusters=best_k, random_state=42, n_init="auto"),
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=best_k),
    }

    comparison: dict = {}
    best_name: str = ""
    best_sil: float = -1.0
    best_model = None
    best_metrics = None

    sample_size = min(5_000, len(X))

    for name, model in candidates.items():
        labels = model.fit_predict(X)
        unique, counts = np.unique(labels, return_counts=True)
        n_unique = len(unique)

        sil = (
            round(float(silhouette_score(X, labels, sample_size=sample_size)), 4)
            if n_unique > 1
            else -1.0
        )

        comparison[name] = {"silhouette_score": sil}

        if sil > best_sil:
            best_sil = sil
            best_name = name
            best_model = model
            best_metrics = {
                "silhouette_score": sil,
                "n_clusters": int(n_unique),
                "cluster_sizes": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            }

    return best_model, best_name, comparison, best_metrics


def _find_best_k(X: np.ndarray, max_k: int = 8) -> int:
    """
    Elbow-aware k selection: tries k in [2, max_k] and picks the value
    with the highest silhouette score, falling back to k=2 on errors.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    cap = min(max_k, len(X) - 1)
    best_k = 2
    best_sc = -1.0

    for k in range(2, cap + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            sc = silhouette_score(X, labels, sample_size=min(3_000, len(X)))
            if sc > best_sc:
                best_sc = sc
                best_k = k
        except Exception:
            continue

    return best_k
