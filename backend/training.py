import warnings
import numpy as np
from typing import Any, Optional, Tuple

from schemas import TaskType

warnings.filterwarnings("ignore")



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
# Classification
# ──────────────────────────────────────────────────────────────────────────────

def _classification(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )

    unique_classes = np.unique(y)
    avg = "binary" if len(unique_classes) == 2 else "weighted"

    # Stratify only when possible (all classes need ≥2 samples)
    stratify = y if np.bincount(y).min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    candidates = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    comparison: dict  = {}
    best_name:  str   = ""
    best_f1:    float = -1.0
    best_model        = None
    best_metrics      = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = round(float(accuracy_score(y_test, y_pred)), 4)
        prec = round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4)
        rec  = round(float(recall_score(y_test,  y_pred, average=avg, zero_division=0)), 4)
        f1   = round(float(f1_score(y_test,      y_pred, average=avg, zero_division=0)), 4)

        comparison[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}

        if f1 > best_f1:
            best_f1    = f1
            best_name  = name
            best_model = model
            cm         = confusion_matrix(y_test, y_pred)
            best_metrics = {
                "accuracy":         acc,
                "precision":        prec,
                "recall":           rec,
                "f1_score":         f1,
                "confusion_matrix": cm.tolist(),
            }

    best_model.fit(X, y)
    return best_model, best_name, comparison, best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Regression
# ──────────────────────────────────────────────────────────────────────────────

def _regression(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "LinearRegression": LinearRegression(n_jobs=-1),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    comparison: dict  = {}
    best_name:  str   = ""
    best_mae:   float = float("inf")
    best_model        = None
    best_metrics      = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = round(float(mean_absolute_error(y_test, y_pred)), 4)
        mse = round(float(mean_squared_error(y_test, y_pred)), 4)
        r2  = round(float(r2_score(y_test, y_pred)), 4)

        comparison[name] = {"mae": mae, "mse": mse, "r2_score": r2}

        if mae < best_mae:
            best_mae   = mae
            best_name  = name
            best_model = model
            best_metrics = {"mae": mae, "mse": mse, "r2_score": r2}

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

    comparison: dict  = {}
    best_name:  str   = ""
    best_sil:   float = -1.0
    best_model        = None
    best_metrics      = None

    for name, model in candidates.items():
        labels            = model.fit_predict(X)
        unique, counts    = np.unique(labels, return_counts=True)
        n_unique          = len(unique)

        sample_size = min(5_000, len(X))
        sil = (
            round(float(silhouette_score(X, labels, sample_size=sample_size)), 4)
            if n_unique > 1
            else -1.0
        )

        comparison[name] = {"silhouette_score": sil}

        if sil > best_sil:
            best_sil   = sil
            best_name  = name
            best_model = model
            best_metrics = {
                "silhouette_score": sil,
                "n_clusters":       int(n_unique),
                "cluster_sizes":    {str(int(k)): int(v) for k, v in zip(unique, counts)},
            }

    return best_model, best_name, comparison, best_metrics


def _find_best_k(X: np.ndarray, max_k: int = 8) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    cap     = min(max_k, len(X) - 1)
    best_k  = 2
    best_sc = -1.0

    for k in range(2, cap + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            sc = silhouette_score(X, labels, sample_size=min(3_000, len(X)))
            if sc > best_sc:
                best_sc = sc
                best_k  = k
        except Exception:
            continue

    return best_k
