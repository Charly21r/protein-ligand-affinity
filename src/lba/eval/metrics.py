"""Regression metrics for affinity prediction."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn import metrics as sk_metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard set of affinity-regression metrics.

    :param y_true: observed pKd/pKi values.
    :param y_pred: predicted values, same length.
    :returns: dict with pearson_r, spearman_r, rmse, mae, r2.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    # Correlation needs variance on *both* sides. A constant prediction is a
    # collapsed model, which is a real outcome worth reporting as 0.0 rather
    # than NaN. A constant y_true is different: it means the subset is
    # degenerate (one distinct label), which happens on the per-family
    # breakdowns in Phase 5. Correlation is undefined there, not zero, so it is
    # reported as NaN and the caller can drop those groups.
    if np.std(y_true) < 1e-12:
        pearson = spearman = float("nan")
    elif np.std(y_pred) < 1e-12:
        pearson = spearman = 0.0
    else:
        pearson = float(stats.pearsonr(y_true, y_pred)[0])
        spearman = float(stats.spearmanr(y_true, y_pred)[0])

    return {
        "pearson_r": pearson,
        "spearman_r": spearman,
        "rmse": float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))),
        "mae": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "r2": float(sk_metrics.r2_score(y_true, y_pred)),
    }


def format_metrics(name: str, scores: dict[str, float]) -> str:
    """One-line rendering for logs and notebook entries."""
    return (
        f"{name:<28} r={scores['pearson_r']:+.3f}  rho={scores['spearman_r']:+.3f}  "
        f"RMSE={scores['rmse']:.3f}  MAE={scores['mae']:.3f}"
    )
