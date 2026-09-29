"""Uncertainty and calibration for affinity regression.

``metrics.py`` answers "how did this model score". This answers "is that score
distinguishable from another one", which is a different and, on a 490-complex
test split, sharper question.

Why this module exists
----------------------
The LBA test split has n = 490. The 95% confidence interval on a single Pearson
r is about +/- 0.065 there, so most of the gaps this project cares about
(ligand-only vs full model, one lambda vs another) are smaller than the error
bar on either number alone.

Comparisons on the *same* test set are paired, which is far more powerful than
comparing two independent intervals, so :func:`paired_bootstrap_delta` is the
tool to reach for whenever two models are being ranked. Reporting each model's
own interval and eyeballing the overlap is the common mistake and it is both
conservative and wrong.

Seed variance does not substitute for any of this. Three seeds measure optimizer
noise (about +/- 0.004 for the RF baselines here); these functions measure
test-set sampling noise, which is an order of magnitude larger and identical
across seeds.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

METRICS = ("pearson_r", "spearman_r", "rmse", "mae")


def _as_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r for each row of two (n_rows, n_points) arrays."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(axis=1) / denominator, np.nan)


def _rowwise_metric(y: np.ndarray, p: np.ndarray, metric: str) -> np.ndarray:
    """One metric per row, vectorized over bootstrap resamples."""
    if metric == "pearson_r":
        return _rowwise_pearson(y, p)
    if metric == "spearman_r":
        # Spearman on a resample is Pearson on the ranks *within* that resample,
        # so the ranking has to happen per row, not once up front.
        return _rowwise_pearson(
            sp_stats.rankdata(y, axis=1), sp_stats.rankdata(p, axis=1)
        )
    if metric == "rmse":
        return np.sqrt(((y - p) ** 2).mean(axis=1))
    if metric == "mae":
        return np.abs(y - p).mean(axis=1)
    raise ValueError(f"unknown metric {metric!r}, have {METRICS}")


def pearson_ci(y_true, y_pred, alpha: float = 0.05) -> dict[str, float]:
    """Confidence interval on Pearson r via the Fisher z transform.

    Use this to report a single model honestly. To *compare* two models, use
    :func:`paired_bootstrap_delta` instead - these intervals ignore the fact
    that both models were scored on the same complexes, so comparing them
    understates your power considerably.

    :returns: dict with r, ci_low, ci_high, n.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)
    n = len(y_true)
    if n < 4 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return {"r": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}

    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    # Fisher z is approximately normal with SE 1/sqrt(n-3); transform, build the
    # interval there, then map back - which is why the interval is asymmetric.
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    half_width = sp_stats.norm.ppf(1 - alpha / 2) / np.sqrt(n - 3)
    return {
        "r": r,
        "ci_low": float(np.tanh(z - half_width)),
        "ci_high": float(np.tanh(z + half_width)),
        "n": n,
    }


def paired_bootstrap_delta(
    y_true,
    pred_a,
    pred_b,
    metric: str = "pearson_r",
    n_boot: int = 3000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Difference in a metric between two models, with a paired bootstrap CI.

    Resamples *complexes* (not predictions) with replacement, recomputes the
    metric for both models on each resample, and reports the distribution of
    ``metric(b) - metric(a)``. Because both models are scored on the same
    resampled complexes, the shared difficulty of those complexes cancels and
    the interval on the difference is much tighter than either model's own.

    Sign convention: positive delta means b scored higher. For rmse and mae,
    higher is *worse*, so read ``b_better`` rather than the sign.

    :returns: dict with delta, ci_low, ci_high, significant, b_better, n_boot.
    """
    y_true, pred_a = _as_pair(y_true, pred_a)
    _, pred_b = _as_pair(y_true, pred_b)
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}, have {METRICS}")

    n = len(y_true)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(n_boot, n))
    y = y_true[index]
    deltas = _rowwise_metric(y, pred_b[index], metric) - _rowwise_metric(
        y, pred_a[index], metric
    )
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {
            "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "significant": False, "b_better": False, "n_boot": 0,
        }

    low, high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(
        _rowwise_metric(y_true[None, :], pred_b[None, :], metric)[0]
        - _rowwise_metric(y_true[None, :], pred_a[None, :], metric)[0]
    )
    significant = bool(low > 0 or high < 0)
    lower_is_better = metric in ("rmse", "mae")
    return {
        "delta": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "significant": significant,
        "b_better": bool((point < 0) if lower_is_better else (point > 0)),
        "n_boot": int(deltas.size),
    }


def calibration_diagnostics(y_true, y_pred) -> dict[str, float]:
    """Whether the predictions are *numerically* right, not just well ordered.

    Pearson r is invariant to shifting and scaling the predictions, so a model
    can rank complexes well and still emit useless numbers. The retrieval probe
    in this project does exactly that: r = +0.34 but R2 = -0.51, because it
    copies real affinities and so inherits their full spread while only weakly
    correlating - over-dispersion is fatal to squared error.

    The gap between ``r_squared`` (Pearson r, squared - the variance a
    *perfectly recalibrated* version of this model could explain) and
    ``r2_score`` (the variance it actually explains) is the cost of that
    miscalibration.

    :returns: bias, pred_sd, true_sd, calibration_slope, r_squared, r2_score,
        calibration_gap.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)
    pred_sd = float(np.std(y_pred))
    true_var = float(np.var(y_true))

    # Slope of observed regressed on predicted. 1.0 is calibrated; below 1.0 the
    # predictions are more spread out than the information in them supports.
    slope = (
        float(np.cov(y_pred, y_true, bias=True)[0, 1] / np.var(y_pred))
        if pred_sd > 1e-12
        else float("nan")
    )
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if pred_sd > 1e-12 and true_var > 0 else float("nan")
    r2 = 1.0 - float(((y_true - y_pred) ** 2).mean()) / true_var if true_var > 0 else float("nan")
    return {
        "bias": float(np.mean(y_pred - y_true)),
        "pred_sd": pred_sd,
        "true_sd": float(np.std(y_true)),
        "calibration_slope": slope,
        "r_squared": r**2 if np.isfinite(r) else float("nan"),
        "r2_score": r2,
        "calibration_gap": (r**2 - r2) if np.isfinite(r) and np.isfinite(r2) else float("nan"),
    }
