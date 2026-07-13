"""
experiments/stats.py
------------------------
Common statistical helpers used across experiment reports
"""

from __future__ import annotations
import numpy as np


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """
    Return (point_estimate, lo, hi) where [lo, hi] is the 95% percentile bootstrap CI of `statistic` applied to `values`.

    Strips None entries first.
    """
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    point = float(statistic(v))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = statistic(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_t_test(a, b) -> tuple[float, float]:
    """Paired t-test on the differences a[i] - b[i]; returns (t, p)."""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats
        t, p = stats.ttest_rel(a, b)
        return float(t), float(p)
    except ImportError:
        d = a - b
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        t = d.mean() / se if se > 0 else float("nan")
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(t), float(p)


def wilcoxon_signed_rank(a, b) -> tuple[float, float]:
    """Non-parametric paired test on a[i] - b[i]; returns (W, p)."""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan"), float("nan")
    try:
        from scipy import stats
        w, p = stats.wilcoxon(a, b)
        return float(w), float(p)
    except ImportError:
        return float("nan"), float("nan")


def fmt_ci(point: float, lo: float, hi: float, prec: int = 1) -> str:
    """Format `point [lo, hi]` as a string."""
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def fmt_pm(mean: float, std: float, prec: int = 1) -> str:
    """Format `mean +- std` as a string (report style, mean +- standard deviation)."""
    return f"{mean:.{prec}f} +- {std:.{prec}f}"


def summarise(values, label: str = "", prec: int = 1) -> dict:
    """Return a dict with the common headline stats."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"label": label, "n": 0}
    point, lo, hi = bootstrap_ci(v)
    return {
        "label":  label,
        "n":      len(v),
        "mean":   float(v.mean()),
        "median": float(np.median(v)),
        "std":    float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min":    float(v.min()),
        "max":    float(v.max()),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }