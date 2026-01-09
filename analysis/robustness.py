# analysis/robustness.py

from __future__ import annotations
import numpy as np
import pandas as pd


def _is_loss(metric: str) -> bool:
    return "loss" in metric.lower()


def compute_robustness_metrics(
    experiments: list[dict],
    metric: str,
    tail_fraction: float = 0.25,
    plateau_tol: float = 1e-3,
) -> pd.DataFrame:
    """
    Compute robustness & stability metrics for experiments.

    Parameters
    ----------
    experiments : list[dict]
    metric : str
    tail_fraction : float
        Fraction of last epochs used for stability analysis
    plateau_tol : float
        Tolerance for plateau detection

    Returns
    -------
    DataFrame with columns:
      experiment_name
      model
      dataset
      n_epochs
      late_variance
      stability_score
      plateau_detected
    """
    rows = []

    for exp in experiments:
        df = exp["metrics_df"]

        if metric not in df.columns:
            continue

        s = df[["epoch", metric]].dropna()
        if len(s) < 5:
            continue

        y = s[metric].to_numpy(dtype=float)
        n = len(y)

        # -------------------------
        # Late-epoch variance
        # -------------------------
        k = max(3, int(n * tail_fraction))
        tail = y[-k:]

        late_var = float(np.var(tail, ddof=1))

        # -------------------------
        # Stability score (normalized)
        # -------------------------
        overall_var = float(np.var(y, ddof=1))
        stability = 1.0 - (late_var / overall_var) if overall_var > 0 else 1.0
        stability = float(np.clip(stability, 0.0, 1.0))

        # -------------------------
        # Plateau detection
        # -------------------------
        diffs = np.abs(np.diff(tail))
        plateau = bool(np.all(diffs < plateau_tol))

        rows.append(
            {
                "experiment_name": exp["experiment_name"],
                "model": exp.get("model", ""),
                "dataset": exp.get("dataset", ""),
                "metric": metric,
                "n_epochs": n,
                "late_variance": late_var,
                "stability_score": stability,
                "plateau_detected": plateau,
            }
        )

    return pd.DataFrame(rows)
