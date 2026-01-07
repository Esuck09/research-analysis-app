# analysis/dynamics.py

from __future__ import annotations
import numpy as np
import pandas as pd


def _find_train_val_pair(columns: list[str], base_metric: str) -> tuple[str | None, str | None]:
    """
    Given columns and a base metric like "loss" or "accuracy",
    attempt to find train_* and val_* columns.
    """
    base = base_metric.lower().strip()

    # common patterns
    train_candidates = [
        f"train_{base}",
        f"training_{base}",
        f"{base}_train",
    ]
    val_candidates = [
        f"val_{base}",
        f"valid_{base}",
        f"validation_{base}",
        f"{base}_val",
    ]

    cols = [c.lower() for c in columns]

    train_col = None
    val_col = None

    for tc in train_candidates:
        if tc in cols:
            train_col = columns[cols.index(tc)]
            break

    for vc in val_candidates:
        if vc in cols:
            val_col = columns[cols.index(vc)]
            break

    return train_col, val_col


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Simple linear regression slope (least squares).
    """
    if len(x) < 2:
        return np.nan
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return np.nan
    return float((x * (y - y.mean())).sum() / denom)


def compute_learning_dynamics(
    experiments: list[dict],
    base_metric: str = "loss",
    tail_fraction: float = 0.25,
) -> pd.DataFrame:
    """
    Compute learning dynamics diagnostics for each experiment.

    Parameters
    ----------
    experiments : list[dict]
    base_metric : str
        "loss", "accuracy", "dice", etc. (must have train/val pair)
    tail_fraction : float
        fraction of last epochs used to compute slope

    Returns
    -------
    DataFrame with:
      experiment_name, model, dataset, train_col, val_col,
      n_epochs, gap_final, gap_slope_tail, best_val_epoch, val_drop_after_best
    """
    rows = []

    for exp in experiments:
        df = exp["metrics_df"]
        cols = list(df.columns)

        train_col, val_col = _find_train_val_pair(cols, base_metric)
        if not train_col or not val_col:
            continue

        sub = df[["epoch", train_col, val_col]].dropna()
        if len(sub) < 5:
            continue

        x = sub["epoch"].to_numpy(dtype=float)
        train = sub[train_col].to_numpy(dtype=float)
        val = sub[val_col].to_numpy(dtype=float)

        is_loss = "loss" in base_metric.lower()

        # Generalization gap definition:
        # loss: val - train (higher gap worse)
        # acc-like: train - val (higher gap worse)
        gap = (val - train) if is_loss else (train - val)

        # Final gap
        gap_final = float(gap[-1])

        # Tail slope
        n = len(gap)
        k = max(3, int(n * tail_fraction))
        gap_slope_tail = _linear_slope(x[-k:], gap[-k:])

        # Best validation epoch and drop after best
        if is_loss:
            best_idx = int(np.argmin(val))
        else:
            best_idx = int(np.argmax(val))

        best_val_epoch = int(sub["epoch"].iloc[best_idx])
        best_val_value = float(val[best_idx])

        # Measure degradation after best: compare last val to best val
        val_drop_after_best = float(val[-1] - best_val_value) if is_loss else float(best_val_value - val[-1])

        rows.append(
            {
                "experiment_name": exp["experiment_name"],
                "model": exp.get("model", ""),
                "dataset": exp.get("dataset", ""),
                "base_metric": base_metric,
                "train_col": train_col,
                "val_col": val_col,
                "n_epochs": int(len(sub)),
                "gap_final": gap_final,
                "gap_slope_tail": gap_slope_tail,
                "best_val_epoch": best_val_epoch,
                "val_drop_after_best": val_drop_after_best,
            }
        )

    return pd.DataFrame(rows)
