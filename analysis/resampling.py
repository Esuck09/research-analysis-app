# analysis/resampling.py

from __future__ import annotations
import numpy as np
import pandas as pd


def resample_to_grid(
    df: pd.DataFrame,
    metric: str,
    epoch_grid: np.ndarray,
    kind: str = "linear",
) -> pd.DataFrame:
    """
    Resample a metric curve to a fixed epoch grid using interpolation.

    df must have columns ["epoch", metric] and be sorted by epoch.

    kind: currently only "linear" supported (defensible default).
    """
    if kind != "linear":
        raise ValueError("Only linear interpolation is supported in V3-D.")

    sub = df[["epoch", metric]].dropna().sort_values("epoch")
    if sub.empty:
        return pd.DataFrame({"epoch": epoch_grid, metric: np.nan})

    x = sub["epoch"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    # np.interp requires increasing x and does linear interpolation
    # outside range -> uses boundary values; we want NaN outside range for defensibility
    y_interp = np.interp(epoch_grid.astype(float), x, y)

    # mask outside original x-range
    mask_outside = (epoch_grid < x.min()) | (epoch_grid > x.max())
    y_interp = y_interp.astype(float)
    y_interp[mask_outside] = np.nan

    return pd.DataFrame({"epoch": epoch_grid.astype(int), metric: y_interp})


def build_epoch_grid(
    dfs: list[pd.DataFrame],
    mode: str = "common_range",
    step: int = 1,
) -> np.ndarray:
    """
    Create a shared epoch grid.

    mode:
      - "common_range": [max(min_epoch), min(max_epoch)] (overlap only)
      - "union_range":  [min(min_epoch), max(max_epoch)] (full union)
    """
    mins = [int(df["epoch"].min()) for df in dfs if not df.empty]
    maxs = [int(df["epoch"].max()) for df in dfs if not df.empty]
    if not mins or not maxs:
        return np.array([], dtype=int)

    if mode == "common_range":
        start = max(mins)
        end = min(maxs)
    elif mode == "union_range":
        start = min(mins)
        end = max(maxs)
    else:
        raise ValueError(f"Unknown grid mode: {mode}")

    if start > end:
        return np.array([], dtype=int)

    return np.arange(start, end + 1, step, dtype=int)
