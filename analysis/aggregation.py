# analysis/aggregation.py

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.alignment import align_epochs
from analysis.resampling import resample_to_grid, build_epoch_grid


def aggregate_runs(
    experiments: list[dict],
    metric: str,
    group_by: list[str],
    alignment_mode: str = "intersection",
    last_n: int | None = None,
    use_interpolation: bool = False,
    grid_mode: str = "common_range",
    grid_step: int = 1,
    interp_kind: str = "linear",
) -> dict:
    """
    Aggregate multiple experiment runs into mean/std curves.

    Parameters
    ----------
    experiments : list[dict]
        Normalized experiment objects
    metric : str
        Metric to aggregate
    group_by : list[str]
        Metadata keys to group by (e.g. ["model", "dataset"])
    alignment_mode : str
        "intersection" | "truncate" | "last_n"
    last_n : int | None
        Used only if alignment_mode == "last_n"
    use_interpolation : bool
        If True, resample curves onto a shared epoch grid using interpolation
    grid_mode : str
        "common_range" | "union_range"
    grid_step : int
        Step size for epoch grid
    interp_kind : str
        Interpolation type (currently only "linear" supported)

    Returns
    -------
    dict[group_key -> DataFrame]
        DataFrame columns:
        - epoch
        - mean
        - std
        - n_runs
    """

    # -------------------------
    # Group experiments
    # -------------------------
    grouped: dict[tuple, list[dict]] = {}

    for exp in experiments:
        key = tuple(exp.get(k, "") for k in group_by)
        grouped.setdefault(key, []).append(exp)

    aggregated: dict[tuple, pd.DataFrame] = {}

    # -------------------------
    # Aggregate per group
    # -------------------------
    for key, runs in grouped.items():
        if len(runs) < 2:
            continue  # need at least 2 runs

        dfs: list[pd.DataFrame] = []

        for exp in runs:
            df = exp["metrics_df"]

            if metric not in df.columns:
                continue

            sub = df[["epoch", metric]].dropna()
            if sub.empty:
                continue

            sub = sub.sort_values("epoch").reset_index(drop=True)
            dfs.append(sub)

        if len(dfs) < 2:
            continue

        # =========================================================
        # INTERPOLATION PATH (V3-D)
        # =========================================================
        if use_interpolation:
            try:
                grid = build_epoch_grid(
                    dfs=dfs,
                    mode=grid_mode,
                    step=grid_step,
                )
            except Exception:
                continue

            if grid.size < 2:
                continue

            resampled = [
                resample_to_grid(
                    df,
                    metric=metric,
                    epoch_grid=grid,
                    kind=interp_kind,
                )
                for df in dfs
            ]

            if len(resampled) < 2:
                continue

            epochs = resampled[0]["epoch"].to_numpy(dtype=int)
            values = [r[metric].to_numpy(dtype=float) for r in resampled]

            stacked = pd.DataFrame(values).T  # epochs × runs

            # Drop epochs where ALL runs are NaN
            keep = ~stacked.isna().all(axis=1)
            stacked = stacked.loc[keep].reset_index(drop=True)
            epochs = epochs[keep]

            if len(epochs) < 2:
                continue

        # =========================================================
        # STRICT ALIGNMENT PATH (V3-C)
        # =========================================================
        else:
            try:
                aligned_dfs, _ = align_epochs(
                    dfs=dfs,
                    metric=metric,
                    mode=alignment_mode,
                    last_n=last_n,
                )
            except Exception:
                continue

            if len(aligned_dfs) < 2:
                continue

            lengths = [len(df) for df in aligned_dfs]
            if len(set(lengths)) != 1:
                continue

            epochs = aligned_dfs[0]["epoch"].to_numpy(dtype=int)
            values = [df[metric].to_numpy(dtype=float) for df in aligned_dfs]

            stacked = pd.DataFrame(values).T  # epochs × runs

        # -------------------------
        # Aggregate statistics
        # -------------------------
        if stacked.shape[1] < 2:
            continue

        agg_df = pd.DataFrame(
            {
                "epoch": epochs,
                "mean": stacked.mean(axis=1),
                "std": stacked.std(axis=1, ddof=1),
                "n_runs": stacked.shape[1],
            }
        )

        aggregated[key] = agg_df.reset_index(drop=True)

    return aggregated
