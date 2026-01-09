# analysis/aggregation.py

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.alignment import align_epochs
from analysis.resampling import build_epoch_grid, resample_to_grid
from analysis.bootstrap import bootstrap_curve_ci


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
    compute_ci: bool = False,
    n_boot: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """
    Aggregate multiple experiment runs into mean/std and optional CI curves.
    """

    grouped: dict[tuple, list[dict]] = {}
    for exp in experiments:
        key = tuple(exp.get(k, "") for k in group_by)
        grouped.setdefault(key, []).append(exp)

    aggregated: dict[tuple, pd.DataFrame] = {}

    for key, runs in grouped.items():
        if len(runs) < 2:
            continue

        dfs = []
        for exp in runs:
            df = exp["metrics_df"]
            if metric not in df.columns:
                continue
            sub = df[["epoch", metric]].dropna().sort_values("epoch")
            if not sub.empty:
                dfs.append(sub)

        if len(dfs) < 2:
            continue

        # =====================================================
        # Interpolation path
        # =====================================================
        if use_interpolation:
            grid = build_epoch_grid(dfs, mode=grid_mode, step=grid_step)
            if len(grid) < 2:
                continue

            resampled = [
                resample_to_grid(df, metric, grid, interp_kind)
                for df in dfs
            ]

            epochs = resampled[0]["epoch"].to_numpy()
            values = [r[metric].to_numpy() for r in resampled]
            stacked = pd.DataFrame(values).T

            keep = ~stacked.isna().all(axis=1)
            stacked = stacked.loc[keep]
            epochs = epochs[keep]

            if len(epochs) < 2:
                continue

        # =====================================================
        # Strict alignment path
        # =====================================================
        else:
            aligned, _ = align_epochs(
                dfs,
                metric=metric,
                mode=alignment_mode,
                last_n=last_n,
            )

            if len(aligned) < 2:
                continue

            epochs = aligned[0]["epoch"].to_numpy()
            values = [df[metric].to_numpy() for df in aligned]
            stacked = pd.DataFrame(values).T

        # =====================================================
        # Aggregate statistics
        # =====================================================
        mean = stacked.mean(axis=1)
        std = stacked.std(axis=1, ddof=1)

        agg_df = pd.DataFrame(
            {
                "epoch": epochs,
                "mean": mean,
                "std": std,
                "n_runs": stacked.shape[1],
            }
        )

        if compute_ci:
            ci_df = bootstrap_curve_ci(
                stacked,
                n_boot=n_boot,
                ci=ci_level,
                random_state=42,
            )
            agg_df = pd.concat([agg_df, ci_df], axis=1)

        aggregated[key] = agg_df.reset_index(drop=True)

    return aggregated
