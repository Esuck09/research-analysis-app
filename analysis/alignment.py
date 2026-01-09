# analysis/alignment.py

from __future__ import annotations
import pandas as pd


def align_epochs(
    dfs: list[pd.DataFrame],
    metric: str,
    mode: str = "intersection",
    last_n: int | None = None,
) -> tuple[list[pd.DataFrame], dict]:
    """
    Align metric DataFrames across runs.

    Parameters
    ----------
    dfs : list of DataFrames with columns ["epoch", metric]
    metric : str
    mode : str
        "intersection" | "truncate" | "last_n"
    last_n : int | None
        Used only if mode == "last_n"

    Returns
    -------
    aligned_dfs : list[pd.DataFrame]
    info : dict with alignment metadata
    """
    info = {
        "mode": mode,
        "n_runs": len(dfs),
        "original_lengths": [len(df) for df in dfs],
    }

    # Ensure sorted
    dfs = [df.sort_values("epoch").reset_index(drop=True) for df in dfs]

    if mode == "intersection":
        epoch_sets = [set(df["epoch"]) for df in dfs]
        common_epochs = sorted(set.intersection(*epoch_sets))
        aligned = [
            df[df["epoch"].isin(common_epochs)].reset_index(drop=True)
            for df in dfs
        ]
        info["aligned_epochs"] = len(common_epochs)
        return aligned, info

    if mode == "truncate":
        min_len = min(len(df) for df in dfs)
        aligned = [df.iloc[:min_len].reset_index(drop=True) for df in dfs]
        info["aligned_epochs"] = min_len
        return aligned, info

    if mode == "last_n":
        if not last_n or last_n <= 0:
            raise ValueError("last_n must be positive when mode == 'last_n'")
        aligned = [
            df.iloc[-last_n:].reset_index(drop=True)
            for df in dfs
            if len(df) >= last_n
        ]
        info["aligned_epochs"] = last_n
        return aligned, info

    raise ValueError(f"Unknown alignment mode: {mode}")
