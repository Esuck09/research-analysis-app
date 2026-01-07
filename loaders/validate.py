"""
Validation utilities for experiment metrics.
"""

import pandas as pd
from utils.io import ExperimentLoadError


def validate_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a metrics DataFrame.

    Rules:
    - epoch column must exist
    - epoch must be integer
    - drop rows with missing epoch
    - coerce metrics to numeric
    - duplicate epochs resolved by keeping last
    - sorted by epoch
    """
    if "epoch" not in df.columns:
        raise ExperimentLoadError("Missing required 'epoch' column.")

    df = df.copy()
    df = df.dropna(subset=["epoch"])

    try:
        df["epoch"] = df["epoch"].astype(int)
    except Exception:
        raise ExperimentLoadError("Epoch column must contain integers.")

    metric_cols = [c for c in df.columns if c != "epoch"]

    if not metric_cols:
        raise ExperimentLoadError("No metric columns found.")

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=metric_cols, how="all")

    if df.empty:
        raise ExperimentLoadError("All metric values are invalid or missing.")

    df = df.sort_values("epoch")
    df = df.drop_duplicates(subset="epoch", keep="last")

    return df.reset_index(drop=True)
