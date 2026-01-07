"""
Validation utilities for experiment metrics.
"""

import pandas as pd



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
        raise ValueError("Missing required 'epoch' column.")

    df = df.copy()

    # Drop rows with missing epoch
    df = df.dropna(subset=["epoch"])

    # Coerce epoch to int
    try:
        df["epoch"] = df["epoch"].astype(int)
    except Exception:
        raise ValueError("Epoch column must be integer-convertible.")

    # Coerce metric columns to numeric
    metric_cols = [c for c in df.columns if c != "epoch"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where all metrics are NaN
    if metric_cols:
        df = df.dropna(subset=metric_cols, how="all")

    if df.empty:
        raise ValueError("No valid metric rows after validation.")

    # Resolve duplicate epochs (keep last)
    df = df.sort_values("epoch")
    df = df.drop_duplicates(subset="epoch", keep="last")

    return df.reset_index(drop=True)
