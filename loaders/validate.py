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
    - duplicate epochs resolved by keeping last
    - sorted by epoch

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.
    """
    raise NotImplementedError
