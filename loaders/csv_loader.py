"""
CSV experiment loader.

Responsible for:
- Reading CSV files
- Basic column sanitation
- Returning raw DataFrame for validation
"""

from typing import Any
import pandas as pd


def load_csv(file: Any) -> pd.DataFrame:
    """
    Load a CSV experiment file into a DataFrame.

    Parameters
    ----------
    file : file-like
        Uploaded CSV file.

    Returns
    -------
    pd.DataFrame
        Raw experiment metrics.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    if df.empty:
        raise ValueError("CSV file is empty.")

    return df
