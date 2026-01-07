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
    raise NotImplementedError
