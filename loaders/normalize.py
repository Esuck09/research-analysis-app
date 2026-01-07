"""
Normalization of experiment data into unified internal format.
"""

from typing import Dict, Any
import pandas as pd


def normalize_experiment(
    metrics_df: pd.DataFrame,
    metadata: Dict[str, Any],
    source_file: str,
) -> Dict[str, Any]:
    """
    Normalize experiment data into internal representation.

    Returns
    -------
    dict
        Unified experiment object.
    """
    raise NotImplementedError
