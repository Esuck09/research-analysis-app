"""
Experiment summary table utilities.
"""

from typing import List
import pandas as pd


def build_summary_table(
    experiments: List[dict],
    metric: str,
) -> pd.DataFrame:
    """
    Compute summary statistics for experiments.
    """
    raise NotImplementedError
