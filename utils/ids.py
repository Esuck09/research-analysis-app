"""
Stable experiment ID generation.
"""

from typing import Dict, Any


def generate_experiment_id(
    metrics_df,
    metadata: Dict[str, Any],
    source_file: str,
) -> str:
    """
    Generate stable hash-based experiment ID.
    """
    raise NotImplementedError
