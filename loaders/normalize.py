"""
Normalization of experiment data into unified internal format.
"""

from typing import Dict, Any
import pandas as pd
from utils.ids import generate_experiment_id


def normalize_experiment(
    metrics_df: pd.DataFrame,
    metadata: Dict[str, Any],
    source_file: str,
) -> Dict[str, Any]:
    """
    Normalize experiment data into unified internal representation.
    """
    required_meta = ["experiment_name", "task", "model", "dataset"]
    for key in required_meta:
        if key not in metadata or not metadata[key]:
            raise ValueError(f"Missing required metadata field: {key}")

    available_metrics = [c for c in metrics_df.columns if c != "epoch"]

    experiment_id = generate_experiment_id(
        metrics_df=metrics_df,
        metadata=metadata,
        source_file=source_file,
    )

    return {
        "id": experiment_id,
        "experiment_name": metadata["experiment_name"],
        "task": metadata["task"],
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "source_file": source_file,
        "metrics_df": metrics_df,
        "available_metrics": available_metrics,
    }

