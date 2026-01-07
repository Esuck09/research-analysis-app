"""
Stable experiment ID generation.
"""

import hashlib
import json
from typing import Dict, Any
import pandas as pd


def generate_experiment_id(
    metrics_df: pd.DataFrame,
    metadata: Dict[str, Any],
    source_file: str,
) -> str:
    """
    Generate stable hash-based experiment ID.
    """
    payload = {
        "metrics": metrics_df.to_dict(orient="list"),
        "metadata": metadata,
        "source_file": source_file,
    }

    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

