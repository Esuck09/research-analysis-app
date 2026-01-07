"""
JSON experiment loader.

Responsible for:
- Parsing structured JSON experiment logs
- Converting metrics list to DataFrame
"""

from typing import Any, Dict
import json
import pandas as pd
from utils.io import ExperimentLoadError


def load_json(file: Any) -> Dict[str, Any]:
    """
    Load a JSON experiment file.

    Expected keys:
    - metrics: list of dicts with epoch + metric values
    - optional: experiment_name, task

    Parameters
    ----------
    file : file-like
        Uploaded JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    try:
        file.seek(0)  # 🔑 CRITICAL FIX
        return json.load(file)
    except json.JSONDecodeError as e:
        raise ExperimentLoadError(
            f"Failed to parse JSON file: {e}"
        )

    if not isinstance(content, dict):
        raise ExperimentLoadError("Top-level JSON must be an object.")

    if "metrics" not in content:
        raise ExperimentLoadError("JSON must contain a 'metrics' field.")

    if not isinstance(content["metrics"], list) or not content["metrics"]:
        raise ExperimentLoadError("'metrics' must be a non-empty list.")

    return content

def metrics_to_dataframe(metrics: list) -> pd.DataFrame:
    """
    Convert metrics list to DataFrame.

    Parameters
    ----------
    metrics : list of dict

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.DataFrame(metrics)
    except Exception as e:
        raise ValueError(f"Failed to convert metrics to DataFrame: {e}")

    if df.empty:
        raise ValueError("Metrics list produced empty DataFrame.")

    return df
