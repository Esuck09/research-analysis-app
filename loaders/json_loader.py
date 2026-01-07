"""
JSON experiment loader.

Responsible for:
- Parsing structured JSON experiment logs
- Converting metrics list to DataFrame
"""

from typing import Any, Dict
import json
import pandas as pd


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
        content = json.load(file)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON file: {e}")

    if not isinstance(content, dict):
        raise ValueError("Top-level JSON structure must be an object.")

    if "metrics" not in content:
        raise ValueError("JSON must contain a 'metrics' field.")

    if not isinstance(content["metrics"], list) or not content["metrics"]:
        raise ValueError("'metrics' must be a non-empty list.")

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
