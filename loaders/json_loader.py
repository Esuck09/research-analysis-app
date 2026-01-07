"""
JSON experiment loader.

Responsible for:
- Parsing structured JSON experiment logs
- Converting metrics list to DataFrame
"""

from typing import Any, Dict
import pandas as pd


def load_json(file: Any) -> Dict[str, Any]:
    """
    Load a JSON experiment file.

    Parameters
    ----------
    file : file-like
        Uploaded JSON file.

    Returns
    -------
    dict
        Parsed experiment data including metrics.
    """
    raise NotImplementedError
