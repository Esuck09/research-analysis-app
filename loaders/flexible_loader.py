import json
import pandas as pd
from utils.io import ExperimentLoadError


def load_metrics_any_format(file):
    """
    Load metrics from ANY reasonable CSV or JSON structure
    and return a flat pandas DataFrame with an epoch column.
    """

    filename = file.name.lower()

    # 🔴 ALWAYS rewind file (critical!)
    file.seek(0)

    # --------------------------------------------------
    # CSV PATH
    # --------------------------------------------------
    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            return _normalize_dataframe(df)
        except Exception as e:
            raise ExperimentLoadError(f"Failed to read CSV: {e}")

    # --------------------------------------------------
    # JSON PATH
    # --------------------------------------------------
    try:
        file.seek(0)
        raw = json.load(file)
    except json.JSONDecodeError:
        raise ExperimentLoadError(
            "File is not valid JSON. "
            "If this is a CSV, ensure it has a .csv extension."
        )

    return _parse_json_metrics(raw)


# ======================================================
# Helpers
# ======================================================

def _parse_json_metrics(raw):
    """
    Handle common JSON experiment formats.
    """

    # Case 1: { "metrics": {...} }
    if isinstance(raw, dict) and "metrics" in raw:
        return _normalize_dataframe(pd.DataFrame(raw["metrics"]))

    # Case 2: { "epoch": [...], "accuracy": [...] }
    if isinstance(raw, dict):
        return _normalize_dataframe(pd.DataFrame(raw))

    # Case 3: [ {epoch: 1, acc: ...}, ... ]
    if isinstance(raw, list):
        return _normalize_dataframe(pd.DataFrame(raw))

    raise ExperimentLoadError("Unsupported JSON structure.")


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure:
    - epoch column exists
    - metrics are numeric
    """

    df = df.copy()

    # 🔹 Create epoch if missing
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))

    # 🔹 Force numeric metrics
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all")

    if len(df.columns) <= 1:
        raise ExperimentLoadError("No valid metric columns found.")

    return df
