"""
Shared IO helpers.
"""
import io
import pandas as pd

class ExperimentLoadError(Exception):
    """Raised when an experiment file cannot be loaded or validated."""
    pass

def dataframe_to_csv_buffer(df: pd.DataFrame) -> io.BytesIO:
    """
    Convert DataFrame to an in-memory CSV buffer.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    io.BytesIO
        CSV buffer.
    """
    buffer = io.BytesIO()
    buffer.write(df.to_csv(index=False).encode("utf-8"))
    buffer.seek(0)
    return buffer