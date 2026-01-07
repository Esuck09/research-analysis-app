from typing import List
import pandas as pd
from metrics.registry import optimization_direction


def build_summary_table(
    experiments: List[dict],
    metric: str,
) -> pd.DataFrame:
    """
    Build summary statistics table for a given metric.

    Returns columns:
    - experiment_name
    - model
    - dataset
    - best_value
    - best_epoch
    - final_value
    """
    rows = []

    for exp in experiments:
        df = exp["metrics_df"]

        if metric not in df.columns:
            continue

        direction = optimization_direction(metric)

        metric_series = df[metric]
        epoch_series = df["epoch"]

        if direction == "min":
            best_idx = metric_series.idxmin()
        else:
            best_idx = metric_series.idxmax()

        rows.append(
            {
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "dataset": exp["dataset"],
                "best_value": metric_series.loc[best_idx],
                "best_epoch": epoch_series.loc[best_idx],
                "final_value": metric_series.iloc[-1],
            }
        )

    return pd.DataFrame(rows)
