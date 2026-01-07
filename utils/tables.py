from typing import List
import pandas as pd
from metrics.registry import optimization_direction
from typing import List, Set

def build_summary_table(
    experiments: List[dict],
    metric: str,
) -> pd.DataFrame:
    """
    Build summary statistics table for a given metric.

    Parameters
    ----------
    experiments : list of dict
        Unified experiment objects.
    metric : str
        Metric name to summarize.

    Returns
    -------
    pd.DataFrame
        Summary table with best and final statistics.
    """
    rows = []

    for exp in experiments:
        df = exp["metrics_df"]

        if metric not in df.columns:
            continue

        direction = optimization_direction(metric)

        metric_series = df[metric]
        epoch_series = df["epoch"]

        if metric_series.isna().all():
            continue

        if direction == "min":
            best_idx = metric_series.idxmin()
        else:
            best_idx = metric_series.idxmax()

        rows.append(
            {
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "dataset": exp["dataset"],
                "task": exp["task"],
                "best_value": metric_series.loc[best_idx],
                "best_epoch": int(epoch_series.loc[best_idx]),
                "final_value": metric_series.iloc[-1],
            }
        )

    return pd.DataFrame(rows)

def intersect_available_metrics(experiments: List[dict]) -> Set[str]:
    """
    Compute intersection of available metrics across experiments.
    """
    if not experiments:
        return set()

    metric_sets = [
        set(exp["available_metrics"]) for exp in experiments
    ]

    return set.intersection(*metric_sets)
