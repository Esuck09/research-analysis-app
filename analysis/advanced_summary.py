import pandas as pd
import numpy as np


def is_loss_metric(metric_name: str) -> bool:
    return "loss" in metric_name.lower()


def compute_advanced_summary(experiments: list[dict], metric: str) -> pd.DataFrame:
    rows = []

    minimize = is_loss_metric(metric)

    for exp in experiments:
        df = exp["metrics_df"]

        if metric not in df.columns:
            continue

        series = df[["epoch", metric]].dropna()
        if series.empty:
            continue

        values = series[metric].values
        epochs = series["epoch"].values

        # Best value
        if minimize:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        best_value = values[best_idx]
        best_epoch = int(epochs[best_idx])

        final_value = values[-1]
        final_epoch = int(epochs[-1])

        delta = final_value - best_value

        pct_change = (
            (delta / abs(best_value)) * 100
            if best_value != 0
            else np.nan
        )

        # Plateau epoch (within 1% of best)
        tol = abs(best_value) * 0.01
        plateau_epoch = None
        for e, v in zip(epochs, values):
            if abs(v - best_value) <= tol:
                plateau_epoch = int(e)
                break

        rows.append(
            {
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "dataset": exp["dataset"],
                "task": exp["task"],
                "best_value": best_value,
                "best_epoch": best_epoch,
                "final_value": final_value,
                "final_epoch": final_epoch,
                "delta_best_final": delta,
                "pct_change_best_final": pct_change,
                "plateau_epoch": plateau_epoch,
                "epochs_to_best": best_epoch,
            }
        )

    return pd.DataFrame(rows)
