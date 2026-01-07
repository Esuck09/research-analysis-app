"""
Metric vs epoch plotting utilities.
"""

from typing import List
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plots.styling import apply_matplotlib_style

def plot_metric_curves(
    experiments: List[dict],
    metric: str,
) -> go.Figure:
    """
    Build an interactive Plotly metric-vs-epoch plot.
    """
    fig = go.Figure()

    for exp in experiments:
        df: pd.DataFrame = exp["metrics_df"]

        if metric not in df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df[metric],
                mode="lines+markers",
                name=f'{exp["experiment_name"]} '
                     f'({exp["model"]}, {exp["dataset"]})',
                hovertemplate=(
                    "Epoch: %{x}<br>"
                    f"{metric}: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title=metric,
        hovermode="x unified",
        legend_title="Experiments",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

def plot_metric_curves_matplotlib(
    experiments: List[dict],
    metric: str,
):
    """
    Build a publication-ready matplotlib comparison plot.
    """
    apply_matplotlib_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    for exp in experiments:
        df: pd.DataFrame = exp["metrics_df"]

        if metric not in df.columns:
            continue

        label = f'{exp["experiment_name"]} ({exp["model"]})'

        ax.plot(
            df["epoch"],
            df[metric],
            marker="o",
            label=label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Epoch")
    ax.grid(True)

    if len(experiments) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    return fig
