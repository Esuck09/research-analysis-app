"""
Metric vs epoch plotting utilities.
"""

from typing import List
import pandas as pd
import plotly.graph_objects as go


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
