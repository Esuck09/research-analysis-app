"""
Metric vs epoch plotting utilities.
"""

from typing import List
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from plots.styling import apply_matplotlib_style
from plots.emphasis import build_color_map, emphasis_style


# =========================
# Plotly (Interactive)
# =========================
def plot_metric_curves(
    experiments,
    metric,
    highlight=None,
    color_mode="by_experiment",
) -> go.Figure:
    """
    Build an interactive Plotly metric-vs-epoch plot.
    """
    if highlight is None:
        highlight = []

    fig = go.Figure()
    color_map = build_color_map(experiments, mode=color_mode)

    for exp in experiments:
        df: pd.DataFrame = exp["metrics_df"]

        if metric not in df.columns:
            continue

        style = emphasis_style(
            exp["experiment_name"],
            highlight,
            base_width=3,
        )

        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df[metric],
                mode="lines+markers",
                name=f'{exp["experiment_name"]} '
                     f'({exp["model"]}, {exp["dataset"]})',
                line=dict(
                    color=color_map[exp["id"]],
                    width=style["line_width"],
                ),
                opacity=style["alpha"],
                hovertemplate=(
                    "Epoch: %{x}<br>"
                    f"{metric}: %{{y:.4f}}"
                    "<extra></extra>"
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


# =========================
# Matplotlib (Publication)
# =========================
def plot_metric_curves_matplotlib(
    experiments: List[dict],
    metric: str,
    highlight: list[str] | None = None,
):
    """
    Build a publication-ready matplotlib comparison plot.

    highlight:
        list of experiment_name to emphasize
    """
    apply_matplotlib_style()

    if highlight is None:
        highlight = []

    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = build_color_map(experiments)

    for exp in experiments:
        df: pd.DataFrame = exp["metrics_df"]

        if metric not in df.columns:
            continue

        name = exp["experiment_name"]
        label = f'{name} ({exp["model"]})'

        style = emphasis_style(
            name,
            highlight,
            base_width=2.5,
        )

        ax.plot(
            df["epoch"],
            df[metric],
            marker="o",
            label=label,
            color=color_map[exp["id"]],
            linewidth=style["line_width"],
            alpha=style["alpha"],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)

    if len(experiments) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    return fig

def plot_aggregated_curves(
    fig,
    aggregated: dict,
    metric: str,
):
    """
    Overlay mean ± std bands onto an existing Plotly figure.
    """
    

    for key, df in aggregated.items():
        label = " / ".join(str(k) for k in key)

        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["mean"],
                mode="lines",
                name=f"{label} (mean)",
                line=dict(width=4, dash="solid"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(df["epoch"]) + list(df["epoch"][::-1]),
                y=list(df["mean"] + df["std"]) + list((df["mean"] - df["std"])[::-1]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.12)",
                line=dict(width=0),
                name=f"{label} ± std",
                showlegend=False,
            )
        )
