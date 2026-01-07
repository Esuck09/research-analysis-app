# plots/curves.py

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from plots.styling import apply_matplotlib_style


# =========================================================
# Interactive Plotly curves (USED BY APP)
# =========================================================
def plot_metric_curves(
    experiments: list[dict],
    metric: str,
    highlight: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()

    for exp in experiments:
        df: pd.DataFrame = exp["metrics_df"]
        if metric not in df.columns:
            continue

        opacity = 1.0
        width = 2.5

        if highlight:
            if exp["experiment_name"] in highlight:
                width = 3.5
            else:
                opacity = 0.25
                width = 1.5

        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df[metric],
                mode="lines+markers",
                name=exp["experiment_name"],
                line=dict(width=width),
                opacity=opacity,
            )
        )

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title=metric,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


# =========================================================
# Aggregated curves overlay (mean / std / CI)
# =========================================================
def plot_aggregated_curves(
    fig: go.Figure,
    aggregated: dict,
    metric: str,
    show_std: bool = True,
    show_ci: bool = True,
):
    for key, df in aggregated.items():
        label = " / ".join(str(k) for k in key)

        # Mean curve
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["mean"],
                mode="lines",
                name=f"{label} (mean)",
                line=dict(width=4),
            )
        )

        # Std band
        if show_std and "std" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(df["epoch"]) + list(df["epoch"][::-1]),
                    y=list(df["mean"] + df["std"])
                    + list((df["mean"] - df["std"])[::-1]),
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.12)",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

        # CI band
        if show_ci and "ci_lower" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(df["epoch"]) + list(df["epoch"][::-1]),
                    y=list(df["ci_upper"]) + list(df["ci_lower"][::-1]),
                    fill="toself",
                    fillcolor="rgba(30,144,255,0.18)",
                    line=dict(width=0),
                    showlegend=False,
                )
            )


# =========================================================
# Matplotlib publication plot
# =========================================================
def plot_metric_curves_matplotlib(
    experiments: list[dict],
    metric: str,
    highlight: list[str] | None = None,
):
    apply_matplotlib_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    for exp in experiments:
        df = exp["metrics_df"]
        if metric not in df.columns:
            continue

        ax.plot(df["epoch"], df[metric], label=exp["experiment_name"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Epoch")

    if len(experiments) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    return fig
