# plots/emphasis.py

from typing import List, Dict
import hashlib

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

def build_color_map(experiments, mode="by_experiment"):
    """
    Returns: dict {experiment_id: valid_plotly_color}
    """
    color_map = {}

    for i, exp in enumerate(experiments):
        exp_id = exp["id"]

        if mode == "by_experiment":
            # Stable palette-based assignment
            color = PALETTE[i % len(PALETTE)]

        elif mode == "by_hash":
            # Hash → valid hex color
            h = hashlib.md5(exp_id.encode()).hexdigest()
            color = "#" + h[:6]

        else:
            raise ValueError(f"Unknown color_mode: {mode}")

        color_map[exp_id] = color

    return color_map


def emphasis_style(
    experiment_name: str,
    highlight_names: List[str] | None,
    base_width: float = 2.5,
):
    """
    Return styling for emphasized vs non-emphasized curves.

    Returns
    -------
    dict with:
        - line_width
        - alpha
    """
    if not highlight_names:
        return {
            "line_width": base_width,
            "alpha": 1.0,
        }

    if experiment_name in highlight_names:
        return {
            "line_width": base_width * 1.3,
            "alpha": 1.0,
        }

    return {
        "line_width": max(base_width * 0.6, 1.0),
        "alpha": 0.25,
    }
