"""
Centralized plotting styles.
"""
import matplotlib.pyplot as plt


def apply_matplotlib_style() -> None:
    """Apply publication-ready matplotlib styling."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
        }
    )
