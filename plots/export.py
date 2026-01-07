"""
Plot export utilities.
"""
import io
from matplotlib.figure import Figure


def export_figure_to_png(fig: Figure, dpi: int = 300) -> io.BytesIO:
    """
    Export a matplotlib figure to an in-memory PNG buffer.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    dpi : int
        Resolution for export.

    Returns
    -------
    io.BytesIO
        PNG image buffer.
    """
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
    )
    buffer.seek(0)
    return buffer
