"""Plot style functions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional
import warnings

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.ticker import (
    LogFormatterExponent,
    LogLocator,
    NullFormatter,
)


PLOTTING_PARAMS: Dict[str, Any] = {
    "lines.linewidth": 4,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 17,
    "legend.fontsize": 15,
    "legend.frameon": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.autolayout": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}


def apply_plot_style(params: Optional[Dict[str, Any]] = None) -> None:
    """Apply project plotting style globally via matplotlib rcParams."""
    cfg = (
        PLOTTING_PARAMS
        if params is None
        else {**PLOTTING_PARAMS, **params}
    )
    mpl.rcParams.update(cfg)


@contextmanager
def plot_style(params: Optional[Dict[str, Any]] = None):
    """Temporarily apply plot style inside a `with` block (restores after)."""
    style = PLOTTING_PARAMS if params is None else {**PLOTTING_PARAMS,
                                                    **params}
    with mpl.rc_context(style):
        yield


def set_log_scale_and_format(
    ax: Axes,
    *,
    x_numticks: int = 10,
    y_numticks: int = 5,
    base: float = 10.0,
    minor: bool = True,
) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")

    # --- majors ---
    ax.xaxis.set_major_locator(LogLocator(base=base,
                                          subs=(1.0,),
                                          numticks=x_numticks))
    ax.yaxis.set_major_locator(LogLocator(base=base,
                                          subs=(1.0,),
                                          numticks=y_numticks))
    ax.xaxis.set_major_formatter(LogFormatterExponent(base=base))
    ax.yaxis.set_major_formatter(LogFormatterExponent(base=base))

    if not minor:
        return

    # --- minors ---
    subs_minor = tuple(range(2, 10))
    ax.xaxis.set_minor_locator(
        LogLocator(base=base, subs=subs_minor, numticks=1000))
    ax.yaxis.set_minor_locator(
        LogLocator(base=base, subs=subs_minor, numticks=1000))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # force minor ticks to be enabled on each side (this is the key)
    ax.tick_params(axis="x", which="minor", bottom=True, top=True)
    ax.tick_params(axis="y", which="minor", left=True, right=True)


def get_colors(
    n: int,
    *,
    cmap: str = "cmr.pride",
    cmap_range: tuple[float, float] = (0.25, 0.75),
) -> list[str]:
    """Returns n colors from a colormap in hex format.

    Args:
        n (int): Number of colors to return.
        cmap (str): Name of the colormap to use (from cmasher).
        cmap_range (tuple): Fractional range of the colormap to sample
            the colors from (between 0 and 1).

    Returns:
        n_colors (list): List of n colors in hex format.
    """
    try:
        import cmasher as cmr
    except ModuleNotFoundError as err:
        warnings.warn(
            "cmasher is required for this plotting style.\n"
            "Install it with:\n\n"
            "    pip install cmasher\n",
            category=ImportWarning,
            stacklevel=2,
        )
        raise err

    return cmr.take_cmap_colors(
        cmap, n, cmap_range=cmap_range, return_fmt="hex")
