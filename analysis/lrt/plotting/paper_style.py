#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared publication style for the LRT analysis figures.

This module changes presentation only.  It does not modify any
scientific selection, binning, statistic, fit result, or data product.

The target is a journal-ready, double-column figure set:
    * consistent typography and line weights;
    * constrained layout to avoid clipped/overlapping labels;
    * vector PDF plus high-resolution PNG output;
    * compact panel labels for multi-panel figures.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np


# Widths are chosen for a typical two-column astronomy manuscript.
# LaTeX can still scale them with \includegraphics without changing
# the internal aspect ratio.
FIGSIZE_SINGLE = (6.8, 4.4)
FIGSIZE_SQUARE = (5.4, 4.8)
FIGSIZE_HEATMAP = (7.2, 5.2)
FIGSIZE_STACKED_2 = (6.8, 5.8)
FIGSIZE_STACKED_3 = (6.8, 7.8)
FIGSIZE_TWO_PANEL = (7.2, 3.5)
FIGSIZE_TWO_PANEL_TALL = (7.2, 4.2)
FIGSIZE_GRID_2X4 = (7.2, 5.0)

PNG_DPI = 300

# Colorblind-safe qualitative cycle (Okabe-Ito).
_COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
]


def apply_paper_style() -> None:
    """Apply the common non-LaTeX Matplotlib publication style."""

    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.035,
            "figure.constrained_layout.w_pad": 0.035,
            "figure.constrained_layout.hspace": 0.035,
            "figure.constrained_layout.wspace": 0.035,

            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",

            "font.size": 16,
            "axes.labelsize": 19,
            "axes.titlesize": 18,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.8,

            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",

            "legend.fontsize": 13,
            "legend.frameon": False,
            "legend.handlelength": 1.8,

            "lines.linewidth": 1.3,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.8,

            "grid.linewidth": 0.6,
            "grid.alpha": 0.20,

            "axes.prop_cycle": cycler(color=_COLOR_CYCLE),

            "image.cmap": "viridis",

            "savefig.dpi": PNG_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _flatten_axes(axes):
    """Return a flat list of Axes from a scalar/list/ndarray container."""

    if axes is None:
        return []

    if hasattr(axes, "plot"):
        return [axes]

    arr = np.asarray(axes, dtype=object).reshape(-1)
    return [ax for ax in arr if hasattr(ax, "plot")]


def label_panels(
    axes,
    labels=None,
    *,
    x: float = 0.01,
    y: float = 0.99,
    fontsize: float = 9.0,
) -> None:
    """Add compact (a), (b), ... labels inside each panel."""

    flat = _flatten_axes(axes)

    if labels is None:
        labels = [
            f"({chr(ord('a') + i)})"
            for i in range(len(flat))
        ]

    for ax, label in zip(flat, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            zorder=20,
        )


def save_pdf_companion(
    fig,
    output_path,
    *,
    dpi: int = PNG_DPI,
) -> Path:
    """
    Save a vector PDF alongside an existing raster output path.

    The original scripts keep their PNG filenames for backwards
    compatibility.  This helper adds the paper-quality PDF with the
    same stem.
    """

    output_path = Path(output_path)
    pdf_path = output_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Align shared labels before the final render.
    try:
        fig.align_labels()
    except Exception:
        pass

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    return pdf_path


def save_figure(
    fig,
    output_path,
    *,
    dpi: int = PNG_DPI,
) -> tuple[Path, Path]:
    """
    Save one scientific figure as both PNG (300 dpi) and vector PDF.

    output_path may be a path with any suffix or a suffix-free stem.
    """

    output_path = Path(output_path)
    stem = output_path.with_suffix("") if output_path.suffix else output_path

    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")

    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.align_labels()
    except Exception:
        pass

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    return png_path, pdf_path


# Apply immediately when the module is imported.  Explicit calls to
# apply_paper_style() remain harmless and make intent clear in scripts.
apply_paper_style()
