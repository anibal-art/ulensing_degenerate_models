#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection map in the (tE_true, |piE_true|) plane for the strictly positive
H1-generated population, masking cells with too few events.

This is meant to fix the presentation figure so that sparse edge cells are
not over-interpreted.

Input:
    analysis/lrt/results/characterization/18_positive_truth/
        h1_positive_truth_characterization.parquet

Output:
    analysis/lrt/figures/characterization/31_detection_map_masked_min_count/
        positive_detection_map_masked.png
    analysis/lrt/results/characterization/31_detection_map_masked_min_count/
        positive_detection_map_masked_table.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _paper_sys
from pathlib import Path as _PaperPath

_PAPER_LRT_DIR = next(
    parent
    for parent in _PaperPath(__file__).resolve().parents
    if parent.name == "lrt"
)
_paper_sys.path.insert(
    0,
    str(_PAPER_LRT_DIR / "plotting"),
)
from paper_style import (
    apply_paper_style,
    label_panels,
    save_pdf_companion,
)

apply_paper_style()


MIN_CELL_N = 30
N_TE_BINS = 9
N_PIE_BINS = 9


def geometric_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def main():
    repo = Path(__file__).resolve().parents[3]

    data_path = (
        repo
        / "analysis/lrt/results/characterization/18_positive_truth/"
        / "h1_positive_truth_characterization.parquet"
    )

    fig_dir = (
        repo
        / "analysis/lrt/figures/characterization/31_detection_map_masked_min_count"
    )
    res_dir = (
        repo
        / "analysis/lrt/results/characterization/31_detection_map_masked_min_count"
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        data_path,
        columns=[
            "tE_true_days",
            "piE_true_amp",
            "positive_lrt_class",
        ],
    )

    df = df.copy()
    df = df[
        np.isfinite(pd.to_numeric(df["tE_true_days"], errors="coerce"))
        & np.isfinite(pd.to_numeric(df["piE_true_amp"], errors="coerce"))
    ].copy()

    df["tE_true_days"] = pd.to_numeric(df["tE_true_days"], errors="coerce")
    df["piE_true_amp"] = pd.to_numeric(df["piE_true_amp"], errors="coerce")
    df = df[(df["tE_true_days"] > 0) & (df["piE_true_amp"] > 0)].copy()

    df["detected"] = df["positive_lrt_class"].eq("parallax_detected")

    # ============================================================
    # NEW BLOCK: logarithmic edges with exact data extrema
    # ============================================================
    #
    # np.logspace(log10(min), log10(max), ...) can reconstruct an
    # endpoint a few floating-point ulps inside the actual extrema.
    # Expanding that reconstructed endpoint by only one ulp is not
    # guaranteed to recover the original data value.
    #
    # Build the grid normally, but force the outer edges from the
    # actual sample extrema themselves.

    te_min = float(df["tE_true_days"].min())
    te_max = float(df["tE_true_days"].max())

    pie_min = float(df["piE_true_amp"].min())
    pie_max = float(df["piE_true_amp"].max())

    te_edges = np.geomspace(
        te_min,
        te_max,
        N_TE_BINS + 1,
    )

    pie_edges = np.geomspace(
        pie_min,
        pie_max,
        N_PIE_BINS + 1,
    )

    # Numerical guard: strictly contain every valid event.
    te_edges[0] = np.nextafter(
        te_min,
        -np.inf,
    )
    te_edges[-1] = np.nextafter(
        te_max,
        np.inf,
    )

    pie_edges[0] = np.nextafter(
        pie_min,
        -np.inf,
    )
    pie_edges[-1] = np.nextafter(
        pie_max,
        np.inf,
    )

    # ============================================================
    # NEW BLOCK: assign integer bin indices directly
    # ============================================================
    #
    # Do NOT reconstruct cells by comparing floating-point
    # Interval boundaries.  pandas may round Interval endpoints
    # for representation, which can break an isclose-based match
    # against the original logarithmic grid.
    #
    # labels=False gives the integer cell index directly:
    #
    #     te_bin_idx  = 0 ... N_TE_BINS-1
    #     pie_bin_idx = 0 ... N_PIE_BINS-1
    #
    # These indices can be used safely to fill the 2D arrays.

    df["te_bin_idx"] = pd.cut(
        df["tE_true_days"],
        bins=te_edges,
        include_lowest=True,
        labels=False,
    )

    df["pie_bin_idx"] = pd.cut(
        df["piE_true_amp"],
        bins=pie_edges,
        include_lowest=True,
        labels=False,
    )

    n_before_binning = len(df)

    df = df[
        df["te_bin_idx"].notna()
        & df["pie_bin_idx"].notna()
    ].copy()

    if len(df) != n_before_binning:
        raise RuntimeError(
            "Binning dropped valid events: "
            f"before={n_before_binning}, after={len(df)}."
        )

    df["te_bin_idx"] = df["te_bin_idx"].astype(int)
    df["pie_bin_idx"] = df["pie_bin_idx"].astype(int)


    # ============================================================
    # NEW BLOCK: fill count and detection grids directly
    # ============================================================

    detection_grid = np.full(
        (N_PIE_BINS, N_TE_BINS),
        np.nan,
        dtype=float,
    )

    count_grid = np.zeros(
        (N_PIE_BINS, N_TE_BINS),
        dtype=int,
    )

    rows = []

    grouped = df.groupby(
        ["te_bin_idx", "pie_bin_idx"],
        sort=True,
    )

    for (i, j), g in grouped:

        i = int(i)
        j = int(j)

        n = int(len(g))
        n_detect = int(g["detected"].sum())

        frac = (
            n_detect / n
            if n > 0
            else np.nan
        )

        count_grid[j, i] = n

        if n >= MIN_CELL_N:
            detection_grid[j, i] = frac

        rows.append(
            {
                "te_bin_idx": i,
                "pie_bin_idx": j,
                "te_left": float(te_edges[i]),
                "te_right": float(te_edges[i + 1]),
                "pie_left": float(pie_edges[j]),
                "pie_right": float(pie_edges[j + 1]),
                "n_total": n,
                "n_detected": n_detect,
                "detection_fraction": frac,
                "displayed": bool(n >= MIN_CELL_N),
            }
        )


    # ============================================================
    # NEW BLOCK: save cell-level table
    # ============================================================

    out_table = pd.DataFrame(rows).sort_values(
        by=[
            "pie_bin_idx",
            "te_bin_idx",
        ]
    )

    out_csv = (
        res_dir
        / "positive_detection_map_masked_table.csv"
    )

    out_table.to_csv(
        out_csv,
        index=False,
    )


    # ============================================================
    # NEW BLOCK: audit the masking
    # ============================================================

    occupied_counts = count_grid[
        count_grid > 0
    ]

    shown_counts = count_grid[
        count_grid >= MIN_CELL_N
    ]

    if shown_counts.size == 0:
        raise RuntimeError(
            "No cell satisfies the requested "
            f"MIN_CELL_N={MIN_CELL_N}."
        )

    min_shown = int(
        shown_counts.min()
    )

    n_occupied_cells = int(
        occupied_counts.size
    )

    n_displayed_cells = int(
        shown_counts.size
    )

    n_masked_occupied_cells = (
        n_occupied_cells
        - n_displayed_cells
    )

    if min_shown < MIN_CELL_N:
        raise RuntimeError(
            "Internal masking error: "
            f"min_shown={min_shown} "
            f"< MIN_CELL_N={MIN_CELL_N}."
        )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    mesh = ax.pcolormesh(
        te_edges,
        pie_edges,
        detection_grid,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Detection fraction")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"True $t_E$ [days]")
    ax.set_ylabel(r"True $|\pi_E|$")

    note = (
        f"Minimum count among displayed cells: {min_shown}\n"
        f"Masked cells: n < {MIN_CELL_N}"
    )

    ax.grid(False)

    out_fig = fig_dir / "positive_detection_map_masked.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    save_pdf_companion(fig, out_fig)
    plt.close(fig)

    print("=" * 80)
    print("MASKED DETECTION MAP")
    print("=" * 80)
    print(f"rows = {len(df):,}")
    print(f"minimum display count = {MIN_CELL_N}")
    print(f"minimum count among displayed cells = {min_shown}")
    print(f"Saved figure: {out_fig}")
    print(f"Saved table : {out_csv}")


if __name__ == "__main__":
    main()


