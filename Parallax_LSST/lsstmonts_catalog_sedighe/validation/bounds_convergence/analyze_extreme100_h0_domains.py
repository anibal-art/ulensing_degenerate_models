#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = {
    "u0": (-5.0, 5.0),
    "tE": (0.1, 5000.0),
    "rho": (1.0e-7, 5.0),
}

CONTROL = {
    "u0": (-10.0, 10.0),
    "tE": (0.1, 20000.0),
    "rho": (1.0e-7, 10.0),
}

TOL_CHI2 = 0.1


def fit_path(
    root,
    profile,
    sample,
    row,
):

    return (
        root
        / "refits"
        / "bounds_convergence"
        / f"{profile}_h0only"
        / sample
        / str(row)
        / "all_refits.csv"
    )


def load_h0(
    root,
    profile,
    sample,
    row,
):

    path = fit_path(
        root,
        profile,
        sample,
        row,
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    x = df[
        (df["hypothesis"] == "H0")
        & (df["status"] == "success")
    ].copy()

    for c in [
        "chi2",
        "t0",
        "u0",
        "tE",
        "rho",
    ]:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    x = x[
        np.isfinite(
            x["chi2"]
        )
    ].copy()

    if len(x) == 0:
        raise RuntimeError(
            f"No successful H0 fits in {path}"
        )

    x["source_profile"] = profile
    x["source_file"] = str(path)

    return x


def feasible(
    df,
    domain,
):

    mask = np.ones(
        len(df),
        dtype=bool,
    )

    for p, (lo, hi) in domain.items():

        x = pd.to_numeric(
            df[p],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        mask &= (
            np.isfinite(x)
            & (x >= lo)
            & (x <= hi)
        )

    return mask


def winner(df):

    if len(df) == 0:
        raise RuntimeError(
            "Cannot choose winner from empty dataframe."
        )

    return df.loc[
        df["chi2"].idxmin()
    ]


def is_near_reference_bound(r):

    flags = []

    u0 = float(r["u0"])
    tE = float(r["tE"])
    rho = float(r["rho"])

    if abs(u0 + 5.0) <= 0.01:
        flags.append("u0_lower")

    if abs(u0 - 5.0) <= 0.01:
        flags.append("u0_upper")

    if tE <= 0.101:
        flags.append("tE_lower")

    if tE >= 0.99 * 5000.0:
        flags.append("tE_upper")

    if rho <= 1.01e-7:
        flags.append("rho_lower")

    if rho >= 0.99 * 5.0:
        flags.append("rho_upper")

    return ";".join(flags)


def make_crossseed_record(
    catalog_row,
    sample,
    row,
):

    return {
        "catalog_row":
            int(catalog_row),

        "sample":
            str(sample),

        "source_profile":
            str(
                row["source_profile"]
            ),

        "source_label":
            str(
                row["label"]
            ),

        "source_chi2":
            float(
                row["chi2"]
            ),

        "t0":
            float(
                row["t0"]
            ),

        "u0":
            float(
                row["u0"]
            ),

        "tE":
            float(
                row["tE"]
            ),

        "rho":
            float(
                row["rho"]
            ),

        "source_file":
            str(
                row["source_file"]
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--crossseed-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    root = (
        args.work_root
        .expanduser()
        .resolve()
    )

    manifest = pd.read_csv(
        args.manifest
    )

    rows = []
    crossseeds = []

    for _, m in manifest.iterrows():

        catalog_row = int(
            m["catalog_row"]
        )

        sample = str(
            m["sample"]
        )

        reason = str(
            m[
                "selection_reason_bounds"
            ]
        )

        ref = load_h0(
            root,
            "reference5000",
            sample,
            catalog_row,
        )

        ctrl = load_h0(
            root,
            "control20000",
            sample,
            catalog_row,
        )

        raw_ref = winner(
            ref
        )

        raw_ctrl = winner(
            ctrl
        )

        allsol = pd.concat(
            [
                ref,
                ctrl,
            ],
            ignore_index=True,
        )

        pooled_ref_df = allsol[
            feasible(
                allsol,
                REFERENCE,
            )
        ].copy()

        pooled_ctrl_df = allsol[
            feasible(
                allsol,
                CONTROL,
            )
        ].copy()

        pooled_ref = winner(
            pooled_ref_df
        )

        pooled_ctrl = winner(
            pooled_ctrl_df
        )

        raw_delta = (
            float(raw_ref["chi2"])
            - float(raw_ctrl["chi2"])
        )

        pooled_delta = (
            float(pooled_ref["chi2"])
            - float(pooled_ctrl["chi2"])
        )

        ref_optimizer_gap = (
            float(raw_ref["chi2"])
            - float(pooled_ref["chi2"])
        )

        ctrl_optimizer_gap = (
            float(raw_ctrl["chi2"])
            - float(pooled_ctrl["chi2"])
        )

        ctrl_best_inside_ref = bool(
            feasible(
                pd.DataFrame(
                    [pooled_ctrl]
                ),
                REFERENCE,
            )[0]
        )

        needs_closure = bool(
            (
                abs(raw_delta)
                > TOL_CHI2
            )
            or (
                pooled_delta
                > TOL_CHI2
            )
            or (
                ref_optimizer_gap
                > TOL_CHI2
            )
            or (
                ctrl_optimizer_gap
                > TOL_CHI2
            )
        )

        rows.append(
            {
                "catalog_row":
                    catalog_row,

                "sample":
                    sample,

                "selection_reason_bounds":
                    reason,

                "chi2_raw_reference":
                    float(
                        raw_ref["chi2"]
                    ),

                "chi2_raw_control":
                    float(
                        raw_ctrl["chi2"]
                    ),

                "delta_raw_ref_minus_control":
                    raw_delta,

                "chi2_pooled_reference":
                    float(
                        pooled_ref["chi2"]
                    ),

                "chi2_pooled_control":
                    float(
                        pooled_ctrl["chi2"]
                    ),

                "delta_pooled_ref_minus_control":
                    pooled_delta,

                "reference_optimizer_gap":
                    ref_optimizer_gap,

                "control_optimizer_gap":
                    ctrl_optimizer_gap,

                "pooled_control_inside_reference":
                    ctrl_best_inside_ref,

                "pooled_reference_source":
                    str(
                        pooled_ref[
                            "source_profile"
                        ]
                    ),

                "pooled_control_source":
                    str(
                        pooled_ctrl[
                            "source_profile"
                        ]
                    ),

                "reference_t0":
                    float(
                        pooled_ref["t0"]
                    ),

                "reference_u0":
                    float(
                        pooled_ref["u0"]
                    ),

                "reference_tE":
                    float(
                        pooled_ref["tE"]
                    ),

                "reference_rho":
                    float(
                        pooled_ref["rho"]
                    ),

                "control_t0":
                    float(
                        pooled_ctrl["t0"]
                    ),

                "control_u0":
                    float(
                        pooled_ctrl["u0"]
                    ),

                "control_tE":
                    float(
                        pooled_ctrl["tE"]
                    ),

                "control_rho":
                    float(
                        pooled_ctrl["rho"]
                    ),

                "reference_bound_flags":
                    is_near_reference_bound(
                        pooled_ref
                    ),

                "needs_crossseed_closure":
                    needs_closure,
            }
        )

        if needs_closure:

            # Add both pooled winners.
            # Each profile will independently discard seeds
            # lying outside its own domain.
            for r in [
                pooled_ref,
                pooled_ctrl,
            ]:

                crossseeds.append(
                    make_crossseed_record(
                        catalog_row,
                        sample,
                        r,
                    )
                )

    out = pd.DataFrame(
        rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        args.output,
        index=False,
    )

    # --------------------------------------------------------
    # Deduplicate cross-seeds.
    # --------------------------------------------------------

    columns = [
        "catalog_row",
        "sample",
        "source_profile",
        "source_label",
        "source_chi2",
        "t0",
        "u0",
        "tE",
        "rho",
        "source_file",
    ]

    if crossseeds:

        cs = pd.DataFrame(
            crossseeds
        )

        cs["_key"] = list(
            zip(
                cs["catalog_row"].astype(int),
                cs["t0"].round(5),
                cs["u0"].round(7),
                cs["tE"].round(5),
                cs["rho"].round(9),
            )
        )

        cs = (
            cs.sort_values(
                "source_chi2"
            )
            .drop_duplicates(
                "_key"
            )
            .drop(
                columns=["_key"]
            )
        )

        cs = cs[
            columns
        ]

    else:

        cs = pd.DataFrame(
            columns=columns
        )

    args.crossseed_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cs.to_csv(
        args.crossseed_output,
        index=False,
    )

    # --------------------------------------------------------
    # Text report.
    # --------------------------------------------------------

    n = len(out)

    n_raw_inversion = int(
        (
            out[
                "delta_raw_ref_minus_control"
            ]
            < -TOL_CHI2
        ).sum()
    )

    n_domain = int(
        (
            out[
                "delta_pooled_ref_minus_control"
            ]
            > TOL_CHI2
        ).sum()
    )

    n_ref_opt = int(
        (
            out[
                "reference_optimizer_gap"
            ]
            > TOL_CHI2
        ).sum()
    )

    n_ctrl_opt = int(
        (
            out[
                "control_optimizer_gap"
            ]
            > TOL_CHI2
        ).sum()
    )

    n_boundary = int(
        (
            out[
                "reference_bound_flags"
            ].fillna("")
            != ""
        ).sum()
    )

    n_closure = int(
        out[
            "needs_crossseed_closure"
        ].sum()
    )

    lines = [
        f"N = {n}",
        f"chi2 tolerance = {TOL_CHI2}",
        "",
        "RAW NESTED-DOMAIN CHECK",
        (
            "N raw optimizer inversions "
            "(control worse than reference by >0.1) = "
            f"{n_raw_inversion}"
        ),
        (
            "min(raw ref-control) = "
            f"{out['delta_raw_ref_minus_control'].min():.9g}"
        ),
        (
            "max(raw ref-control) = "
            f"{out['delta_raw_ref_minus_control'].max():.9g}"
        ),
        "",
        "POOLED DOMAIN CHECK",
        (
            "N known reference5000 domain losses >0.1 = "
            f"{n_domain}"
        ),
        (
            "max(pooled reference-control) = "
            f"{out['delta_pooled_ref_minus_control'].max():.9g}"
        ),
        "",
        "OPTIMIZER CHECK",
        (
            "N reference5000 optimizer gaps >0.1 = "
            f"{n_ref_opt}"
        ),
        (
            "N control20000 optimizer gaps >0.1 = "
            f"{n_ctrl_opt}"
        ),
        "",
        "BOUNDARY CHECK",
        (
            "N pooled reference winners near a "
            f"reference bound = {n_boundary}"
        ),
        "",
        "CROSS-SEED CLOSURE",
        (
            "N events requiring second-pass closure = "
            f"{n_closure}"
        ),
        (
            "N cross-seeds written = "
            f"{len(cs)}"
        ),
    ]

    txt = (
        args.output.with_suffix(
            ".txt"
        )
    )

    txt.write_text(
        "\n".join(lines)
        + "\n"
    )

    print()
    print("\n".join(lines))

    print()
    print("=" * 100)
    print("LARGEST POOLED DOMAIN LOSSES")
    print("=" * 100)

    print(
        out.sort_values(
            "delta_pooled_ref_minus_control",
            ascending=False,
        )
        .head(20)[
            [
                "catalog_row",
                "selection_reason_bounds",
                "delta_pooled_ref_minus_control",
                "pooled_control_inside_reference",
                "control_u0",
                "control_tE",
                "control_rho",
                "reference_bound_flags",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.9g}",
        )
    )

    print()
    print("=" * 100)
    print("LARGEST REFERENCE OPTIMIZER GAPS")
    print("=" * 100)

    print(
        out.sort_values(
            "reference_optimizer_gap",
            ascending=False,
        )
        .head(20)[
            [
                "catalog_row",
                "selection_reason_bounds",
                "reference_optimizer_gap",
                "pooled_reference_source",
                "reference_tE",
                "reference_rho",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.9g}",
        )
    )

    print()
    print("saved =", args.output)
    print("saved =", txt)
    print(
        "cross-seeds =",
        args.crossseed_output,
    )


if __name__ == "__main__":
    main()
