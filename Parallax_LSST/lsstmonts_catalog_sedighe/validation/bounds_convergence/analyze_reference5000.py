#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE5000 = {
    "u0": (-5.0, 5.0),
    "tE": (0.1, 5000.0),
    "rho": (1.0e-7, 5.0),
}

HISTORICAL = {
    "u0": (-5.0, 5.0),
    "tE": (0.1, 20000.0),
    "rho": (1.0e-7, 10.0),
}


def path_for(
    root,
    profile,
    sample,
    row,
):

    if profile == "historical":

        return (
            root
            / "refits"
            / sample
            / str(row)
            / "all_refits.csv"
        )

    return (
        root
        / "refits"
        / "bounds_convergence"
        / profile
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

    path = path_for(
        root,
        profile,
        sample,
        row,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    x = df[
        (df["hypothesis"] == "H0")
        & (df["status"] == "success")
    ].copy()

    x["chi2"] = pd.to_numeric(
        x["chi2"],
        errors="coerce",
    )

    x = x[
        np.isfinite(x["chi2"])
    ].copy()

    if len(x) == 0:
        raise RuntimeError(
            f"No successful H0 fits: {path}"
        )

    x["source_profile"] = profile

    return x


def feasible(df, domain):

    mask = np.ones(
        len(df),
        dtype=bool,
    )

    for p, (lo, hi) in domain.items():

        v = pd.to_numeric(
            df[p],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        mask &= (
            np.isfinite(v)
            & (v >= lo)
            & (v <= hi)
        )

    return mask


def winner(df):

    return df.loc[
        df["chi2"].idxmin()
    ]


def boundary_flags(r):

    flags = []

    u0 = float(r["u0"])
    tE = float(r["tE"])
    rho = float(r["rho"])

    # 0.1% of the linear u0 range.
    if (
        abs(u0 + 5.0) <= 0.01
        or abs(u0 - 5.0) <= 0.01
    ):
        flags.append("u0")

    # Only an upper-bound tE flag is scientifically relevant here.
    if tE >= 0.99 * 5000.0:
        flags.append("tE_upper")

    if tE <= 0.101:
        flags.append("tE_lower")

    if rho >= 0.99 * 5.0:
        flags.append("rho_upper")

    if rho <= 1.01e-7:
        flags.append("rho_lower")

    return ";".join(flags)


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

    args = parser.parse_args()

    root = args.work_root.expanduser().resolve()

    manifest = pd.read_csv(
        args.manifest
    )

    rows = []

    for _, m in manifest.iterrows():

        row = int(
            m["catalog_row"]
        )

        sample = str(
            m["sample"]
        )

        pieces = {}

        for profile in [
            "historical",
            "candidate",
            "stress",
            "reference5000",
        ]:

            pieces[profile] = load_h0(
                root,
                profile,
                sample,
                row,
            )

        # ---------------------------------------------
        # Actual minimum found by the reference5000 run
        # ---------------------------------------------

        ref_actual = winner(
            pieces["reference5000"]
        )

        # ---------------------------------------------
        # Best solution already known BEFORE
        # reference5000, but feasible in reference5000
        # ---------------------------------------------

        pre = pd.concat(
            [
                pieces["historical"],
                pieces["candidate"],
                pieces["stress"],
            ],
            ignore_index=True,
        )

        pre_ref = pre[
            feasible(
                pre,
                REFERENCE5000,
            )
        ].copy()

        if len(pre_ref) == 0:
            raise RuntimeError(
                f"No pre-existing reference5000-feasible "
                f"solution for row {row}"
            )

        pre_ref_best = winner(
            pre_ref
        )

        # ---------------------------------------------
        # Pool every known solution.
        # ---------------------------------------------

        allsol = pd.concat(
            list(
                pieces.values()
            ),
            ignore_index=True,
        )

        pooled_ref = allsol[
            feasible(
                allsol,
                REFERENCE5000,
            )
        ].copy()

        pooled_hist = allsol[
            feasible(
                allsol,
                HISTORICAL,
            )
        ].copy()

        best_ref = winner(
            pooled_ref
        )

        best_hist = winner(
            pooled_hist
        )

        rows.append(
            {
                "catalog_row":
                    row,

                "sample":
                    sample,

                "chi2_reference5000_refit":
                    float(
                        ref_actual["chi2"]
                    ),

                "chi2_best_preexisting_inside_reference5000":
                    float(
                        pre_ref_best["chi2"]
                    ),

                "delta_refit_minus_preexisting":
                    float(
                        ref_actual["chi2"]
                        - pre_ref_best["chi2"]
                    ),

                "preexisting_source":
                    str(
                        pre_ref_best[
                            "source_profile"
                        ]
                    ),

                "chi2_pooled_reference5000":
                    float(
                        best_ref["chi2"]
                    ),

                "chi2_pooled_historical_domain":
                    float(
                        best_hist["chi2"]
                    ),

                "delta_reference5000_minus_historical":
                    float(
                        best_ref["chi2"]
                        - best_hist["chi2"]
                    ),

                "best_reference_source":
                    str(
                        best_ref[
                            "source_profile"
                        ]
                    ),

                "best_historical_source":
                    str(
                        best_hist[
                            "source_profile"
                        ]
                    ),

                "best_reference_t0":
                    float(
                        best_ref["t0"]
                    ),

                "best_reference_u0":
                    float(
                        best_ref["u0"]
                    ),

                "best_reference_tE":
                    float(
                        best_ref["tE"]
                    ),

                "best_reference_rho":
                    float(
                        best_ref["rho"]
                    ),

                "reference_bound_flags":
                    boundary_flags(
                        best_ref
                    ),
            }
        )

    out = pd.DataFrame(rows)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        args.output,
        index=False,
    )

    txt = (
        args.output.with_suffix(
            ".txt"
        )
    )

    dopt = (
        out[
            "delta_refit_minus_preexisting"
        ]
    )

    dbounds = (
        out[
            "delta_reference5000_minus_historical"
        ]
    )

    lines = [
        f"N = {len(out)}",
        "",
        "OPTIMIZER / CROSS-SEED CHECK",
        (
            "N reference5000 refit worse than "
            f"known feasible minimum by >0.1 = "
            f"{int((dopt > 0.1).sum())}"
        ),
        (
            "max(refit - preexisting) = "
            f"{dopt.max():.9g}"
        ),
        "",
        "DOMAIN CHECK",
        (
            "N with known better solution outside "
            f"reference5000 by >0.1 = "
            f"{int((dbounds > 0.1).sum())}"
        ),
        (
            "max(reference5000 - historical) = "
            f"{dbounds.max():.9g}"
        ),
        "",
        (
            "N pooled reference winners at/near "
            f"a physical bound = "
            f"{int((out['reference_bound_flags'] != '').sum())}"
        ),
    ]

    txt.write_text(
        "\n".join(lines)
        + "\n"
    )

    print()
    print("\n".join(lines))

    print()
    print("=" * 100)
    print("Largest optimizer discrepancies")
    print("=" * 100)

    print(
        out.sort_values(
            "delta_refit_minus_preexisting",
            ascending=False,
        )
        .head(15)[
            [
                "catalog_row",
                "sample",
                "delta_refit_minus_preexisting",
                "preexisting_source",
                "best_reference_tE",
                "best_reference_rho",
                "reference_bound_flags",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print()
    print("=" * 100)
    print("Largest domain discrepancies")
    print("=" * 100)

    print(
        out.sort_values(
            "delta_reference5000_minus_historical",
            ascending=False,
        )
        .head(15)[
            [
                "catalog_row",
                "sample",
                "delta_reference5000_minus_historical",
                "best_reference_source",
                "best_historical_source",
                "best_reference_u0",
                "best_reference_tE",
                "best_reference_rho",
                "reference_bound_flags",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print()
    print("saved =", args.output)
    print("saved =", txt)


if __name__ == "__main__":
    main()
