from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT = Path(
    "~/Downloads/hidden_parallax/"
    "profile_likelihood_case_study/"
    "run_profile_likelihood.py"
).expanduser().resolve()

text = SCRIPT.read_text()

marker = (
    "# ============================================================\n"
    "# MAIN\n"
    "# ============================================================"
)

definitions = text.split(marker)[0]

ns = {"__name__": "profile_wide_rho"}

exec(
    compile(
        definitions,
        str(SCRIPT),
        "exec",
    ),
    ns,
)

ROOT = ns["ROOT"]

OUT = ROOT / "profile_results_wide_rho"
OUT.mkdir(parents=True, exist_ok=True)

ns["OUT"] = OUT

# ============================================================
# WIDE RHO BOUND
# ============================================================
#
# Keep the existing "relative" machinery but choose such a huge
# fraction that the explicit lower/upper clamps dominate:
#
#     1e-7 <= rho <= 10
#
# ============================================================

ns["H1_BOUNDS_BASE"]["rho"] = {
    "type": "relative",
    "frac": 1.0e6,
    "lower": 1.0e-7,
    "upper": 10.0,
    "min_width": 1.0e-7,
}

# Use the same grid as before.
S_GRID = ns["S_GRID"]

load_case = ns["load_case"]
run_branch = ns["run_branch"]
nuisance_from_best_model = ns["nuisance_from_best_model"]
local_quadratic_d2 = ns["local_quadratic_d2"]
combine_profiles = ns["combine_profiles"]


for catalog_row in [63218, 72168]:

    print("\n" + "=" * 100)
    print("CASE", catalog_row)
    print("=" * 100)

    meta = load_case(catalog_row)

    D2, _ = local_quadratic_d2(meta)

    print("stored H1 chi2 =", meta["stored_chi2"])
    print("original D2    =", D2)
    print("rho truth      =", float(meta["truth"]["rho"]))
    print("rho winner     =", float(meta["best"][3]))
    print("NEW rho bounds = [1e-7, 10]")

    # --------------------------------------------------------
    # Winner branch
    # --------------------------------------------------------

    winner_branch = run_branch(
        meta=meta,
        branch="winner_wide_rho",
        anchor_s=0.0,
        anchor_nuisance=nuisance_from_best_model(
            meta["best"]
        ),
    )

    # --------------------------------------------------------
    # Truth branch
    # --------------------------------------------------------

    truth_start = {
        "t0": float(meta["truth"]["t0"]),
        "u0": float(meta["truth"]["u0"]),
        "tE": float(meta["truth"]["tE"]),
        "rho": float(meta["truth"]["rho"]),
    }

    truth_branch = run_branch(
        meta=meta,
        branch="truth_wide_rho",
        anchor_s=1.0,
        anchor_nuisance=truth_start,
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    df, D2, _ = combine_profiles(
        meta,
        winner_branch,
        truth_branch,
    )

    case_out = OUT / str(catalog_row)
    case_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        winner_branch.values()
    ).sort_values("s").to_csv(
        case_out / "winner_branch_wide_rho.csv",
        index=False,
    )

    pd.DataFrame(
        truth_branch.values()
    ).sort_values("s").to_csv(
        case_out / "truth_branch_wide_rho.csv",
        index=False,
    )

    df.to_csv(
        case_out / "profile_wide_rho.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Compact output
    # --------------------------------------------------------

    show = df[
        np.isclose(df["s"], 0.0)
        | np.isclose(df["s"], 0.25)
        | np.isclose(df["s"], 0.50)
        | np.isclose(df["s"], 0.75)
        | np.isclose(df["s"], 1.0)
    ].copy()

    print()
    print(
        show[
            [
                "s",
                "chi2_winner_branch",
                "chi2_truth_branch",
                "chi2_profile",
                "delta_chi2_profile_from_stored_H1",
                "delta_chi2_quadratic",
                "rho_profile",
                "selected_branch",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "minimum wide-rho chi2 =",
        df["chi2_profile"].min(),
    )

    print(
        "original stored chi2  =",
        meta["stored_chi2"],
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        df["s"],
        df["delta_chi2_profile_from_stored_H1"],
        marker="o",
        markersize=3,
        label="wide-rho profile",
    )

    ax.plot(
        df["s"],
        df["delta_chi2_quadratic"],
        linestyle="--",
        label="original covariance prediction",
    )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.axvline(1, linewidth=0.8)

    ax.set_xlabel(
        r"$s$  (0 = stored winner, 1 = truth)"
    )

    ax.set_ylabel(
        r"$\Delta\chi^2$ relative to stored H1"
    )

    ax.set_title(
        f"Wide-rho profile — {catalog_row}"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        case_out / f"profile_wide_rho_{catalog_row}.png",
        dpi=220,
    )

    plt.close(fig)


print("\nDONE")
print(OUT)
