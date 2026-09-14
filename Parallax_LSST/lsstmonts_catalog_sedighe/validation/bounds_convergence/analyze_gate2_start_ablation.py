#!/usr/bin/env python3

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


ROOT4 = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/"
    "gate2_controlled4_extreme100"
).expanduser()

ROOTOLD = Path(
    "~/Downloads/hidden_parallax/"
    "production_validation/"
    "gate2_oldfinal_extreme100"
).expanduser()

H0_FILE = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/data/"
    "gate2_h0_anchor_manifest_extreme100.csv"
)

OUT_EVENT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate2_start_ablation_per_event.csv"
)

OUT_SUMMARY = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results/"
    "gate2_start_ablation_summary.csv"
)


STARTS = [
    "truth",
    "H0_NESTED_piE_0",
    "truth_half_piE",
    "truth_mirror_u0_piEN",
    "old_final_reseed",
]


h0 = pd.read_csv(H0_FILE)

records = []


def read_unique(root, row):

    matches = list(
        root.rglob(
            f"extreme100/{row}/all_refits.csv"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"row={row}, root={root}: "
            f"found {len(matches)} all_refits.csv"
        )

    return pd.read_csv(matches[0])


for _, r0 in h0.iterrows():

    row = int(r0["catalog_row"])
    chi2_h0 = float(r0["chi2_h0"])

    d4 = read_unique(
        ROOT4,
        row,
    )

    dold = read_unique(
        ROOTOLD,
        row,
    )

    d4 = d4[
        (d4["hypothesis"] == "H1")
        & (d4["status"] == "success")
    ].copy()

    dold = dold[
        (dold["hypothesis"] == "H1")
        & (dold["status"] == "success")
    ].copy()

    values = {
        "chi2_h0_exact":
            chi2_h0,
    }

    for label in STARTS[:4]:

        x = d4[
            d4["label"] == label
        ]

        if len(x) != 1:
            raise RuntimeError(
                f"row={row}, label={label}: "
                f"found {len(x)}"
            )

        values[label] = float(
            x.iloc[0]["chi2"]
        )

    x = dold[
        dold["label"]
        == "old_final_reseed"
    ]

    if len(x) != 1:
        raise RuntimeError(
            f"row={row}, old_final_reseed: "
            f"found {len(x)}"
        )

    values["old_final_reseed"] = float(
        x.iloc[0]["chi2"]
    )

    # Reference oracle:
    # exact embedded H0 plus all five H1 optimizations.
    oracle = min(
        [chi2_h0]
        + [
            values[s]
            for s in STARTS
        ]
    )

    values["chi2_oracle5"] = oracle

    winner_values = {
        "H0_EXACT":
            chi2_h0,
        **{
            s: values[s]
            for s in STARTS
        },
    }

    values["oracle_winner"] = min(
        winner_values,
        key=winner_values.get,
    )

    values["catalog_row"] = row

    records.append(values)


events = pd.DataFrame(records)

summary = []


for k in range(
    1,
    len(STARTS) + 1,
):

    for combo in combinations(
        STARTS,
        k,
    ):

        # H0 exact is always available at zero additional
        # H1-fit cost because H0 is already fitted.
        candidate = events[
            ["chi2_h0_exact", *combo]
        ].min(
            axis=1
        )

        delta = (
            candidate
            - events["chi2_oracle5"]
        )

        summary.append(
            {
                "n_h1_fits":
                    k,

                "starts":
                    " + ".join(combo),

                "n_delta_gt_0p1":
                    int(
                        (delta > 0.1).sum()
                    ),

                "n_delta_gt_1":
                    int(
                        (delta > 1.0).sum()
                    ),

                "n_delta_gt_10":
                    int(
                        (delta > 10.0).sum()
                    ),

                "max_delta":
                    float(delta.max()),

                "p50_delta":
                    float(
                        delta.quantile(0.50)
                    ),

                "p90_delta":
                    float(
                        delta.quantile(0.90)
                    ),

                "p95_delta":
                    float(
                        delta.quantile(0.95)
                    ),

                "p99_delta":
                    float(
                        delta.quantile(0.99)
                    ),
            }
        )


summary = pd.DataFrame(summary)

summary = summary.sort_values(
    [
        "n_h1_fits",
        "n_delta_gt_0p1",
        "n_delta_gt_1",
        "max_delta",
    ]
)


OUT_EVENT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

events.to_csv(
    OUT_EVENT,
    index=False,
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


print()
print("=" * 100)
print("ORACLE WINNERS")
print("=" * 100)

print(
    events["oracle_winner"]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("BEST FIXED STRATEGY FOR EACH NUMBER OF H1 FITS")
print("=" * 100)

best = (
    summary
    .groupby(
        "n_h1_fits",
        as_index=False,
    )
    .first()
)

print(
    best.to_string(
        index=False,
        float_format=lambda x: f"{x:.10g}",
    )
)


print()
print("=" * 100)
print("ALL 1–3 FIT STRATEGIES")
print("=" * 100)

print(
    summary[
        summary["n_h1_fits"] <= 3
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.10g}",
    )
)


print()
print("saved:")
print(OUT_EVENT)
print(OUT_SUMMARY)
