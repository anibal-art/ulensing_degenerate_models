#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE5000 = {
    "u0": (-5.0, 5.0),
    "tE": (0.1, 5000.0),
    "rho": (1.0e-7, 5.0),
}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def source_path(root, profile, sample, row):

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


def best_h0(path):

    df = pd.read_csv(path)

    x = df[
        (df["hypothesis"] == "H0")
        & (df["status"] == "success")
        & np.isfinite(
            pd.to_numeric(
                df["chi2"],
                errors="coerce",
            )
        )
    ].copy()

    if len(x) == 0:
        raise RuntimeError(
            f"No successful H0 fits in {path}"
        )

    x["chi2"] = pd.to_numeric(
        x["chi2"],
        errors="coerce",
    )

    return x.loc[
        x["chi2"].idxmin()
    ]


def inside_reference5000(row):

    for p, (lo, hi) in REFERENCE5000.items():

        v = float(row[p])

        if not np.isfinite(v):
            return False

        if not (lo <= v <= hi):
            return False

    return True


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
    manifest = pd.read_csv(args.manifest)

    records = []

    for _, m in manifest.iterrows():

        catalog_row = int(
            m["catalog_row"]
        )

        sample = str(
            m["sample"]
        )

        candidates = []

        for profile in [
            "historical",
            "candidate",
            "stress",
        ]:

            path = source_path(
                root,
                profile,
                sample,
                catalog_row,
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {profile} result: {path}"
                )

            r = best_h0(path)

            if not inside_reference5000(r):
                print(
                    "SKIP outside reference5000:",
                    catalog_row,
                    profile,
                    "u0=",
                    r["u0"],
                    "tE=",
                    r["tE"],
                    "rho=",
                    r["rho"],
                )
                continue

            candidates.append(
                {
                    "catalog_row":
                        catalog_row,

                    "sample":
                        sample,

                    "source_profile":
                        profile,

                    "source_label":
                        str(r["label"]),

                    "source_chi2":
                        float(r["chi2"]),

                    "t0":
                        float(r["t0"]),

                    "u0":
                        float(r["u0"]),

                    "tE":
                        float(r["tE"]),

                    "rho":
                        float(r["rho"]),

                    "source_file":
                        str(
                            path.relative_to(root)
                        ),

                    "source_sha256":
                        sha256(path),
                }
            )

        # Lowest chi2 first.
        candidates = sorted(
            candidates,
            key=lambda x: x["source_chi2"],
        )

        # Deduplicate numerically identical solutions.
        seen = set()

        for c in candidates:

            key = (
                round(c["t0"], 5),
                round(c["u0"], 7),
                round(c["tE"], 5),
                round(c["rho"], 9),
            )

            if key in seen:
                continue

            seen.add(key)
            records.append(c)

    out = pd.DataFrame(records)

    if len(out) == 0:
        raise RuntimeError(
            "No cross-seeds were generated."
        )

    missing = (
        set(
            manifest["catalog_row"]
            .astype(int)
        )
        - set(
            out["catalog_row"]
            .astype(int)
        )
    )

    if missing:
        raise RuntimeError(
            "Events without any usable cross-seed: "
            + repr(sorted(missing))
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        args.output,
        index=False,
    )

    print()
    print("saved =", args.output)
    print(
        "N events =",
        out["catalog_row"].nunique(),
    )
    print(
        "N cross-seeds =",
        len(out),
    )

    print()
    print(
        out.groupby(
            "source_profile"
        ).size()
    )

    print()
    print(
        out[
            [
                "catalog_row",
                "sample",
                "source_profile",
                "source_chi2",
                "u0",
                "tE",
                "rho",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.9g}",
        )
    )


if __name__ == "__main__":
    main()
