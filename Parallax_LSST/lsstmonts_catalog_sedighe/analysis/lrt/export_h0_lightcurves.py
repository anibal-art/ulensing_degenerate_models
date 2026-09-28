#!/usr/bin/env python3

import argparse
import re
import tarfile
from pathlib import Path

import h5py

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "runs/LSSTMONTS_LRT_population_FAST"
)

ANALYSIS_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "analysis"
)


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--run-tag",
        required=True,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=10,
    )

    return p.parse_args()


def score_member(
    name,
    row,
    catalog_row,
):

    if not name.lower().endswith(
        ".h5"
    ):
        return -1

    score = 0

    low = name.lower()

    if "/models/" in low:
        score += 10

    if (
        "/h0/" in low
        or "_h0" in low
        or "truth_h0" in low
    ):
        score += 20

    if "path_event" in row.index:

        path_event = row["path_event"]

        if isinstance(
            path_event,
            str,
        ):

            basename = Path(
                path_event
            ).name

            stem = Path(
                path_event
            ).stem

            if basename in name:
                score += 100

            elif stem in name:
                score += 80

            for token in re.findall(
                r"Event_[A-Za-z0-9_.-]+",
                path_event,
            ):

                token = token.rsplit(
                    ".",
                    1,
                )[0]

                if token in name:
                    score += 120

    for value in row.values:

        if not isinstance(
            value,
            str,
        ):
            continue

        for token in re.findall(
            r"Event_[A-Za-z0-9_.-]+",
            value,
        ):

            token = token.rsplit(
                ".",
                1,
            )[0]

            if token in name:
                score += 100

    if re.search(
        rf"(?<!\d){catalog_row}(?!\d)",
        name,
    ):
        score += 30

    return score


def find_lightcurve_groups(h5):

    groups = []

    def visit(name, obj):

        if not isinstance(
            obj,
            h5py.Group,
        ):
            return

        keys = set(
            obj.keys()
        )

        if (
            "time" in keys
            and (
                "mag" in keys
                or "flux" in keys
            )
        ):
            groups.append(
                (
                    name,
                    obj,
                )
            )

    h5.visititems(visit)

    return groups


def plot_h5(
    h5_path,
    png,
    pdf,
    title,
):

    with h5py.File(
        h5_path,
        "r",
    ) as h5:

        groups = (
            find_lightcurve_groups(h5)
        )

        if not groups:
            raise RuntimeError(
                "No light-curve groups found"
            )

        n = len(groups)

        fig, axes = plt.subplots(
            n,
            1,
            figsize=(
                9,
                max(
                    3,
                    2.6 * n,
                ),
            ),
            sharex=True,
        )

        if n == 1:
            axes = [axes]

        xoffset = None

        for ax, (
            name,
            group,
        ) in zip(
            axes,
            groups,
        ):

            time = np.ravel(
                np.asarray(
                    group["time"],
                    dtype=float,
                )
            )

            if xoffset is None:

                median_time = np.nanmedian(
                    time
                )

                if median_time > 1e6:
                    xoffset = 2450000.0
                else:
                    xoffset = 0.0

            x = time - xoffset

            band_name = Path(
                name
            ).name

            if "mag" in group:

                y = np.ravel(
                    np.asarray(
                        group["mag"],
                        dtype=float,
                    )
                )

                if "err_mag" in group:

                    error = np.ravel(
                        np.asarray(
                            group["err_mag"],
                            dtype=float,
                        )
                    )

                else:
                    error = None

                good = (
                    np.isfinite(x)
                    & np.isfinite(y)
                )

                if (
                    error is not None
                    and len(error) == len(y)
                ):

                    good &= np.isfinite(
                        error
                    )

                    ax.errorbar(
                        x[good],
                        y[good],
                        yerr=error[good],
                        fmt=".",
                        ms=3,
                        alpha=0.7,
                        linewidth=0.5,
                    )

                else:

                    ax.plot(
                        x[good],
                        y[good],
                        ".",
                        ms=3,
                        alpha=0.7,
                    )

                if "mag_model" in group:

                    model = np.ravel(
                        np.asarray(
                            group[
                                "mag_model"
                            ],
                            dtype=float,
                        )
                    )

                    if len(model) == len(x):

                        model_good = (
                            np.isfinite(x)
                            & np.isfinite(
                                model
                            )
                        )

                        order = np.argsort(
                            x[model_good]
                        )

                        ax.plot(
                            x[
                                model_good
                            ][order],
                            model[
                                model_good
                            ][order],
                            linewidth=1.2,
                            label="saved model",
                        )

                        ax.legend()

                ax.set_ylabel(
                    f"{band_name} mag"
                )

                ax.invert_yaxis()

            else:

                y = np.ravel(
                    np.asarray(
                        group["flux"],
                        dtype=float,
                    )
                )

                good = (
                    np.isfinite(x)
                    & np.isfinite(y)
                )

                ax.plot(
                    x[good],
                    y[good],
                    ".",
                    ms=3,
                    alpha=0.7,
                )

                ax.set_ylabel(
                    f"{band_name} flux"
                )

            ax.grid(alpha=0.2)

        if xoffset:

            axes[-1].set_xlabel(
                "JD - 2450000"
            )

        else:

            axes[-1].set_xlabel(
                "time"
            )

        fig.suptitle(title)

        fig.tight_layout()

        fig.savefig(
            png,
            dpi=200,
        )

        fig.savefig(
            pdf
        )

        plt.close(fig)


def main():

    args = parse_args()

    h0_dir = (
        ANALYSIS_ROOT
        / args.run_tag
        / "empirical_h0"
    )

    selected = pd.read_csv(
        h0_dir
        / "selected_h0_events.csv"
    )

    export_dir = (
        ANALYSIS_ROOT
        / args.run_tag
        / "export_h0"
    )

    lightcurve_dir = (
        export_dir
        / "lightcurves"
    )

    raw_dir = (
        export_dir
        / "raw_h5"
    )

    lightcurve_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    for _, selected_row in (
        selected.iterrows()
    ):

        label = str(
            selected_row["label"]
        )

        catalog_row = int(
            selected_row[
                "catalog_row"
            ]
        )

        start = int(
            selected_row[
                "chunk_start"
            ]
        )

        stop = int(
            selected_row[
                "chunk_stop"
            ]
        )

        T = float(
            selected_row["T"]
        )

        run_dir = (
            RUN_ROOT
            / (
                f"{args.run_tag}_rows_"
                f"{start}_{stop}_"
                f"w{args.workers}"
            )
        )

        summary_path = (
            run_dir
            / "logs"
            / "run_summary.parquet"
        )

        summary = pd.read_parquet(
            summary_path
        )

        rows = summary[
            (
                summary[
                    "catalog_row"
                ]
                == catalog_row
            )
            & (
                summary[
                    "truth_case"
                ]
                == "H0"
            )
        ]

        if len(rows) != 1:

            print(
                "WARNING:",
                label,
                "summary matches =",
                len(rows),
            )

            continue

        row = rows.iloc[0]

        tar_path = (
            run_dir
            / "artifacts"
            / (
                f"artifacts_rows_"
                f"{start}_{stop}.tar"
            )
        )

        if not tar_path.is_file():

            print(
                "WARNING missing:",
                tar_path,
            )

            continue

        with tarfile.open(
            tar_path,
            "r",
        ) as tf:

            members = [
                member
                for member
                in tf.getmembers()
                if (
                    member.isfile()
                    and member.name
                    .lower()
                    .endswith(".h5")
                )
            ]

            scored = [
                (
                    score_member(
                        member.name,
                        row,
                        catalog_row,
                    ),
                    member,
                )
                for member in members
            ]

            scored.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            if (
                not scored
                or scored[0][0] <= 0
            ):

                print(
                    "WARNING: no reliable H5",
                    label,
                    catalog_row,
                )

                continue

            score, member = scored[0]

            source = tf.extractfile(
                member
            )

            if source is None:
                continue

            raw_path = (
                raw_dir
                / (
                    f"{label}_"
                    f"row{catalog_row}_"
                    f"{Path(member.name).name}"
                )
            )

            raw_path.write_bytes(
                source.read()
            )

        png = (
            lightcurve_dir
            / (
                f"{label}_"
                f"row{catalog_row}_"
                f"T{T:.3f}.png"
            )
        )

        pdf = (
            lightcurve_dir
            / (
                f"{label}_"
                f"row{catalog_row}_"
                f"T{T:.3f}.pdf"
            )
        )

        try:

            plot_h5(
                raw_path,
                png,
                pdf,
                (
                    f"H0 {label}: "
                    f"catalog row "
                    f"{catalog_row}, "
                    f"T={T:.3f}"
                ),
            )

            plot_status = "ok"

        except Exception as exc:

            plot_status = (
                f"plot_failed: {exc}"
            )

            print(
                "WARNING:",
                plot_status,
            )

        manifest.append({
            "label": label,
            "catalog_row": catalog_row,
            "T": T,
            "chunk_start": start,
            "chunk_stop": stop,
            "archive_member": (
                member.name
            ),
            "match_score": score,
            "raw_h5": str(
                raw_path
            ),
            "plot_status": (
                plot_status
            ),
        })

    manifest = pd.DataFrame(
        manifest
    )

    manifest.to_csv(
        export_dir
        / "lightcurve_export_manifest.csv",
        index=False,
    )

    print()
    print("EXPORT MANIFEST")
    print(
        manifest.to_string(
            index=False
        )
    )

    print()
    print("Export directory:")
    print(export_dir)


if __name__ == "__main__":
    main()
