#!/usr/bin/env python3

import argparse
import shutil
import tarfile
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax"
)

RUN_ROOT = (
    ROOT
    / "runs"
    / "LSSTMONTS_LRT_population_FAST"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", required=True)
    p.add_argument("--workers", type=int, default=10)
    return p.parse_args()


def archive_member_from_row(row):

    model_dir = Path(str(row["model_dir"]))

    if "models" not in model_dir.parts:
        raise RuntimeError(
            f"'models' not found in model_dir: {model_dir}"
        )

    i = model_dir.parts.index("models")

    relative_model_dir = Path(
        *model_dir.parts[i:]
    )

    simulation_seed = int(
        row["simulation_seed"]
    )

    return str(
        relative_model_dir
        / f"Event_{simulation_seed}.h5"
    )


def main():

    args = parse_args()

    analysis_dir = (
        ROOT
        / "analysis"
        / args.run_tag
    )

    export_dir = (
        analysis_dir
        / "local_export"
    )

    statistics_dir = (
        export_dir
        / "statistics"
    )

    lc_dir = (
        export_dir
        / "lightcurves_h5"
    )

    selected = pd.read_csv(
        analysis_dir
        / "empirical_h0"
        / "selected_h0_events.csv"
    )

    h0 = pd.read_parquet(
        statistics_dir
        / "h0_ok_full.parquet"
    )

    # Remove the previous heuristic extraction.
    if lc_dir.exists():
        shutil.rmtree(lc_dir)

    lc_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = []

    for _, sel in selected.iterrows():

        label = str(sel["label"])
        catalog_row = int(sel["catalog_row"])
        start = int(sel["chunk_start"])
        stop = int(sel["chunk_stop"])
        T = float(sel["T"])

        rows = h0[
            h0["catalog_row"].eq(
                catalog_row
            )
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"{label}: catalog_row={catalog_row} "
                f"has {len(rows)} matching H0 rows"
            )

        row = rows.iloc[0]

        simulation_seed = int(
            row["simulation_seed"]
        )

        noise_realization_id = int(
            row["noise_realization_id"]
        )

        expected_event_dir = (
            f"event_{catalog_row:07d}"
            f"_h0_r{noise_realization_id:06d}"
        )

        actual_event_dir = Path(
            str(row["model_dir"])
        ).name

        if actual_event_dir != expected_event_dir:
            raise RuntimeError(
                f"{label}: event directory mismatch\n"
                f"expected={expected_event_dir}\n"
                f"actual={actual_event_dir}"
            )

        expected_member = (
            archive_member_from_row(row)
        )

        run_dir = (
            RUN_ROOT
            / (
                f"{args.run_tag}_rows_"
                f"{start}_{stop}_w{args.workers}"
            )
        )

        archive = (
            run_dir
            / "artifacts"
            / f"artifacts_rows_{start}_{stop}.tar"
        )

        if not archive.is_file():
            raise FileNotFoundError(
                archive
            )

        with tarfile.open(
            archive,
            "r"
        ) as tf:

            matches = [
                m
                for m in tf.getmembers()
                if (
                    m.isfile()
                    and m.name.lstrip("./")
                    == expected_member
                )
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    f"{label}: expected exactly one match for\n"
                    f"{expected_member}\n"
                    f"found={len(matches)}"
                )

            member = matches[0]

            source = tf.extractfile(
                member
            )

            if source is None:
                raise RuntimeError(
                    f"Could not extract {member.name}"
                )

            target = (
                lc_dir
                / (
                    f"{label}_"
                    f"row{catalog_row}_"
                    f"Event_{simulation_seed}.h5"
                )
            )

            with target.open("wb") as f:
                shutil.copyfileobj(
                    source,
                    f
                )

        manifest.append({
            "label": label,
            "catalog_row": catalog_row,
            "T": T,
            "simulation_seed": simulation_seed,
            "noise_realization_id": noise_realization_id,
            "field_name": row["field_name"],
            "chunk_start": start,
            "chunk_stop": stop,
            "archive_member": expected_member,
            "local_filename": target.name,
            "match_method": "exact_model_dir_plus_simulation_seed",
            "match_count": 1,
        })

        print(
            f"{label:8s} "
            f"row={catalog_row:6d} "
            f"seed={simulation_seed} "
            f"T={T:.6f}"
        )

        print(
            "   ",
            expected_member
        )

    manifest = pd.DataFrame(
        manifest
    )

    manifest.to_csv(
        export_dir
        / "lightcurve_manifest.csv",
        index=False
    )

    print()
    print("=" * 72)
    print("EXACT EXTRACTION COMPLETE")
    print("=" * 72)
    print(
        manifest[
            [
                "label",
                "catalog_row",
                "simulation_seed",
                "T",
                "match_count",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
