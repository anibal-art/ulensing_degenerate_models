#!/usr/bin/env python3

import argparse
import re
import shutil
import tarfile
from pathlib import Path

import pandas as pd


RUN_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "runs/LSSTMONTS_LRT_population_FAST"
)

DONE_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "slurm_done"
)

ANALYSIS_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "analysis"
)

PRODUCTION_CONFIG_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/hidden_parallax/"
    "production_configs"
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", required=True)
    p.add_argument("--workers", type=int, default=10)
    return p.parse_args()


def load_done(run_tag, workers):

    done_dir = DONE_ROOT / run_tag

    frames = []

    for marker in sorted(done_dir.glob("rows_*.DONE")):

        m = re.fullmatch(
            r"rows_(\d+)_(\d+)\.DONE",
            marker.name
        )

        if not m:
            continue

        start, stop = map(int, m.groups())

        summary = (
            RUN_ROOT
            / f"{run_tag}_rows_{start}_{stop}_w{workers}"
            / "logs"
            / "run_summary.parquet"
        )

        if not summary.exists():
            continue

        df = pd.read_parquet(summary)

        df["_chunk_start"] = start
        df["_chunk_stop"] = stop

        frames.append(df)

    if not frames:
        raise RuntimeError("No DONE chunks found")

    return pd.concat(
        frames,
        ignore_index=True
    )


def member_score(member_name, row):

    name = member_name.lower()

    if not name.endswith(".h5"):
        return -1

    score = 0

    if "/models/" in name:
        score += 20

    if (
        "/h0/" in name
        or "_h0_" in name
        or "truth_h0" in name
        or "/h0_" in name
    ):
        score += 100

    strings = [
        str(v)
        for v in row.values
        if isinstance(v, str)
    ]

    for value in strings:

        for token in re.findall(
            r"Event_[A-Za-z0-9_.-]+",
            value
        ):

            token = token.rsplit(".", 1)[0]

            if token.lower() in name:
                score += 1000

        base = Path(value).name

        if base.lower().endswith(".h5"):
            if base.lower() in name:
                score += 2000

    return score


def main():

    args = get_args()

    source_analysis = (
        ANALYSIS_ROOT
        / args.run_tag
        / "empirical_h0"
    )

    export_dir = (
        ANALYSIS_ROOT
        / args.run_tag
        / "local_export"
    )

    if export_dir.exists():
        shutil.rmtree(export_dir)

    export_dir.mkdir(
        parents=True
    )

    numeric_dir = export_dir / "statistics"
    lc_dir = export_dir / "lightcurves_h5"
    config_dir = export_dir / "configuration"

    numeric_dir.mkdir()
    lc_dir.mkdir()
    config_dir.mkdir()

    # --------------------------------------------------
    # Load all completed production chunks
    # --------------------------------------------------

    df = load_done(
        args.run_tag,
        args.workers
    )

    h0_all = df[
        df["truth_case"].eq("H0")
    ].copy()

    h0_ok = h0_all[
        h0_all["status"].eq("ok")
    ].copy()

    h0_timeout = h0_all[
        h0_all["status"].eq("fit_timeout")
    ].copy()

    h0_all.to_parquet(
        numeric_dir / "h0_all_done_rows.parquet",
        index=False
    )

    h0_ok.to_parquet(
        numeric_dir / "h0_ok_full.parquet",
        index=False
    )

    h0_timeout.to_parquet(
        numeric_dir / "h0_fit_timeout.parquet",
        index=False
    )

    # --------------------------------------------------
    # Copy numerical diagnostic outputs only
    # --------------------------------------------------

    if source_analysis.exists():

        for path in source_analysis.iterdir():

            if path.suffix.lower() in {
                ".csv",
                ".parquet",
            }:
                shutil.copy2(
                    path,
                    numeric_dir / path.name
                )

    # --------------------------------------------------
    # Selected representative H0 events
    # --------------------------------------------------

    selected_path = (
        source_analysis
        / "selected_h0_events.csv"
    )

    if selected_path.exists():

        selected = pd.read_csv(
            selected_path
        )

        extraction_manifest = []

        for _, sel in selected.iterrows():

            label = str(sel["label"])
            catalog_row = int(sel["catalog_row"])

            start = int(sel["chunk_start"])
            stop = int(sel["chunk_stop"])

            rows = h0_ok[
                h0_ok["catalog_row"].eq(
                    catalog_row
                )
            ]

            if len(rows) != 1:

                print(
                    "WARNING:",
                    label,
                    catalog_row,
                    "summary rows =",
                    len(rows)
                )

                continue

            row = rows.iloc[0]

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

            if not archive.exists():

                print(
                    "WARNING archive missing:",
                    archive
                )

                continue

            with tarfile.open(
                archive,
                "r"
            ) as tf:

                candidates = [
                    (
                        member_score(
                            m.name,
                            row
                        ),
                        m
                    )
                    for m in tf.getmembers()
                    if m.isfile()
                    and m.name.lower().endswith(".h5")
                ]

                candidates.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                if (
                    not candidates
                    or candidates[0][0] <= 0
                ):

                    print(
                        "WARNING no H5 match:",
                        label,
                        catalog_row
                    )

                    continue

                score, member = candidates[0]

                source = tf.extractfile(
                    member
                )

                if source is None:
                    continue

                target = (
                    lc_dir
                    / (
                        f"{label}_"
                        f"row{catalog_row}_"
                        f"{Path(member.name).name}"
                    )
                )

                with open(
                    target,
                    "wb"
                ) as f:
                    shutil.copyfileobj(
                        source,
                        f
                    )

                extraction_manifest.append({
                    "label": label,
                    "catalog_row": catalog_row,
                    "T": float(sel["T"]),
                    "chunk_start": start,
                    "chunk_stop": stop,
                    "archive_member": member.name,
                    "match_score": score,
                    "local_filename": target.name,
                })

        pd.DataFrame(
            extraction_manifest
        ).to_csv(
            export_dir
            / "lightcurve_manifest.csv",
            index=False
        )

    # --------------------------------------------------
    # Frozen production configuration
    # --------------------------------------------------

    prod_cfg = (
        PRODUCTION_CONFIG_ROOT
        / args.run_tag
    )

    for filename in [
        "config.json",
        "manifest.txt",
    ]:

        source = prod_cfg / filename

        if source.exists():
            shutil.copy2(
                source,
                config_dir / filename
            )

    # --------------------------------------------------
    # Export summary
    # --------------------------------------------------

    summary = pd.DataFrame([{
        "run_tag": args.run_tag,
        "done_chunks": int(
            df["_chunk_start"].nunique()
        ),
        "physical_events": int(
            df["catalog_row"].nunique()
        ),
        "h0_total": len(h0_all),
        "h0_ok": len(h0_ok),
        "h0_fit_timeout": len(h0_timeout),
    }])

    summary.to_csv(
        export_dir / "export_summary.csv",
        index=False
    )

    print()
    print("=" * 70)
    print("LOCAL DATA EXPORT")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    print("Export directory:")
    print(export_dir)

    print()
    print("Files:")
    for path in sorted(
        export_dir.rglob("*")
    ):
        if path.is_file():
            print(
                path.relative_to(
                    export_dir
                )
            )


if __name__ == "__main__":
    main()
