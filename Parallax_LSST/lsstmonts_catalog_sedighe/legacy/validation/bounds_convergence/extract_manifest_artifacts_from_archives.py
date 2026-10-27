#!/usr/bin/env python3

import argparse
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd


def safe_target(root, member_name):
    """
    Return a safe extraction target below root.
    Reject absolute paths and path traversal.
    """

    p = PurePosixPath(member_name)

    if p.is_absolute():
        raise RuntimeError(
            f"Absolute tar member path: {member_name}"
        )

    if ".." in p.parts:
        raise RuntimeError(
            f"Unsafe tar member path: {member_name}"
        )

    target = root.joinpath(
        *p.parts
    )

    root_abs = os.path.abspath(root)
    target_abs = os.path.abspath(target)

    if os.path.commonpath(
        [root_abs, target_abs]
    ) != root_abs:
        raise RuntimeError(
            f"Unsafe extraction target: {target}"
        )

    return target


def extract_file(tf, member, destination_root):

    target = safe_target(
        destination_root,
        member.name,
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = tf.extractfile(
        member
    )

    if source is None:
        raise RuntimeError(
            f"Cannot read tar member: {member.name}"
        )

    with source, open(target, "wb") as fout:
        shutil.copyfileobj(
            source,
            fout,
        )


def verify_case(case):

    h5 = list(
        case.rglob(
            "Event_*.h5"
        )
    )

    h0 = [
        p
        for p in case.rglob(
            "*TRF_FSPL_NoParallax.npy"
        )
        if "_H1_multistart"
        not in str(p)
    ]

    h1 = [
        p
        for p in case.rglob(
            "*TRF_FSPL_Parallax.npy"
        )
        if "_H1_multistart"
        not in str(p)
    ]

    truth = list(
        case.rglob(
            "true_rr_manual_*.parquet"
        )
    )

    multifit = list(
        case.rglob(
            "multi_fit_manual_*.parquet"
        )
    )

    multistarts = list(
        case.rglob(
            "_H1_multistart/**/*Parallax.npy"
        )
    )

    return {
        "H5":
            len(h5),

        "H0_final":
            len(h0),

        "H1_final":
            len(h1),

        "truth":
            len(truth),

        "multi_fit":
            len(multifit),

        "old_H1_multistarts":
            len(multistarts),

        "ok":
            (
                len(h5) == 1
                and len(h0) == 1
                and len(h1) == 1
                and len(truth) == 1
                and len(multifit) == 1
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Destination artifacts directory. "
            "Events are written as "
            "<output-root>/<sample>/<catalog_row>/..."
        ),
    )

    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    required = {
        "sample",
        "catalog_row",
        "source_archive",
    }

    missing = (
        required
        - set(manifest.columns)
    )

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            + repr(sorted(missing))
        )

    if manifest["catalog_row"].duplicated().any():
        raise RuntimeError(
            "Manifest contains duplicate catalog_row."
        )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Group by archive so every multi-GB TAR is scanned once.
    # ========================================================

    groups = list(
        manifest.groupby(
            "source_archive",
            sort=True,
        )
    )

    print("=" * 100)
    print("SELECTIVE ARTIFACT EXTRACTION")
    print("=" * 100)
    print("events   =", len(manifest))
    print("archives =", len(groups))
    print("output   =", output_root)

    extracted_counts = {
        int(row):
            0
        for row in manifest[
            "catalog_row"
        ].astype(int)
    }

    for i, (
        archive_text,
        group,
    ) in enumerate(
        groups,
        1,
    ):

        archive = Path(
            str(archive_text)
        )

        if not archive.exists():
            raise FileNotFoundError(
                f"Archive does not exist: {archive}"
            )

        wanted = {}

        for _, r in group.iterrows():

            catalog_row = int(
                r["catalog_row"]
            )

            sample = str(
                r["sample"]
            )

            event_token = (
                f"event_{catalog_row:07d}"
                "_h1_r000000"
            )

            wanted[event_token] = {
                "catalog_row":
                    catalog_row,

                "sample":
                    sample,
            }

        print()
        print(
            f"[{i}/{len(groups)}]",
            archive,
        )

        print(
            "  requested events =",
            len(wanted),
        )

        with tarfile.open(
            archive,
            "r",
        ) as tf:

            for member in tf:

                if not member.isfile():
                    continue

                parts = PurePosixPath(
                    member.name
                ).parts

                matches = [
                    token
                    for token in wanted
                    if token in parts
                ]

                if not matches:
                    continue

                if len(matches) != 1:
                    raise RuntimeError(
                        "Ambiguous event match for "
                        f"{member.name}: {matches}"
                    )

                token = matches[0]

                info = wanted[token]

                catalog_row = info[
                    "catalog_row"
                ]

                sample = info[
                    "sample"
                ]

                case = (
                    output_root
                    / sample
                    / str(catalog_row)
                )

                # If forcing, remove the event only once,
                # immediately before its first extracted file.
                if (
                    args.force
                    and extracted_counts[
                        catalog_row
                    ] == 0
                    and case.exists()
                ):
                    shutil.rmtree(
                        case
                    )

                extract_file(
                    tf,
                    member,
                    case,
                )

                extracted_counts[
                    catalog_row
                ] += 1

        print(
            "  members extracted =",
            sum(
                extracted_counts[
                    int(r)
                ]
                for r in group[
                    "catalog_row"
                ]
            ),
        )

    # ========================================================
    # Verification
    # ========================================================

    rows = []

    print()
    print("=" * 100)
    print("VERIFY EXTRACTED EVENTS")
    print("=" * 100)

    for _, r in manifest.iterrows():

        catalog_row = int(
            r["catalog_row"]
        )

        sample = str(
            r["sample"]
        )

        case = (
            output_root
            / sample
            / str(catalog_row)
        )

        check = verify_case(
            case
        )

        record = {
            "sample":
                sample,

            "catalog_row":
                catalog_row,

            "source_archive":
                str(
                    r["source_archive"]
                ),

            "members_extracted":
                extracted_counts[
                    catalog_row
                ],

            **check,
        }

        rows.append(
            record
        )

        print(
            f"{catalog_row:7d}",
            f"H5={check['H5']}",
            f"H0={check['H0_final']}",
            f"H1={check['H1_final']}",
            f"truth={check['truth']}",
            f"multifit={check['multi_fit']}",
            f"H1_ms={check['old_H1_multistarts']}",
            f"members={extracted_counts[catalog_row]}",
            f"OK={check['ok']}",
        )

    inventory = pd.DataFrame(
        rows
    )

    args.inventory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        args.inventory,
        index=False,
    )

    bad = inventory[
        ~inventory["ok"]
    ]

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        "N events =",
        len(inventory),
    )

    print(
        "N valid =",
        int(
            inventory["ok"].sum()
        ),
    )

    print(
        "N problematic =",
        len(bad),
    )

    print(
        "inventory =",
        args.inventory,
    )

    if len(bad):

        print()
        print(
            bad.to_string(
                index=False
            )
        )

        raise SystemExit(
            "Artifact verification failed."
        )


if __name__ == "__main__":
    main()
