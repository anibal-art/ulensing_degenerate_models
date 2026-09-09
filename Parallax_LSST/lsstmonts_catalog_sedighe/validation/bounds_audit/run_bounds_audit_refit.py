#!/usr/bin/env python3

"""
Portable launcher for the validated hidden-parallax bounds-audit runner.

The scientific implementation is deliberately preserved in

    run_bounds_audit_refit_core.py

which is an exact snapshot of the runner used for the 17+17 validation
sample on 2026-09-09.

This wrapper changes filesystem locations only. It does not change the
fit model, bounds, starts, optimizer, bounded flux profile, or Jacobian.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

CORE = HERE / "run_bounds_audit_refit_core.py"


# Paths present in the validated local snapshot.
LEGACY_WORK_ROOTS = [
    "~/Downloads/hidden_parallax/hidden_parallax_refit_test",
    "/home/anibal-pc/Downloads/hidden_parallax/hidden_parallax_refit_test",
]

LEGACY_ROMAN_RUBIN = (
    "/home/anibal-pc/microlensing/"
    "simulation_Rubin/roman_rubin"
)

LEGACY_PROJECT = (
    "/home/anibal-pc/ulensing_degenerate_models/"
    "Parallax_LSST/lsstmonts_catalog_sedighe"
)


def build_parser():

    p = argparse.ArgumentParser(
        description="Run one event of the hidden-parallax bounds audit."
    )

    p.add_argument(
        "--catalog-row",
        type=int,
        required=True,
    )

    p.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help=(
            "Directory containing artifacts/ and receiving refits/. "
            "Can also be set with HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT."
        ),
    )

    p.add_argument(
        "--roman-rubin-dir",
        type=Path,
        default=None,
        help=(
            "Path to microlensing/simulation_Rubin/roman_rubin. "
            "Can also be set with ROMAN_RUBIN_DIR."
        ),
    )

    p.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT,
        help="Path to lsstmonts_catalog_sedighe.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    p.add_argument(
        "--keep-runtime",
        action="store_true",
    )

    return p


def resolve_work_root(args):

    if args.work_root is not None:
        return args.work_root.expanduser().resolve()

    env = os.environ.get(
        "HIDDEN_PARALLAX_BOUNDS_AUDIT_ROOT"
    )

    if env:
        return Path(env).expanduser().resolve()

    return (
        HERE
        / "work"
    ).resolve()


def resolve_roman_rubin(args):

    if args.roman_rubin_dir is not None:
        return args.roman_rubin_dir.expanduser().resolve()

    env = os.environ.get(
        "ROMAN_RUBIN_DIR"
    )

    if env:
        return Path(env).expanduser().resolve()

    local_default = Path(
        "/home/anibal-pc/microlensing/"
        "simulation_Rubin/roman_rubin"
    )

    if local_default.exists():
        return local_default.resolve()

    raise RuntimeError(
        "ROMAN_RUBIN_DIR is not defined. "
        "Set it to microlensing/simulation_Rubin/roman_rubin."
    )


def make_runtime_copy(
    *,
    work_root,
    roman_rubin,
    project_dir,
    catalog_row,
):

    if not CORE.exists():
        raise FileNotFoundError(
            CORE
        )

    text = CORE.read_text()

    replacements = {}

    for old in LEGACY_WORK_ROOTS:
        replacements[
            str(
                Path(old).expanduser()
                if old.startswith("~")
                else old
            )
        ] = str(work_root)

        replacements[old] = str(work_root)

    replacements[
        LEGACY_ROMAN_RUBIN
    ] = str(roman_rubin)

    replacements[
        LEGACY_PROJECT
    ] = str(project_dir)

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    runtime_dir = (
        work_root
        / ".runtime"
    )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime = (
        runtime_dir
        / (
            "run_bounds_audit_refit_"
            f"{int(catalog_row)}.py"
        )
    )

    runtime.write_text(
        text
    )

    return runtime


def main():

    args = build_parser().parse_args()

    work_root = resolve_work_root(
        args
    )

    roman_rubin = resolve_roman_rubin(
        args
    )

    project_dir = (
        args.project_dir
        .expanduser()
        .resolve()
    )

    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime = make_runtime_copy(
        work_root=work_root,
        roman_rubin=roman_rubin,
        project_dir=project_dir,
        catalog_row=args.catalog_row,
    )

    cmd = [
        sys.executable,
        str(runtime),
        "--catalog-row",
        str(args.catalog_row),
    ]

    if args.dry_run:
        cmd.append(
            "--dry-run"
        )

    if args.force:
        cmd.append(
            "--force"
        )

    env = dict(
        os.environ
    )

    env[
        "HIDDEN_PARALLAX_BOUNDED_PROFILE"
    ] = "1"

    print(
        "work_root       =",
        work_root,
    )

    print(
        "roman_rubin_dir =",
        roman_rubin,
    )

    print(
        "project_dir     =",
        project_dir,
    )

    print(
        "runtime_runner  =",
        runtime,
    )

    result = subprocess.run(
        cmd,
        env=env,
    )

    if (
        not args.keep_runtime
        and runtime.exists()
    ):
        runtime.unlink()

    raise SystemExit(
        result.returncode
    )


if __name__ == "__main__":
    main()
