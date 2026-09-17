#!/usr/bin/env python3

from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import os
import platform
import site
import sys

import numpy
import scipy
import pandas
import h5py
import astropy
import pyLIMA
import erfa


PROJECT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT
    / "configs"
    / "repro"
    / "LRT_REPRO_V1_manifest.json"
)

CONFIG = (
    PROJECT
    / "configs"
    / "repro"
    / "LRT_REPRO_V1.json"
)


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def tree_sha256(root):
    root = Path(root)
    h = hashlib.sha256()
    n_files = 0

    for path in sorted(
        p for p in root.rglob("*")
        if p.is_file()
    ):
        rel = path.relative_to(root).as_posix()

        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(
            sha256_file(path).encode("ascii")
        )
        h.update(b"\0")

        n_files += 1

    return {
        "sha256": h.hexdigest(),
        "n_files": n_files,
    }


def require_env(name):
    value = os.environ.get(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Required environment variable is not set: {name}"
        )

    return Path(value).expanduser().resolve()


def status(ok):
    return "OK" if ok else "FAIL"


def main():
    expected = json.loads(
        MANIFEST.read_text()
    )

    rr = require_env("ROMAN_RUBIN_DIR")
    rubin = require_env("RUBIN_SIM_DATA_DIR")

    files = {
        "config":
            CONFIG,

        "catalog_columns":
            PROJECT.parent
            / "data_sedighe"
            / "columns",

        "catalog_data":
            PROJECT.parent
            / "data_sedighe"
            / "LSSTMONTS.dat",

        "opsim_db":
            rubin
            / "sim_baseline"
            / "sim_baseline_2026_07_23"
            / "sim_baseline"
            / "baseline_v5.3.5_10yrs.db",

        "roman_ephemerides":
            rr
            / "ephemerides"
            / "Roman_positions.npy",

        "driver":
            PROJECT
            / "run_lsstmonts_catalog_hidden_parallax.py",

        "standalone_materialize":
            PROJECT
            / "validation"
            / "production_profiling"
            / "standalone_materialize.py",

        "functions_roman_rubin":
            rr
            / "functions_roman_rubin.py",

        "set_telescopes_pyLIMA":
            rr
            / "set_telescopes_pyLIMA.py",

        "fit_lc":
            rr
            / "fit_lc.py",
    }

    versions = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "h5py": h5py.__version__,
        "astropy": astropy.__version__,
        "pyerfa": erfa.__version__,
        "pyLIMA": pyLIMA.__version__,
        "rubin-sim": md.version("rubin-sim"),
        "rubin-scheduler":
            md.version("rubin-scheduler"),
    }

    failures = 0

    print("=" * 72)
    print("LRT_REPRO_V1 ENVIRONMENT CHECK")
    print("=" * 72)

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------
    actual_python = platform.python_version()
    expected_python = expected["python"]["version"]

    ok = actual_python == expected_python
    failures += int(not ok)

    print(
        f"{'python':28s} "
        f"{status(ok):4s} "
        f"actual={actual_python} "
        f"expected={expected_python}"
    )

    # --------------------------------------------------------
    # User site
    # --------------------------------------------------------
    ok = not site.ENABLE_USER_SITE
    failures += int(not ok)

    print(
        f"{'PYTHONNOUSERSITE':28s} "
        f"{status(ok):4s} "
        f"ENABLE_USER_SITE={site.ENABLE_USER_SITE}"
    )

    # --------------------------------------------------------
    # Scientific packages
    # --------------------------------------------------------
    print()
    print("PACKAGES")

    for name, actual in versions.items():
        wanted = expected["packages"][name]

        ok = actual == wanted
        failures += int(not ok)

        print(
            f"{name:28s} "
            f"{status(ok):4s} "
            f"actual={actual} "
            f"expected={wanted}"
        )

    # --------------------------------------------------------
    # Scientific input/code files
    # --------------------------------------------------------
    print()
    print("FILES")

    for name, path in files.items():
        wanted = expected["files"][name]

        if not path.is_file():
            failures += 1

            print(
                f"{name:28s} FAIL "
                f"missing={path}"
            )

            continue

        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size

        hash_ok = (
            actual_hash
            == wanted["sha256"]
        )

        size_ok = (
            actual_size
            == wanted["size"]
        )

        ok = hash_ok and size_ok
        failures += int(not ok)

        print(
            f"{name:28s} "
            f"{status(ok):4s} "
            f"size={actual_size}"
        )

        if not ok:
            print(
                f"    actual sha256   = {actual_hash}"
            )
            print(
                f"    expected sha256 = "
                f"{wanted['sha256']}"
            )
            print(
                f"    path            = {path}"
            )

    # --------------------------------------------------------
    # Rubin throughputs
    # --------------------------------------------------------
    print()
    print("DIRECTORIES")

    throughputs = (
        rubin
        / "throughputs"
        / "baseline"
    )

    if not throughputs.is_dir():
        failures += 1

        print(
            "rubin_throughputs_baseline "
            f"FAIL missing={throughputs}"
        )
    else:
        actual = tree_sha256(
            throughputs
        )

        wanted = expected[
            "directories"
        ][
            "rubin_throughputs_baseline"
        ]

        ok = (
            actual["sha256"]
            == wanted["sha256"]
            and actual["n_files"]
            == wanted["n_files"]
        )

        failures += int(not ok)

        print(
            f"{'rubin_throughputs_baseline':28s} "
            f"{status(ok):4s} "
            f"n_files={actual['n_files']}"
        )

        if not ok:
            print(
                "    actual sha256   =",
                actual["sha256"],
            )
            print(
                "    expected sha256 =",
                wanted["sha256"],
            )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------
    print()
    print("=" * 72)

    if failures == 0:
        print(
            "REPRODUCIBILITY CONTRACT: PASS"
        )
        return 0

    print(
        "REPRODUCIBILITY CONTRACT: FAIL"
    )
    print(
        f"failed checks = {failures}"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
