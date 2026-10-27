#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import sys

import h5py
import numpy as np


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def exact_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return False

    if a.dtype != b.dtype:
        return False

    if (
        np.issubdtype(a.dtype, np.number)
        or np.issubdtype(a.dtype, np.bool_)
    ):
        try:
            return np.array_equal(
                a,
                b,
                equal_nan=True,
            )
        except TypeError:
            return np.array_equal(a, b)

    return np.array_equal(a, b)


def compare_attrs(label, a, b, failures):
    ka = set(a.attrs.keys())
    kb = set(b.attrs.keys())

    if ka != kb:
        failures.append(
            f"{label}: attribute names differ: "
            f"A-only={sorted(ka-kb)}, "
            f"B-only={sorted(kb-ka)}"
        )

        return

    for key in sorted(ka):
        va = a.attrs[key]
        vb = b.attrs[key]

        if not exact_equal(va, vb):
            failures.append(
                f"{label}: attribute {key!r} differs"
            )


def collect_objects(h5):
    out = {
        "/": h5,
    }

    def visitor(name, obj):
        out["/" + name] = obj

    h5.visititems(visitor)

    return out


def numeric_diagnostic(a, b):
    try:
        x = np.asarray(a, dtype=float)
        y = np.asarray(b, dtype=float)
    except Exception:
        return ""

    if x.shape != y.shape:
        return ""

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if not np.any(finite):
        return ""

    d = np.abs(
        x[finite] - y[finite]
    )

    return (
        f" max_abs_diff="
        f"{float(np.max(d)):.17g}"
    )


def compare_files(path_a, path_b):
    failures = []

    with h5py.File(path_a, "r") as fa, \
         h5py.File(path_b, "r") as fb:

        oa = collect_objects(fa)
        ob = collect_objects(fb)

        names_a = set(oa)
        names_b = set(ob)

        if names_a != names_b:
            failures.append(
                "Object names differ: "
                f"A-only={sorted(names_a-names_b)}, "
                f"B-only={sorted(names_b-names_a)}"
            )

        for name in sorted(
            names_a & names_b
        ):
            a = oa[name]
            b = ob[name]

            if type(a) is not type(b):
                failures.append(
                    f"{name}: object types differ: "
                    f"{type(a).__name__} vs "
                    f"{type(b).__name__}"
                )

                continue

            compare_attrs(
                name,
                a,
                b,
                failures,
            )

            if isinstance(
                a,
                h5py.Dataset,
            ):
                if a.shape != b.shape:
                    failures.append(
                        f"{name}: shape differs: "
                        f"{a.shape} vs {b.shape}"
                    )

                    continue

                if a.dtype != b.dtype:
                    failures.append(
                        f"{name}: dtype differs: "
                        f"{a.dtype} vs {b.dtype}"
                    )

                    continue

                va = a[()]
                vb = b[()]

                if not exact_equal(
                    va,
                    vb,
                ):
                    diagnostic = (
                        numeric_diagnostic(
                            va,
                            vb,
                        )
                    )

                    failures.append(
                        f"{name}: dataset values differ"
                        f"{diagnostic}"
                    )

    return failures


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "event_a",
        type=Path,
    )

    parser.add_argument(
        "event_b",
        type=Path,
    )

    args = parser.parse_args()

    a = args.event_a.expanduser().resolve()
    b = args.event_b.expanduser().resolve()

    if not a.is_file():
        raise FileNotFoundError(a)

    if not b.is_file():
        raise FileNotFoundError(b)

    print("A =", a)
    print("B =", b)

    hash_a = sha256_file(a)
    hash_b = sha256_file(b)

    print()
    print("SHA256 A =", hash_a)
    print("SHA256 B =", hash_b)
    print(
        "BYTE IDENTICAL =",
        hash_a == hash_b,
    )

    failures = compare_files(
        a,
        b,
    )

    print()

    if not failures:
        print(
            "SCIENTIFIC H5 CONTENT: EXACT"
        )
        print(
            "EVENT COMPARISON: PASS"
        )
        return 0

    print(
        "SCIENTIFIC H5 CONTENT: DIFFERENT"
    )

    for failure in failures:
        print("FAIL:", failure)

    print(
        f"EVENT COMPARISON: FAIL "
        f"({len(failures)} differences)"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
