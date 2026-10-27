#!/usr/bin/env python3

from pathlib import Path


PROJECT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe"
)

CORE = (
    PROJECT
    / "validation/bounds_audit"
    / "run_bounds_audit_refit_core.py"
)

WRAPPER = (
    PROJECT
    / "validation/bounds_audit"
    / "run_bounds_audit_refit.py"
)


# ============================================================
# CORE
# ============================================================

s = CORE.read_text()


# ------------------------------------------------------------
# Add --manifest to the core CLI.
# ------------------------------------------------------------

manifest_arg = '''parser.add_argument(
    "--manifest",
    type=Path,
    default=None,
    help=(
        "Explicit event manifest. If omitted, use the historical "
        "34-event bounds-audit manifest."
    ),
)

'''

if '''parser.add_argument(
    "--manifest",''' not in s:

    needle = '''parser.add_argument(
    "--catalog-row",
    type=int,
    required=True,
)

'''

    if needle not in s:
        raise RuntimeError(
            "Could not locate core --catalog-row block."
        )

    s = s.replace(
        needle,
        needle + manifest_arg,
        1,
    )


# ------------------------------------------------------------
# Replace the hard-coded manifest read.
# ------------------------------------------------------------

old = '''manifest = pd.read_csv(
    MANIFEST
)
'''

new = '''MANIFEST_PATH = (
    Path(args.manifest).expanduser().resolve()
    if args.manifest is not None
    else Path(MANIFEST).expanduser().resolve()
)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

print(
    "manifest         =",
    MANIFEST_PATH,
)

manifest = pd.read_csv(
    MANIFEST_PATH
)
'''

if old in s:
    s = s.replace(
        old,
        new,
        1,
    )
elif "MANIFEST_PATH = (" not in s:
    raise RuntimeError(
        "Could not locate historical manifest read in core."
    )


# ------------------------------------------------------------
# Record manifest provenance in summary.json.
# ------------------------------------------------------------

if '"input_manifest":' not in s:

    needle = '''    summary = {
        "sample":
            meta["sample"],

'''

    replacement = '''    summary = {
        "sample":
            meta["sample"],

        "input_manifest":
            str(MANIFEST_PATH),

'''

    if needle not in s:
        raise RuntimeError(
            "Could not locate summary dictionary in core."
        )

    s = s.replace(
        needle,
        replacement,
        1,
    )


CORE.write_text(s)


# ============================================================
# WRAPPER
# ============================================================

w = WRAPPER.read_text()


# ------------------------------------------------------------
# Add --manifest to wrapper CLI.
# ------------------------------------------------------------

manifest_arg_wrapper = '''    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Explicit event manifest. If omitted, use the historical "
            "34-event bounds-audit manifest."
        ),
    )

'''

if '''    p.add_argument(
        "--manifest",''' not in w:

    needle = '''    p.add_argument(
        "--catalog-row",
        type=int,
        required=True,
    )

'''

    if needle not in w:
        raise RuntimeError(
            "Could not locate wrapper --catalog-row block."
        )

    w = w.replace(
        needle,
        needle + manifest_arg_wrapper,
        1,
    )


# ------------------------------------------------------------
# Forward explicit manifest to runtime core.
# ------------------------------------------------------------

if 'cmd.extend(["--manifest"' not in w:

    marker = '''    if args.dry_run:
'''

    insert = '''    if args.manifest is not None:
        manifest_path = (
            args.manifest
            .expanduser()
            .resolve()
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        cmd.extend(
            [
                "--manifest",
                str(manifest_path),
            ]
        )

'''

    pos = w.rfind(marker)

    if pos < 0:
        raise RuntimeError(
            "Could not locate wrapper command-option block."
        )

    w = (
        w[:pos]
        + insert
        + w[pos:]
    )


WRAPPER.write_text(w)


print("updated:", CORE)
print("updated:", WRAPPER)
