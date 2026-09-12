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


def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(
            f"Could not find patch target: {label}"
        )
    return text.replace(old, new, 1)


# ============================================================
# CORE
# ============================================================

s = CORE.read_text()


# ------------------------------------------------------------
# 1. Add a deliberately broader nuisance-domain control.
#
# piE is deliberately kept identical to reference5000 here.
# The present experiment isolates ONLY the shared nuisance
# domain. This is not a proposal for final H1 production bounds.
# ------------------------------------------------------------

if '"control20000": {' not in s:

    needle = '''    "stress": {
        "u0": [-5.0, 5.0],
'''

    replacement = '''    "control20000": {
        "u0": [-10.0, 10.0],
        "tE": [0.1, 20000.0],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "stress": {
        "u0": [-5.0, 5.0],
'''

    s = replace_once(
        s,
        needle,
        replacement,
        "control20000 bounds profile",
    )


# ------------------------------------------------------------
# 2. Add FIT_SCOPE global while preserving old behavior.
# ------------------------------------------------------------

if 'FIT_SCOPE = "both"' not in s:

    needle = '''# Default preserves historical behavior.
BOUNDS_PROFILE = "audit_legacy"
'''

    replacement = '''# Default preserves historical behavior.
BOUNDS_PROFILE = "audit_legacy"
FIT_SCOPE = "both"
'''

    s = replace_once(
        s,
        needle,
        replacement,
        "FIT_SCOPE default",
    )


# ------------------------------------------------------------
# 3. Scope-specific output directories.
#
# Existing "both" paths remain unchanged.
# H0-only:
#   bounds_convergence/reference5000_h0only/...
#   bounds_convergence/control20000_h0only/...
# ------------------------------------------------------------

start = s.index(
    "def bounds_output_root():"
)

end = s.index(
    "# Common rho starts.",
    start,
)

new_function = '''def bounds_output_root():

    # Preserve the exact historical layout for the original
    # full H0+H1 audit.
    if (
        BOUNDS_PROFILE == "audit_legacy"
        and FIT_SCOPE == "both"
    ):
        return OUT

    profile_dir = BOUNDS_PROFILE

    if FIT_SCOPE != "both":
        profile_dir = (
            f"{BOUNDS_PROFILE}_{FIT_SCOPE}only"
        )

    return (
        OUT
        / "bounds_convergence"
        / profile_dir
    )


'''

s = (
    s[:start]
    + new_function
    + s[end:]
)


# ------------------------------------------------------------
# 4. Add control20000 to core CLI choices.
# ------------------------------------------------------------

cli_start = s.index(
    "parser = argparse.ArgumentParser()"
)

cli = s[cli_start:]

if '"control20000"' not in cli:

    needle = '''        "reference5000",
        "stress",
'''

    replacement = '''        "reference5000",
        "control20000",
        "stress",
'''

    cli = replace_once(
        cli,
        needle,
        replacement,
        "core control20000 CLI choice",
    )

    s = (
        s[:cli_start]
        + cli
    )


# ------------------------------------------------------------
# 5. Add --fit-scope to core CLI.
# ------------------------------------------------------------

if '"--fit-scope"' not in s:

    needle = '''parser.add_argument(
    "--dry-run",
    action="store_true",
)
'''

    replacement = '''parser.add_argument(
    "--fit-scope",
    choices=[
        "both",
        "h0",
        "h1",
    ],
    default="both",
    help=(
        "Which hypothesis to refit. Default 'both' preserves "
        "historical behavior. Use 'h0' for shared-bounds "
        "validation without rerunning expensive H1 fits."
    ),
)

parser.add_argument(
    "--dry-run",
    action="store_true",
)
'''

    s = replace_once(
        s,
        needle,
        replacement,
        "core --fit-scope argument",
    )


# ------------------------------------------------------------
# 6. Set global FIT_SCOPE after parsing.
# ------------------------------------------------------------

if "FIT_SCOPE = args.fit_scope" not in s:

    needle = '''BOUNDS_PROFILE = args.bounds_profile
'''

    replacement = '''BOUNDS_PROFILE = args.bounds_profile
FIT_SCOPE = args.fit_scope
'''

    s = replace_once(
        s,
        needle,
        replacement,
        "FIT_SCOPE assignment",
    )


# ------------------------------------------------------------
# 7. Execute only the requested hypothesis.
# ------------------------------------------------------------

run_event_start = s.index(
    "def run_event("
)

loop_start = s.index(
    '''    print()
    print("RUN H0")
''',
    run_event_start,
)

loop_end = s.index(
    '''    df = pd.DataFrame(
        results
    )
''',
    loop_start,
)

new_loops = '''    if FIT_SCOPE in ("both", "h0"):

        print()
        print("RUN H0")

        for i, (label, initial) in enumerate(
            h0_starts,
            1,
        ):

            r = run_one_fit(
                meta,
                "H0",
                initial,
                label,
            )

            results.append(r)

            print(
                f"H0 {i:02d}/{len(h0_starts):02d}",
                r["status"],
                f"chi2={r['chi2']}",
                label,
                flush=True,
            )

    if FIT_SCOPE in ("both", "h1"):

        print()
        print("RUN H1")

        for i, (label, initial) in enumerate(
            h1_starts,
            1,
        ):

            r = run_one_fit(
                meta,
                "H1",
                initial,
                label,
            )

            results.append(r)

            print(
                f"H1 {i:02d}/{len(h1_starts):02d}",
                r["status"],
                f"chi2={r['chi2']}",
                label,
                flush=True,
            )

'''

s = (
    s[:loop_start]
    + new_loops
    + s[loop_end:]
)


# ------------------------------------------------------------
# 8. Record scope explicitly in summary.
# ------------------------------------------------------------

if '"fit_scope":' not in s:

    needle = '''        "input_manifest":
            str(MANIFEST_PATH),

        "bounds_profile":
'''

    replacement = '''        "input_manifest":
            str(MANIFEST_PATH),

        "fit_scope":
            FIT_SCOPE,

        "bounds_profile":
'''

    s = replace_once(
        s,
        needle,
        replacement,
        "fit_scope summary field",
    )


# ------------------------------------------------------------
# 9. Record actual numbers of executed fits.
# ------------------------------------------------------------

if '"n_h0_fits_executed":' not in s:

    needle = '''        "n_h0_starts":
            len(h0_starts),

'''

    replacement = '''        "n_h0_starts":
            len(h0_starts),

        "n_h0_fits_executed":
            int(
                (
                    df["hypothesis"] == "H0"
                ).sum()
            ),

        "n_h1_fits_executed":
            int(
                (
                    df["hypothesis"] == "H1"
                ).sum()
            ),

'''

    s = replace_once(
        s,
        needle,
        replacement,
        "executed-fit summary fields",
    )


CORE.write_text(s)


# ============================================================
# WRAPPER
# ============================================================

w = WRAPPER.read_text()


# ------------------------------------------------------------
# Add control profile choice.
# ------------------------------------------------------------

if '"control20000"' not in w:

    needle = '''            "reference5000",
            "stress",
'''

    replacement = '''            "reference5000",
            "control20000",
            "stress",
'''

    w = replace_once(
        w,
        needle,
        replacement,
        "wrapper control20000 CLI choice",
    )


# ------------------------------------------------------------
# Add wrapper --fit-scope.
# ------------------------------------------------------------

if '"--fit-scope"' not in w:

    needle = '''    p.add_argument(
        "--work-root",
'''

    replacement = '''    p.add_argument(
        "--fit-scope",
        choices=[
            "both",
            "h0",
            "h1",
        ],
        default="both",
        help=(
            "Which hypothesis to refit. Default: both."
        ),
    )

    p.add_argument(
        "--work-root",
'''

    w = replace_once(
        w,
        needle,
        replacement,
        "wrapper --fit-scope argument",
    )


# ------------------------------------------------------------
# Forward scope to runtime core.
# ------------------------------------------------------------

if '''        "--fit-scope",
        str(args.fit_scope),
''' not in w:

    needle = '''        "--bounds-profile",
        str(args.bounds_profile),
    ]
'''

    replacement = '''        "--bounds-profile",
        str(args.bounds_profile),
        "--fit-scope",
        str(args.fit_scope),
    ]
'''

    w = replace_once(
        w,
        needle,
        replacement,
        "wrapper fit-scope forwarding",
    )


WRAPPER.write_text(w)

print("updated:", CORE)
print("updated:", WRAPPER)
