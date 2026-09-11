#!/usr/bin/env python3

from pathlib import Path


PROJECT = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe"
)

CORE = (
    PROJECT
    / "validation/bounds_audit/"
    / "run_bounds_audit_refit_core.py"
)

WRAPPER = (
    PROJECT
    / "validation/bounds_audit/"
    / "run_bounds_audit_refit.py"
)


# ============================================================
# CORE
# ============================================================

s = CORE.read_text()


# ------------------------------------------------------------
# Ensure os is imported.
# ------------------------------------------------------------

if "import os\n" not in s[:2000]:
    s = "import os\n" + s


# ------------------------------------------------------------
# Add reference5000 profile.
# ------------------------------------------------------------

if '"reference5000": {' not in s:

    needle = '''    "stress": {
        "u0": [-5.0, 5.0],
        "tE": [0.1, 2000.0],
'''

    replacement = '''    "reference5000": {
        "u0": [-5.0, 5.0],
        "tE": [0.1, 5000.0],
        "rho": [1.0e-7, 5.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    },

    "stress": {
        "u0": [-5.0, 5.0],
        "tE": [0.1, 2000.0],
'''

    if needle not in s:
        raise RuntimeError(
            "Could not find stress profile."
        )

    s = s.replace(
        needle,
        replacement,
        1,
    )


# ------------------------------------------------------------
# Add core CLI choice.
# ------------------------------------------------------------

cli_start = s.index(
    "parser = argparse.ArgumentParser()"
)

cli = s[cli_start:]

if '"reference5000"' not in cli:

    needle = '''        "stress",
'''

    replacement = '''        "reference5000",
        "stress",
'''

    if needle not in cli:
        raise RuntimeError(
            "Could not locate core bounds-profile choices."
        )

    absolute = (
        cli_start
        + cli.index(needle)
    )

    s = (
        s[:absolute]
        + replacement
        + s[
            absolute
            + len(needle):
        ]
    )


# ------------------------------------------------------------
# Cross-seed loader.
# ------------------------------------------------------------

helper_marker = '''# ============================================================
# ONE FIT
# ============================================================
'''

if "def load_h0_crossseeds(" not in s:

    helper = r'''
# ============================================================
# OPTIONAL H0 CROSS-SEEDS FOR BOUNDS VALIDATION
# ============================================================

def load_h0_crossseeds(meta):
    """
    Load diagnostic H0 starts discovered in previous validation fits.

    These starts are used only when the environment variable

        HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST

    is defined.

    This is a validation tool, not a production initialization
    strategy. It is designed to separate parameter-domain
    convergence from optimizer-basin effects.
    """

    text = os.environ.get(
        "HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST",
        "",
    ).strip()

    if not text:
        return []

    path = Path(
        text
    ).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"H0 cross-seed manifest not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "catalog_row",
        "sample",
        "source_profile",
        "source_label",
        "source_chi2",
        "t0",
        "u0",
        "tE",
        "rho",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Cross-seed manifest missing columns: "
            + repr(sorted(missing))
        )

    x = df[
        (
            df["catalog_row"].astype(int)
            == int(meta["row"])
        )
        & (
            df["sample"].astype(str)
            == str(meta["sample"])
        )
    ].copy()

    if len(x) == 0:
        return []

    # Data-driven t0 range, identical to run_one_fit().
    all_times = []

    for band in [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
    ]:

        lc = meta["curves"][band]

        if len(lc):
            all_times.extend(
                np.asarray(
                    lc[:, 0],
                    dtype=float,
                ).tolist()
            )

    if len(all_times) == 0:
        raise RuntimeError(
            "Cannot validate H0 cross-seeds: "
            "no Rubin times."
        )

    t_min = float(
        np.min(all_times)
    )

    t_max = float(
        np.max(all_times)
    )

    spec = BOUNDS_PROFILES[
        BOUNDS_PROFILE
    ]

    starts = []

    for _, r in x.sort_values(
        "source_chi2"
    ).iterrows():

        initial = {
            "t0": float(r["t0"]),
            "u0": float(r["u0"]),
            "tE": float(r["tE"]),
            "rho": float(r["rho"]),
        }

        if not all(
            np.isfinite(
                list(
                    initial.values()
                )
            )
        ):
            continue

        violations = []

        if not (
            t_min
            <= initial["t0"]
            <= t_max
        ):
            violations.append("t0")

        for p in [
            "u0",
            "tE",
            "rho",
        ]:

            lo, hi = spec[p]

            if not (
                lo
                <= initial[p]
                <= hi
            ):
                violations.append(p)

        if violations:

            print(
                "SKIP H0 cross-seed outside current domain:",
                r["source_profile"],
                r["source_label"],
                "violations=",
                violations,
            )

            continue

        label = (
            "crossseed_"
            + str(
                r["source_profile"]
            )
            + "_"
            + str(
                r["source_label"]
            )
        )

        starts.append(
            (
                label,
                initial,
            )
        )

    return starts


'''

    if helper_marker not in s:
        raise RuntimeError(
            "Could not find ONE FIT marker."
        )

    s = s.replace(
        helper_marker,
        helper
        + helper_marker,
        1,
    )


# ------------------------------------------------------------
# Add exact cross-seeds AFTER the existing H0 rho-grid starts.
#
# They are NOT crossed with RHO_GRID. Each is used exactly once.
# ------------------------------------------------------------

if "n_h0_crossseeds_added = 0" not in s:

    event_start = s.index(
        "def run_event("
    )

    h1_marker = '''    # --------------------------------------------------------
    # H1:
'''

    h1_pos = s.index(
        h1_marker,
        event_start,
    )

    block = r'''
    # --------------------------------------------------------
    # H0 validation cross-seeds.
    #
    # These are exact previously discovered solutions and are
    # deliberately NOT crossed with RHO_GRID.
    # --------------------------------------------------------

    crossseed_starts = load_h0_crossseeds(
        meta
    )

    n_h0_crossseeds_added = 0

    for label, initial in crossseed_starts:

        key = (
            round(
                float(initial["t0"]),
                5,
            ),
            round(
                float(initial["u0"]),
                6,
            ),
            round(
                float(initial["tE"]),
                5,
            ),
            round(
                float(initial["rho"]),
                8,
            ),
        )

        if key in h0_seen:
            continue

        h0_seen.add(
            key
        )

        h0_starts.append(
            (
                label,
                initial,
            )
        )

        n_h0_crossseeds_added += 1

    print(
        "H0 validation cross-seeds added =",
        n_h0_crossseeds_added,
    )

'''

    s = (
        s[:h1_pos]
        + block
        + s[h1_pos:]
    )


# ------------------------------------------------------------
# Record cross-seed provenance in summary.json.
# ------------------------------------------------------------

if '"n_h0_crossseeds_added":' not in s:

    needle = '''        "n_h0_starts":
            len(h0_starts),

'''

    replacement = '''        "n_h0_starts":
            len(h0_starts),

        "n_h0_crossseeds_added":
            n_h0_crossseeds_added,

        "h0_crossseed_manifest":
            os.environ.get(
                "HIDDEN_PARALLAX_H0_CROSSSEED_MANIFEST",
                None,
            ),

'''

    if needle not in s:
        raise RuntimeError(
            "Could not find n_h0_starts summary field."
        )

    s = s.replace(
        needle,
        replacement,
        1,
    )


CORE.write_text(
    s
)


# ============================================================
# WRAPPER
# ============================================================

w = WRAPPER.read_text()

if '"reference5000"' not in w:

    needle = '''            "stress",
'''

    replacement = '''            "reference5000",
            "stress",
'''

    if needle not in w:
        raise RuntimeError(
            "Could not locate wrapper profile choices."
        )

    w = w.replace(
        needle,
        replacement,
        1,
    )

WRAPPER.write_text(
    w
)


print(
    "updated:",
    CORE,
)

print(
    "updated:",
    WRAPPER,
)
