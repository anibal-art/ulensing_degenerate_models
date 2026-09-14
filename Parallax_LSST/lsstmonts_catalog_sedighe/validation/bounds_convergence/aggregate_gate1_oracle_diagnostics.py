#!/usr/bin/env python3

"""
Aggregate existing Gate-1 H0 per-strategy diagnostics into repo-tracked
CSVs.

This script does NOT run any fits. It only reads all_refits.csv files
that were already produced locally by run_bounds_audit_refit.py under
three known root families, and combines them into two tables:

  - gate1_oracle_diagnostics.csv
        one row per (source, mode, catalog_row, start_slot): chi2,
        status, optimizer diagnostics, and best-fit parameters for
        every individual H0 start, plus a canonical `strategy_id`
        column (see STRATEGY-ID DESIGN below).

  - gate1_oracle_call_wall_times.csv
        one row per (source, mode, catalog_row): wall-clock seconds
        for the whole wrapper call that produced that all_refits.csv
        (i.e. all starts for that mode run together), plus a naive
        per-start average. This is NOT true per-start timing; the
        core fitter does not currently record per-start wall time.
        Treat wall time as a secondary, approximate cost diagnostic
        only -- N_fits is the primary cost metric at this stage (see
        ADAPTIVE_H0_ANALYSIS.md, section 3).

Nothing under validation/bounds_audit/ or any production code is
read or modified by this script.

THIS SCRIPT DOES NOT RUN FITS AND DOES NOT USE ORACLE INFORMATION.
It re-aggregates already-computed local results and derives a
strategy identifier from the fitter's own start-construction code
(see below); it does not compare any run's chi2 against another
run's chi2. Outputs are development/audit tooling for the H0
adaptive-strategy analysis (see ADAPTIVE_H0_ANALYSIS.md), not a
production or scientific validation artifact themselves.

STRATEGY-ID DESIGN
    The `label` column is copied verbatim from the fitter's own
    per-fit record and is NOT a safe cross-event strategy key for
    the `gate1_oracle_52` source: it embeds the numeric initial rho
    value for the two `old_H0`-anchor slots that seed rho from the
    anchor's own value (start_slot 1) or from that event's own truth
    rho (start_slot 2), and for the one `truth`-anchor slot that
    seeds rho from that event's own truth rho (start_slot 8) -- see
    `run_bounds_audit_refit_core.py:2436-2492`. The remaining 10/13
    slots per mode draw rho from the fixed `RHO_GRID`
    (`run_bounds_audit_refit_core.py:1489-1495`,
    `[1e-6, 1e-4, 1e-2, 1e-1, 1.0]`) and are already label-stable.

    `strategy_id` makes all 13 slots per mode explicitly stable and
    reconstructs the *semantic* start (`anchor/rho_rule`) rather than
    its per-event numeric label:

        slot  anchor   rho rule            strategy_id
        1     old_H0   anchor's own rho    old_H0/rho_from_old_H0
        2     old_H0   this event's truth  old_H0/truth_rho
                        rho
        3-7   old_H0   RHO_GRID            old_H0/1e-06 ... old_H0/1
        8     truth    this event's truth  truth/truth_rho
                        rho (== anchor's
                        own rho)
        9-13  truth    RHO_GRID            truth/1e-06 ... truth/1

    This ordering is a direct transcription of `h0_anchor_vectors`
    (old_H0 first, then truth) crossed with
    `unique_rhos([v[3], truth_rho, *RHO_GRID])` in
    `run_bounds_audit_refit_core.py:2436-2492`, and is valid only
    because every `gate1_oracle_52` (mode, catalog_row) call in this
    dataset produced exactly 13 rows (verified by
    `audit_gate1_oracle_diagnostics.py`; a `dedup` in `unique_rhos`
    could in principle drop a slot for some event and shift this
    mapping, which is why `_gate1_oracle_52_strategy_id` below raises
    instead of silently mis-labelling if a call does not have exactly
    13 rows). `strategy_id` uses its own "anchor/rho" spelling (e.g.
    "old_H0/1e-06") and is not expected to be byte-identical to the
    fitter's raw label (e.g. "old_H0_rho_1e-06") even for the 10
    already label-stable grid slots.

    For the two `gate1_final18` sources, `label` is already a stable,
    event-independent strategy key by construction (the plan's rho
    values are either literal constants or the literal tag `"truth"`,
    never a per-event number -- see `gate1_plan` in
    `run_bounds_audit_refit_core.py:2585-2615`), so `strategy_id` is
    simply set equal to `label` there.

Usage (run from the repository root):

    python Parallax_LSST/lsstmonts_catalog_sedighe/validation/ \
        bounds_convergence/aggregate_gate1_oracle_diagnostics.py

    # Fail loudly instead of warning on missing files / unexpected
    # row counts:
    python .../aggregate_gate1_oracle_diagnostics.py --strict

    # Override a root directory (repeatable), instead of editing
    # ROOTS in this file:
    python .../aggregate_gate1_oracle_diagnostics.py \
        --root gate1_oracle_52=/path/to/gate1_coordinates

    # Equivalently, via environment variable (CLI --root wins if
    # both are given):
    GATE1_DIAG_ROOT_GATE1_ORACLE_52=/path/to/gate1_coordinates \
        python .../aggregate_gate1_oracle_diagnostics.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODES = [
    "physical",
    "log_te",
    "log_rho",
    "log_te_rho",
]

# ============================================================
# Local root families.
#
# Each entry maps a source tag to:
#   root_default      - default local directory containing <mode>/
#                        subdirs on the machine this script was
#                        developed on. Override per-source with
#                        --root SOURCE=PATH or the environment
#                        variable GATE1_DIAG_ROOT_<SOURCE_UPPER>
#                        (SOURCE_UPPER = source.upper() with every
#                        non [A-Z0-9] character replaced by "_"),
#                        e.g. GATE1_DIAG_ROOT_GATE1_FINAL18_RESCUE_T0MARGIN0_25.
#   expected          - expected row count in all_refits.csv per
#                        mode (used only to flag mismatches, not to
#                        drop data)
#   t0_margin_factor  - value of HIDDEN_PARALLAX_T0_MARGIN_FACTOR
#                        used for that root (0 = base t0 interval,
#                        as read from the data; the fitter default
#                        is also 0, but we record it explicitly here
#                        as a property of the *run*, not re-derived)
#   h0_start_plan     - value of HIDDEN_PARALLAX_H0_START_PLAN used
#                        for that root ("all" = full per-mode start
#                        list that the 52-strategy oracle is drawn
#                        from; "gate1_final18" = the current reduced
#                        production-candidate plan)
#
# IMPORTANT -- what "gate1_final18_rescue_t0margin0.25" actually is:
# it is a diagnostic run of the full gate1_final18 plan (18 starts)
# on ALL 100 extreme100 events with t0_margin_factor=0.25 GLOBALLY
# applied. It does NOT represent the validated production policy,
# which only ever reruns t0_margin_factor=0.25 for the ONE event
# whose t0_margin_factor=0 winner has an active t0 bound (in
# extreme100, only catalog_row 36103). See
# ADAPTIVE_H0_ANALYSIS.md section 1a for the full distinction; do not
# conflate "this run exists for 100 events" with "the pipeline reruns
# 100 events at margin 0.25".
#
# These paths and values are read directly out of
# run_gate1_extreme100_coordinates.sh, run_gate1B_final18_t0base.sh
# and run_gate1B_final18.sh. If any of these local directories have
# moved, override them via --root/env var (see module docstring)
# instead of editing ROOTS in place, so the mapping used for a given
# invocation stays visible in the command line / environment rather
# than only in a local, unversioned edit.
# ============================================================

ROOTS = {
    "gate1_oracle_52": {
        "root_default": Path(
            "~/Downloads/hidden_parallax/"
            "production_validation/gate1_coordinates"
        ).expanduser(),
        "expected": {mode: 13 for mode in MODES},
        "t0_margin_factor": 0.0,
        "h0_start_plan": "all",
    },
    "gate1_final18_base_t0margin0": {
        "root_default": Path(
            "~/Downloads/hidden_parallax/"
            "production_validation/gate1B_final18_t0base"
        ).expanduser(),
        "expected": {
            "physical": 3,
            "log_te": 6,
            "log_rho": 8,
            "log_te_rho": 1,
        },
        "t0_margin_factor": 0.0,
        "h0_start_plan": "gate1_final18",
    },
    "gate1_final18_rescue_t0margin0.25": {
        "root_default": Path(
            "~/Downloads/hidden_parallax/"
            "production_validation/gate1B_final18"
        ).expanduser(),
        "expected": {
            "physical": 3,
            "log_te": 6,
            "log_rho": 8,
            "log_te_rho": 1,
        },
        "t0_margin_factor": 0.25,
        "h0_start_plan": "gate1_final18",
    },
}


def _env_key_for_source(source: str) -> str:

    safe = "".join(
        c if (c.isalnum()) else "_" for c in source.upper()
    )

    return f"GATE1_DIAG_ROOT_{safe}"


def resolve_roots(
    cli_overrides: dict[str, str],
) -> dict[str, Path]:
    """
    Resolve the effective root directory for every source in ROOTS,
    applying, in priority order: --root SOURCE=PATH (highest),
    the GATE1_DIAG_ROOT_<SOURCE_UPPER> environment variable, then
    ROOTS[source]["root_default"].
    """

    unknown = set(cli_overrides) - set(ROOTS)
    if unknown:
        raise RuntimeError(
            "Unknown --root source(s): "
            f"{sorted(unknown)}; known sources are "
            f"{sorted(ROOTS)}"
        )

    resolved = {}

    for source, cfg in ROOTS.items():

        if source in cli_overrides:
            root = Path(cli_overrides[source]).expanduser()
        else:
            env_val = os.environ.get(
                _env_key_for_source(source)
            )
            root = (
                Path(env_val).expanduser()
                if env_val
                else cfg["root_default"]
            )

        resolved[source] = root

    return resolved


def _parse_root_override(spec: str) -> tuple[str, str]:

    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--root must be SOURCE=PATH, got {spec!r}"
        )

    source, path = spec.split("=", 1)
    return source.strip(), path.strip()


# ============================================================
# Canonical strategy_id for the gate1_oracle_52 ("all") plan.
#
# Reconstructed directly from the start-construction loop in
# run_bounds_audit_refit_core.py:2436-2492, crossing
# h0_anchor_vectors = [("old_H0", meta["old_h0"][:4]),
#                      ("truth", [t0,u0,tE,rho]_truth)]
# with
#   unique_rhos([v[3], truth_rho, *RHO_GRID])
# where RHO_GRID = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
# (run_bounds_audit_refit_core.py:1489-1495) and unique_rhos()
# preserves input order while dropping near-duplicates
# (run_bounds_audit_refit_core.py:1609-1634).
#
# For the "old_H0" anchor, v[3] is old_H0's own rho (distinct from
# truth's rho in general), so its rho list is
#   [rho_from_old_H0, truth_rho, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
# -> 7 slots (assuming no accidental duplicate; see the guard in
# _gate1_oracle_52_strategy_id below).
#
# For the "truth" anchor, v[3] IS truth_rho, so the list's first two
# entries collapse under unique_rhos() to one:
#   [truth_rho, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
# -> 6 slots.
#
# 7 + 6 = 13 slots per mode, matching the empirically verified
# per-(mode, catalog_row) row count of 13/13 for every extreme100
# event in this source (see
# gate1_oracle_diagnostics_audit_by_source.csv). Grid-value numbers
# are formatted with the same "%.6g" rule as the fitter's own
# f-string (run_bounds_audit_refit_core.py:2484), but strategy_id
# uses "anchor/rho" (e.g. "old_H0/1e-06") while the fitter's raw
# label uses "anchor_rho_<value>" (e.g. "old_H0_rho_1e-06") -- the
# two strings are NOT expected to be byte-identical even for the
# already-stable grid slots; strategy_id is a separate, explicit
# identifier, not a reformatting of label.
# ============================================================

_RHO_GRID_LABELS = [
    f"{v:.6g}" for v in (1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)
]

GATE1_ORACLE_52_SLOT_STRATEGY_ID = {
    1: "old_H0/rho_from_old_H0",
    2: "old_H0/truth_rho",
    3: f"old_H0/{_RHO_GRID_LABELS[0]}",
    4: f"old_H0/{_RHO_GRID_LABELS[1]}",
    5: f"old_H0/{_RHO_GRID_LABELS[2]}",
    6: f"old_H0/{_RHO_GRID_LABELS[3]}",
    7: f"old_H0/{_RHO_GRID_LABELS[4]}",
    8: "truth/truth_rho",
    9: f"truth/{_RHO_GRID_LABELS[0]}",
    10: f"truth/{_RHO_GRID_LABELS[1]}",
    11: f"truth/{_RHO_GRID_LABELS[2]}",
    12: f"truth/{_RHO_GRID_LABELS[3]}",
    13: f"truth/{_RHO_GRID_LABELS[4]}",
}


def _gate1_oracle_52_strategy_id(
    catalog_row: int,
    mode: str,
    start_slot: int,
    n_slots_in_call: int,
) -> str:

    if n_slots_in_call != 13:
        raise RuntimeError(
            "GATE1_ORACLE_52_SLOT_STRATEGY_ID assumes exactly 13 "
            "starts per (mode, catalog_row) call (7 old_H0 + 6 "
            "truth slots, see module docstring); "
            f"catalog_row={catalog_row}, mode={mode} has "
            f"{n_slots_in_call} rows instead. Refusing to guess a "
            "strategy_id mapping for this call -- inspect this "
            "event's all_refits.csv and, if a genuine rho "
            "collision under unique_rhos() occurred, extend the "
            "mapping logic explicitly rather than assuming the "
            "default 13-slot order."
        )

    strategy_id = GATE1_ORACLE_52_SLOT_STRATEGY_ID.get(start_slot)

    if strategy_id is None:
        raise RuntimeError(
            f"No strategy_id mapping for start_slot={start_slot} "
            "in the 13-slot gate1_oracle_52 plan "
            f"(catalog_row={catalog_row}, mode={mode})."
        )

    return strategy_id

# Sub-path from <root>/<mode>/ down to the per-event all_refits.csv,
# shared by all three root families.
SUBPATH = Path(
    "refits/bounds_convergence/te500000_h0only/extreme100"
)

MANIFEST = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/data/"
    "extreme100_refit_manifest.csv"
)

RESULTS_DIR = Path(
    "Parallax_LSST/lsstmonts_catalog_sedighe/"
    "validation/bounds_convergence/results"
)

OUT_FITS = RESULTS_DIR / "gate1_oracle_diagnostics.csv"
OUT_CALLS = RESULTS_DIR / "gate1_oracle_call_wall_times.csv"

# Columns kept from all_refits.csv, in this order, if present.
# start_slot, strategy_id, source, root, mode, catalog_row,
# h0_start_plan and t0_margin_factor are added by this script;
# everything else is whatever run_one_fit() wrote into the per-fit
# record. strategy_id == label for the two gate1_final18 sources
# (label is already event-independent there); for gate1_oracle_52 it
# is reconstructed from start_slot via
# GATE1_ORACLE_52_SLOT_STRATEGY_ID (see module docstring) because
# raw label embeds a per-event numeric rho for 3/13 slots per mode.
KEEP_COLUMNS = [
    "source",
    "root",
    "h0_start_plan",
    "t0_margin_factor",
    "mode",
    "catalog_row",
    "start_slot",
    "hypothesis",
    "label",
    "strategy_id",
    "status",
    "chi2",
    "t0",
    "u0",
    "tE",
    "rho",
    "optimizer_success",
    "optimizer_optimality",
    "optimizer_active_mask",
    "log",
]


def read_extreme100_rows() -> list[int]:

    manifest = pd.read_csv(MANIFEST)

    rows = (
        pd.to_numeric(
            manifest["catalog_row"],
            errors="raise",
        )
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    return rows


def load_all_refits(
    path: Path,
    *,
    source: str,
    root_name: str,
    mode: str,
    catalog_row: int,
    t0_margin_factor: float,
    h0_start_plan: str,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    # Preserve on-disk row order: this is the execution order of the
    # starts for this (source, mode, catalog_row) call. It is used
    # both as provenance (start_slot) and, for h0_start_plan=="all",
    # to reconstruct the canonical strategy_id (see module
    # docstring); raw label is NOT used as the cross-event key for
    # that plan because it embeds event-specific numeric values for
    # 3/13 slots per mode.
    df = df.reset_index(drop=True)
    df["start_slot"] = np.arange(len(df)) + 1

    df["source"] = source
    df["root"] = root_name
    df["mode"] = mode
    df["catalog_row"] = int(catalog_row)
    df["t0_margin_factor"] = t0_margin_factor
    df["h0_start_plan"] = h0_start_plan

    if h0_start_plan == "all":
        n_slots = len(df)
        df["strategy_id"] = [
            _gate1_oracle_52_strategy_id(
                catalog_row=catalog_row,
                mode=mode,
                start_slot=int(slot),
                n_slots_in_call=n_slots,
            )
            for slot in df["start_slot"]
        ]
    elif h0_start_plan == "gate1_final18":
        # label is already event-independent for this plan
        # (verified in gate1_oracle_diagnostics_audit_by_source.csv:
        # 18/18 (mode, start_slot) -> label pairs constant across
        # all 100 extreme100 events), so it is used directly as the
        # canonical strategy_id.
        df["strategy_id"] = df["label"]
    else:
        raise RuntimeError(
            "No strategy_id rule defined for "
            f"h0_start_plan={h0_start_plan!r} "
            f"(source={source}). Add one explicitly rather than "
            "falling back to raw label."
        )

    return df


def load_call_wall_seconds(
    root: Path,
    mode: str,
    catalog_row: int,
) -> float:

    p = root / mode / f"wall_{catalog_row}.txt"

    if not p.exists():
        return float("nan")

    try:
        return float(p.read_text().strip())
    except Exception:
        return float("nan")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate existing Gate-1 all_refits.csv diagnostics "
            "into repo-tracked CSVs. Runs no fits."
        )
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Drop (source, mode, catalog_row) entries whose row "
            "count does not match the expected count, instead of "
            "keeping them with a warning."
        ),
    )

    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        type=_parse_root_override,
        help=(
            "Override the local root directory for one source "
            "(repeatable). SOURCE must be one of: "
            f"{sorted(ROOTS)}. Takes priority over the "
            "GATE1_DIAG_ROOT_<SOURCE_UPPER> environment variable "
            "and over the built-in default."
        ),
    )

    args = parser.parse_args()

    cli_overrides = dict(args.root)
    roots_by_source = resolve_roots(cli_overrides)

    rows = read_extreme100_rows()

    fit_frames: list[pd.DataFrame] = []
    call_records: list[dict] = []
    missing: list[tuple] = []
    mismatched: list[tuple] = []

    for source, cfg in ROOTS.items():

        root: Path = roots_by_source[source]

        for mode in MODES:

            expected_n = cfg["expected"].get(mode)

            for row in rows:

                event_dir = root / mode / SUBPATH / str(row)
                p = event_dir / "all_refits.csv"

                if not p.exists():
                    missing.append((source, mode, row, str(p)))
                    continue

                df = load_all_refits(
                    p,
                    source=source,
                    root_name=root.name,
                    mode=mode,
                    catalog_row=row,
                    t0_margin_factor=cfg["t0_margin_factor"],
                    h0_start_plan=cfg["h0_start_plan"],
                )

                if (
                    expected_n is not None
                    and len(df) != expected_n
                ):

                    mismatched.append(
                        (source, mode, row, len(df), expected_n)
                    )

                    if args.strict:
                        continue

                fit_frames.append(df)

                wall = load_call_wall_seconds(root, mode, row)
                n_starts = int(len(df))

                call_records.append(
                    {
                        "source": source,
                        "root": root.name,
                        "mode": mode,
                        "catalog_row": int(row),
                        "n_starts_in_call": n_starts,
                        "wall_seconds_call": wall,
                        "wall_seconds_per_start_avg": (
                            wall / n_starts
                            if wall == wall and n_starts > 0
                            else float("nan")
                        ),
                    }
                )

    if not fit_frames:
        raise RuntimeError(
            "No all_refits.csv files were found under any of the "
            "resolved roots. Check that the default paths in ROOTS "
            "match this machine, or override them with "
            "--root SOURCE=PATH or GATE1_DIAG_ROOT_<SOURCE_UPPER>."
        )

    fits = pd.concat(fit_frames, ignore_index=True)

    keep = [c for c in KEEP_COLUMNS if c in fits.columns]
    fits = fits[keep]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fits.to_csv(OUT_FITS, index=False)

    calls = pd.DataFrame(call_records)
    calls.to_csv(OUT_CALLS, index=False)

    print(f"wrote {len(fits)} fit rows to {OUT_FITS}")
    print(f"wrote {len(calls)} call rows to {OUT_CALLS}")
    print(f"missing files: {len(missing)}")
    print(f"row-count mismatches: {len(mismatched)}")

    if missing:
        print()
        print("MISSING (first 20):")
        for m in missing[:20]:
            print("  ", m)

    if mismatched:
        print()
        print("ROW-COUNT MISMATCHES (first 20):")
        print("   (source, mode, catalog_row, found_n, expected_n)")
        for m in mismatched[:20]:
            print("  ", m)


if __name__ == "__main__":
    main()
