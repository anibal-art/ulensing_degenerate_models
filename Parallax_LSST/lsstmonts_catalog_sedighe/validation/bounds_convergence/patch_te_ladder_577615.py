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


PROFILES = {
    "te5000": 5000.0,
    "te50000": 50000.0,
    "te100000": 100000.0,
    "te500000": 500000.0,
    "te1000000": 1000000.0,
}


def ensure_profile_definitions(text):

    marker = '''    "stress": {
'''

    if marker not in text:
        raise RuntimeError(
            "Could not locate stress profile."
        )

    blocks = []

    for name, te_max in PROFILES.items():

        token = f'    "{name}": {{'

        if token in text:
            continue

        blocks.append(
f'''    "{name}": {{
        "u0": [-10.0, 10.0],
        "tE": [0.1, {te_max}],
        "rho": [1.0e-7, 10.0],
        "piEN": [-20.0, 20.0],
        "piEE": [-20.0, 20.0],
    }},

'''
        )

    if blocks:

        text = text.replace(
            marker,
            "".join(blocks) + marker,
            1,
        )

    return text


def ensure_bounds_profile_choices(text):

    # Work only inside the CLI argument for --bounds-profile.
    pos = text.find(
        '"--bounds-profile"'
    )

    if pos < 0:
        raise RuntimeError(
            "--bounds-profile argument not found."
        )

    choices_pos = text.find(
        "choices=[",
        pos,
    )

    if choices_pos < 0:
        raise RuntimeError(
            "choices=[ not found after --bounds-profile."
        )

    open_bracket = text.find(
        "[",
        choices_pos,
    )

    close_bracket = text.find(
        "]",
        open_bracket,
    )

    if (
        open_bracket < 0
        or close_bracket < 0
    ):
        raise RuntimeError(
            "Could not isolate bounds-profile choices."
        )

    block = text[
        open_bracket + 1:
        close_bracket
    ]

    control_line = None

    for line in block.splitlines():

        if '"control20000",' in line:
            control_line = line
            break

    if control_line is None:
        raise RuntimeError(
            "control20000 choice not found."
        )

    indent = control_line[
        :len(control_line)
        - len(control_line.lstrip())
    ]

    missing = [
        name
        for name in PROFILES
        if f'"{name}"' not in block
    ]

    if not missing:
        return text

    insertion = control_line

    for name in missing:
        insertion += (
            "\n"
            + indent
            + f'"{name}",'
        )

    new_block = block.replace(
        control_line,
        insertion,
        1,
    )

    return (
        text[:open_bracket + 1]
        + new_block
        + text[close_bracket:]
    )


# ============================================================
# Core
# ============================================================

core = CORE.read_text()

core = ensure_profile_definitions(
    core
)

core = ensure_bounds_profile_choices(
    core
)

CORE.write_text(core)


# ============================================================
# Wrapper
# ============================================================

wrapper = WRAPPER.read_text()

wrapper = ensure_bounds_profile_choices(
    wrapper
)

WRAPPER.write_text(wrapper)


print("updated:", CORE)
print("updated:", WRAPPER)
