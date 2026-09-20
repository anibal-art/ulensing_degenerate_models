"""
Strict JSON serialization utilities for final LRT production.

Scientific arrays must remain numerical JSON arrays. Unknown Python
objects are rejected instead of being silently converted with str().

Non-finite floating-point values are represented as JSON null.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


def to_jsonable(value):
    """
    Recursively convert production results to standard JSON types.

    Rules
    -----
    numpy arrays        -> nested lists
    numpy integer       -> int
    numpy floating      -> float
    numpy bool          -> bool
    Path                -> str
    tuple               -> list
    non-finite float    -> None

    Unsupported objects raise TypeError.
    """

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, np.ndarray):
        return to_jsonable(
            value.tolist()
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item)
            for item in value
        ]

    raise TypeError(
        "Unsupported object in production JSON: "
        f"type={type(value).__name__} "
        f"value={value!r}"
    )


def atomic_write_json(
    payload,
    path,
):
    """
    Serialize strictly and replace the destination atomically.
    """

    path = Path(path).resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = Path(
        str(path) + ".tmp"
    )

    jsonable = to_jsonable(
        payload
    )

    with tmp_path.open(
        "w"
    ) as handle:
        json.dump(
            jsonable,
            handle,
            indent=2,
            allow_nan=False,
        )

        handle.write("\n")
        handle.flush()
        os.fsync(
            handle.fileno()
        )

    os.replace(
        tmp_path,
        path,
    )
