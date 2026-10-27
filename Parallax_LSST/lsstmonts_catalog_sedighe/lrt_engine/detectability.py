"""Pre-fit (Asimov) detectability gate of the hidden-parallax LRT study.

Functions are moved verbatim from run_lsstmonts_catalog_hidden_parallax.py
(commit deeeaae) with tools/extract_module.py; only this header is new.
The configuration is the same production JSON loaded by sedighe_catalog.
"""

import numpy as np

from sedighe_catalog import CONFIG, cfg


def _prefit_detectability_config():
    """
    Resolve selection.prefit_detectability.

    Fallback:
        simulation.prefit_detectability

    Disabled by default, preserving all historical runs.
    """

    raw = cfg(
        "selection",
        "prefit_detectability",
        None,
    )

    if raw is None:
        raw = cfg(
            "simulation",
            "prefit_detectability",
            None,
        )

    if raw is None:
        raw = {}

    if isinstance(raw, bool):
        raw = {
            "enabled": bool(raw),
        }

    if not isinstance(raw, dict):
        raise TypeError(
            "selection.prefit_detectability must be a dict, bool, or null."
        )

    out = {
        "enabled": bool(
            raw.get(
                "enabled",
                False,
            )
        ),

        # If true:
        #   simulate + audit + reject before fit, independently of pass/fail.
        "audit_only": bool(
            raw.get(
                "audit_only",
                False,
            )
        ),

        # Recommended:
        #   no_parallax
        #
        # Alternative diagnostic:
        #   current_truth
        "reference_model": str(
            raw.get(
                "reference_model",
                "no_parallax",
            )
        ).strip().lower(),

        # "rubin", "active", or explicit list of bands.
        "bands": raw.get(
            "bands",
            "rubin",
        ),

        "peak_window_tE": float(
            raw.get(
                "peak_window_tE",
                1.0,
            )
        ),

        "min_total_points": int(
            raw.get(
                "min_total_points",
                10,
            )
        ),

        "min_bands": int(
            raw.get(
                "min_bands",
                3,
            )
        ),

        "min_peak_points": int(
            raw.get(
                "min_peak_points",
                5,
            )
        ),

        "min_left_peak_points": int(
            raw.get(
                "min_left_peak_points",
                1,
            )
        ),

        "min_right_peak_points": int(
            raw.get(
                "min_right_peak_points",
                1,
            )
        ),

        "nsigma": float(
            raw.get(
                "nsigma",
                3.0,
            )
        ),

        "min_nsigma_points": int(
            raw.get(
                "min_nsigma_points",
                6,
            )
        ),

        "min_delta_chi2_per_point": float(
            raw.get(
                "min_delta_chi2_per_point",
                2.0,
            )
        ),

        # None means diagnostic only.
        "max_nearest_peak_distance_tE": raw.get(
            "max_nearest_peak_distance_tE",
            None,
        ),
    }

    if out["reference_model"] not in {
        "no_parallax",
        "current_truth",
    }:
        raise ValueError(
            "prefit_detectability.reference_model must be "
            "'no_parallax' or 'current_truth'."
        )

    if out["peak_window_tE"] <= 0:
        raise ValueError(
            "peak_window_tE must be > 0."
        )

    if out["min_total_points"] < 0:
        raise ValueError(
            "min_total_points must be >= 0."
        )

    if out["min_bands"] < 0:
        raise ValueError(
            "min_bands must be >= 0."
        )

    if out["nsigma"] <= 0:
        raise ValueError(
            "nsigma must be > 0."
        )

    if out["min_delta_chi2_per_point"] < 0:
        raise ValueError(
            "min_delta_chi2_per_point must be >= 0."
        )

    nearest = out[
        "max_nearest_peak_distance_tE"
    ]

    if nearest is not None:
        nearest = float(nearest)

        if nearest <= 0:
            raise ValueError(
                "max_nearest_peak_distance_tE must be > 0 or null."
            )

        out[
            "max_nearest_peak_distance_tE"
        ] = nearest

    return out


def _prefit_plain_array(values):
    if hasattr(values, "value"):
        values = values.value

    return np.asarray(
        values,
        dtype=float,
    )


def _prefit_selected_band(
    band_name,
    bands_config,
):
    band_name = str(
        band_name
    )

    if isinstance(
        bands_config,
        str,
    ):
        mode = bands_config.strip().lower()

        if mode == "rubin":
            return band_name in {
                "u",
                "g",
                "r",
                "i",
                "z",
                "y",
            }

        if mode == "active":
            return True

        return (
            band_name.lower()
            == mode
        )

    if isinstance(
        bands_config,
        (list, tuple, set),
    ):
        allowed = {
            str(x)
            for x in bands_config
        }

        return (
            band_name
            in allowed
        )

    raise TypeError(
        "prefit_detectability.bands must be "
        "'rubin', 'active', a band name, or a list."
    )


def _compute_prefit_detectability_metrics(
    pyLIMA_model,
    event_data,
    detect_cfg,
):
    """
    Compute a noise-independent microlensing-vs-constant diagnostic.

    For each band, fit the best weighted constant magnitude:

        m0 = sum(w_i m_i) / sum(w_i)

    using noiseless model magnitudes.

    Then:

        Delta chi2_const =
            sum_i [(m_model_i - m0_band)/sigma_i]^2

    This is an Asimov expected signal metric evaluated at the
    ACTUAL Rubin timestamps and photometric uncertainties.
    """

    out = {
        "detectability_enabled": True,
        "detectability_audit_only": bool(
            detect_cfg[
                "audit_only"
            ]
        ),
        "detectability_reference_model": str(
            detect_cfg[
                "reference_model"
            ]
        ),
        "detectability_noise_independent": True,

        "detectability_n_obs": 0,
        "detectability_n_bands": 0,

        "detectability_n_peak": 0,
        "detectability_n_left_peak": 0,
        "detectability_n_right_peak": 0,

        "detectability_nearest_dt_over_tE": np.nan,

        "detectability_delta_chi2_const_asimov": 0.0,
        "detectability_delta_chi2_per_point": np.nan,

        "detectability_n_nsigma": 0,
        "detectability_max_snr": 0.0,

        "detectability_pass": False,
        "detectability_reasons": "",
    }

    # Save thresholds explicitly.
    out[
        "detectability_threshold_min_total_points"
    ] = int(
        detect_cfg[
            "min_total_points"
        ]
    )

    out[
        "detectability_threshold_min_bands"
    ] = int(
        detect_cfg[
            "min_bands"
        ]
    )

    out[
        "detectability_threshold_min_peak_points"
    ] = int(
        detect_cfg[
            "min_peak_points"
        ]
    )

    out[
        "detectability_threshold_min_left_peak_points"
    ] = int(
        detect_cfg[
            "min_left_peak_points"
        ]
    )

    out[
        "detectability_threshold_min_right_peak_points"
    ] = int(
        detect_cfg[
            "min_right_peak_points"
        ]
    )

    out[
        "detectability_threshold_nsigma"
    ] = float(
        detect_cfg[
            "nsigma"
        ]
    )

    out[
        "detectability_threshold_min_nsigma_points"
    ] = int(
        detect_cfg[
            "min_nsigma_points"
        ]
    )

    out[
        "detectability_threshold_min_delta_chi2_per_point"
    ] = float(
        detect_cfg[
            "min_delta_chi2_per_point"
        ]
    )

    nearest_limit = detect_cfg[
        "max_nearest_peak_distance_tE"
    ]

    out[
        "detectability_threshold_max_nearest_dt_over_tE"
    ] = (
        np.nan
        if nearest_limit is None
        else float(nearest_limit)
    )

    if pyLIMA_model is None:
        out[
            "detectability_reasons"
        ] = "no_model"

        return out

    if not isinstance(
        event_data,
        dict,
    ):
        out[
            "detectability_reasons"
        ] = "event_data_not_dict"

        return out

    try:
        t0 = float(
            event_data[
                "t0"
            ]
        )

        tE = abs(
            float(
                event_data[
                    "tE"
                ]
            )
        )

    except Exception:
        out[
            "detectability_reasons"
        ] = "invalid_t0_or_tE"

        return out

    if (
        not np.isfinite(t0)
        or not np.isfinite(tE)
        or tE <= 0.0
    ):
        out[
            "detectability_reasons"
        ] = "invalid_t0_or_tE"

        return out

    all_times = []

    total_chi2 = 0.0
    total_nsigma = 0
    total_max_snr = 0.0
    n_bands = 0
    n_obs = 0

    for telescope in pyLIMA_model.event.telescopes:

        band = str(
            telescope.name
        )

        if not _prefit_selected_band(
            band,
            detect_cfg[
                "bands"
            ],
        ):
            continue

        lc = getattr(
            telescope,
            "lightcurve",
            None,
        )

        if lc is None or len(lc) == 0:
            continue

        colnames = list(
            getattr(
                lc,
                "colnames",
                [],
            )
        )

        required = {
            "time",
            "mag_model",
            "err_mag",
        }

        if not required.issubset(
            set(colnames)
        ):
            continue

        times = _prefit_plain_array(
            lc[
                "time"
            ]
        )

        model_mag = _prefit_plain_array(
            lc[
                "mag_model"
            ]
        )

        sigma_mag = _prefit_plain_array(
            lc[
                "err_mag"
            ]
        )

        valid = (
            np.isfinite(times)
            & np.isfinite(model_mag)
            & np.isfinite(sigma_mag)
            & (sigma_mag > 0.0)
        )

        times = times[
            valid
        ]

        model_mag = model_mag[
            valid
        ]

        sigma_mag = sigma_mag[
            valid
        ]

        n = int(
            len(times)
        )

        if n == 0:
            continue

        weights = (
            1.0
            / sigma_mag**2
        )

        sum_w = float(
            np.sum(weights)
        )

        if (
            not np.isfinite(sum_w)
            or sum_w <= 0.0
        ):
            continue

        m0 = float(
            np.sum(
                weights
                * model_mag
            )
            / sum_w
        )

        residual_sigma = (
            model_mag
            - m0
        ) / sigma_mag

        residual_sigma = np.asarray(
            residual_sigma,
            dtype=float,
        )

        chi2_band = float(
            np.sum(
                residual_sigma**2
            )
        )

        abs_snr = np.abs(
            residual_sigma
        )

        n_nsigma_band = int(
            np.sum(
                abs_snr
                >= float(
                    detect_cfg[
                        "nsigma"
                    ]
                )
            )
        )

        max_snr_band = float(
            np.max(
                abs_snr
            )
        )

        n_bands += 1
        n_obs += n

        total_chi2 += (
            chi2_band
        )

        total_nsigma += (
            n_nsigma_band
        )

        total_max_snr = max(
            total_max_snr,
            max_snr_band,
        )

        all_times.append(
            times
        )

        prefix = (
            f"detectability_band_{band}"
        )

        out[
            f"{prefix}_n"
        ] = n

        out[
            f"{prefix}_constant_mag"
        ] = m0

        out[
            f"{prefix}_delta_chi2_const"
        ] = chi2_band

        out[
            f"{prefix}_n_nsigma"
        ] = n_nsigma_band

        out[
            f"{prefix}_max_snr"
        ] = max_snr_band

    out[
        "detectability_n_obs"
    ] = int(
        n_obs
    )

    out[
        "detectability_n_bands"
    ] = int(
        n_bands
    )

    out[
        "detectability_delta_chi2_const_asimov"
    ] = float(
        total_chi2
    )

    out[
        "detectability_n_nsigma"
    ] = int(
        total_nsigma
    )

    out[
        "detectability_max_snr"
    ] = float(
        total_max_snr
    )

    if n_obs > 0:

        delta_per_point = (
            total_chi2
            / float(n_obs)
        )

        out[
            "detectability_delta_chi2_per_point"
        ] = float(
            delta_per_point
        )

    else:

        delta_per_point = np.nan

    if all_times:

        times_all = np.concatenate(
            all_times
        )

        dt = (
            times_all
            - t0
        )

        peak_half_width = (
            float(
                detect_cfg[
                    "peak_window_tE"
                ]
            )
            * tE
        )

        peak = (
            np.abs(dt)
            <= peak_half_width
        )

        left = (
            (dt < 0.0)
            & (
                dt
                >= -peak_half_width
            )
        )

        right = (
            (dt > 0.0)
            & (
                dt
                <= peak_half_width
            )
        )

        n_peak = int(
            np.sum(
                peak
            )
        )

        n_left = int(
            np.sum(
                left
            )
        )

        n_right = int(
            np.sum(
                right
            )
        )

        nearest = float(
            np.min(
                np.abs(
                    dt
                )
            )
            / tE
        )

    else:

        n_peak = 0
        n_left = 0
        n_right = 0
        nearest = np.nan

    out[
        "detectability_n_peak"
    ] = n_peak

    out[
        "detectability_n_left_peak"
    ] = n_left

    out[
        "detectability_n_right_peak"
    ] = n_right

    out[
        "detectability_nearest_dt_over_tE"
    ] = nearest

    reasons = []

    if (
        n_obs
        < detect_cfg[
            "min_total_points"
        ]
    ):
        reasons.append(
            "min_total_points"
        )

    if (
        n_bands
        < detect_cfg[
            "min_bands"
        ]
    ):
        reasons.append(
            "min_bands"
        )

    if (
        n_peak
        < detect_cfg[
            "min_peak_points"
        ]
    ):
        reasons.append(
            "min_peak_points"
        )

    if (
        n_left
        < detect_cfg[
            "min_left_peak_points"
        ]
    ):
        reasons.append(
            "min_left_peak_points"
        )

    if (
        n_right
        < detect_cfg[
            "min_right_peak_points"
        ]
    ):
        reasons.append(
            "min_right_peak_points"
        )

    if (
        total_nsigma
        < detect_cfg[
            "min_nsigma_points"
        ]
    ):
        reasons.append(
            "min_nsigma_points"
        )

    if (
        not np.isfinite(
            delta_per_point
        )
        or (
            delta_per_point
            < detect_cfg[
                "min_delta_chi2_per_point"
            ]
        )
    ):
        reasons.append(
            "min_delta_chi2_per_point"
        )

    if nearest_limit is not None:

        if (
            not np.isfinite(
                nearest
            )
            or (
                nearest
                > float(
                    nearest_limit
                )
            )
        ):
            reasons.append(
                "max_nearest_peak_distance_tE"
            )

    out[
        "detectability_pass"
    ] = (
        len(reasons)
        == 0
    )

    out[
        "detectability_reasons"
    ] = ";".join(
        reasons
    )

    return out


def _prefit_force_failure(
    metrics,
    reason,
):
    reasons = [
        item
        for item in str(
            metrics.get(
                "detectability_reasons",
                "",
            )
        ).split(";")
        if item
    ]

    if reason not in reasons:
        reasons.append(
            str(reason)
        )

    metrics[
        "detectability_pass"
    ] = False

    metrics[
        "detectability_reasons"
    ] = ";".join(
        reasons
    )

    return metrics
