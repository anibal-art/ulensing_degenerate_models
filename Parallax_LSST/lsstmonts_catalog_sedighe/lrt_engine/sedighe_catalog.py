"""Sedighe LSSTMONTS catalogue -> prepared rows for roman_rubin.

Functions are moved verbatim from run_lsstmonts_catalog_hidden_parallax.py
(commit deeeaae) with tools/extract_module.py; only this header and the
configuration block below are new. The configuration file is the same
production JSON, given by the environment variable LRT_CONFIG.
"""

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import L_sun, sigma_sb
from astropy.coordinates import SkyCoord

import set_telescopes_pyLIMA as stp

yaml = None
HOME = Path.home()


def log_step(message):
    """Print progress immediately, useful for large catalog reads."""
    print(message, flush=True)


def _load_config(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de configuración: {path}\n"
            "Usá --config /ruta/al/archivo.json"
        )

    suffix = path.suffix.lower()

    with open(path, "r", encoding="utf-8") as file:
        if suffix == ".json":
            config = json.load(file)
        else:
            if yaml is None:
                raise ImportError(
                    "Para leer YAML instalá PyYAML o usá un config JSON."
                )
            config = yaml.safe_load(file)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise TypeError(
            "La raíz del archivo de configuración debe ser un diccionario."
        )

    return config


def cfg(section, key, default=None):
    value = CONFIG.get(section, {})

    if value is None:
        return default

    if not isinstance(value, dict):
        raise TypeError(
            f"La sección '{section}' del config debe ser un diccionario."
        )

    return value.get(key, default)


def topcfg(key, default=None):
    return CONFIG.get(key, default)


def first_config_value(*values, default=None):
    for value in values:
        if value not in (None, ""):
            return value
    return default


def parse_optional_nonnegative_int(value, name):
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {"all", "none", ""}:
        return None

    parsed = int(value)

    if parsed < 0:
        raise ValueError(f"{name} debe ser >= 0, None o 'all'.")

    return parsed


def resolve_catalog_row_window(start, stop):
    start = parse_optional_nonnegative_int(start, "catalog_row_start")
    stop = parse_optional_nonnegative_int(stop, "catalog_row_stop")

    if start is None:
        start = 0

    if stop is not None and stop <= start:
        raise ValueError(
            "catalog_row_stop debe ser mayor que catalog_row_start. "
            f"Recibido start={start}, stop={stop}."
        )

    return int(start), stop


def _mag_to_flux(magnitude, zeropoint):
    return 10.0 ** (0.4 * (float(zeropoint) - float(magnitude)))


def _canonical_catalog_band(telescope_name):
    """
    Map pyLIMA/Rubin telescope names to the six catalog filter names.
    """

    name = str(telescope_name).strip()
    lower = name.lower()

    if lower in {"w149", "roman", "roman_w149", "y"}:
        return "y"

    if lower in {"u", "g", "r", "i", "z"}:
        return lower

    for band in ("u", "g", "r", "i", "z", "y"):
        if lower.endswith("_" + band):
            return band

    return None


def _canonical_maf_filter(filter_name):
    """
    Canonicalize OpSim/MAF filter names to u,g,r,i,z,y.

    This also accepts names with suffixes such as r_57.
    """

    lower = str(filter_name).strip().lower()

    if lower in set(CATALOG_BANDS):
        return lower

    for band in CATALOG_BANDS:
        if lower.startswith(band + "_") or lower.endswith("_" + band):
            return band

    if len(lower) > 0 and lower[0] in set(CATALOG_BANDS):
        return lower[0]

    return lower


def _find_dataslice_filter_column(dataSlice):
    """
    Return the filter-column name used by the current MAF/dataSlice.
    """

    names = dataSlice.dtype.names

    if names is None:
        raise RuntimeError(
            "dataSlice no tiene dtype.names; no puedo identificar filtros."
        )

    for candidate in [
        "filter",
        "band",
        "filtername",
        "filterName",
        "filter_name",
    ]:
        if candidate in names:
            return candidate

    raise KeyError(
        "No encontré columna de filtro en dataSlice. "
        f"Columnas disponibles: {names}"
    )


def _visible_bands_from_row(base_row, mode=None):
    """
    Define which catalog bands are allowed to enter the simulated event.

    Preferred mode:
        detection_flag:
            DetectionFlag_band == 1 means the source is visible in that band.

    The blending factors are not used as availability flags in this mode.
    They are only used as source fractions for active bands.
    """

    if mode is None:
        mode = BAND_AVAILABILITY_MODE

    mode = str(mode).lower()

    if mode == "all":
        return list(CATALOG_BANDS)

    if mode == "blend_positive":
        return [
            band for band in CATALOG_BANDS
            if float(base_row.get(f"blend_{band}", 0.0)) > 0.0
        ]

    if mode == "detection_flag":
        missing = [
            f"DetectionFlag_{band}"
            for band in CATALOG_BANDS
            if f"DetectionFlag_{band}" not in base_row.index
        ]

        if missing:
            raise KeyError(
                "band_availability='detection_flag' requiere estas columnas: "
                f"{missing}"
            )

        visible_bands = []

        for band in CATALOG_BANDS:
            flag = float(base_row[f"DetectionFlag_{band}"])

            if not np.isfinite(flag):
                raise ValueError(
                    f"DetectionFlag_{band} no es finito: {flag}"
                )

            if int(round(flag)) == 1:
                visible_bands.append(band)

        return visible_bands

    raise ValueError(
        f"band availability mode desconocido: {mode!r}"
    )


def _count_data_columns(data_file):
    with open(data_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return len(line.split())
    raise ValueError(f"El catálogo está vacío: {data_file}")


def _normalized_column_name(name):
    return " ".join(str(name).strip().split())


def _compact_column_name(name):
    return (
        _normalized_column_name(name)
        .lower()
        .replace("\\", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def rename_sedighe_xi_columns(data):
    """
    Detecta las columnas nuevas de Sedighe aunque el archivo columns
    contenga barras, espacios dobles o descripciones largas.

    En particular, mapea la columna
        \\xi (degree): The angle of the lens-source trajectory ...
    a xi_catalog.
    """

    rename = {}

    for col in data.columns:
        clean = _normalized_column_name(col)
        compact = _compact_column_name(col)

        # La columna xi explícita del nuevo catálogo.
        if (
            compact.startswith("xi(degree):")
            or compact.startswith("xi(degree)")
            or compact.startswith("xidegree")
            or compact.startswith("xi(rad):")
            or compact.startswith("xirad")
            or (
                "angleofthelenssourcetrajectory" in compact
                and compact.startswith("xi")
            )
        ):
            rename[col] = "xi_catalog"

        # Alpha queda como metadato, no como fuente de xi.
        elif compact.startswith("alpha(degree)") or compact.startswith("alphadegree"):
            rename[col] = "alpha_catalog"

        elif compact.startswith("murel1(mas/days)"):
            rename[col] = "murel_1_mas_per_day"

        elif compact.startswith("murel2(mas/days)"):
            rename[col] = "murel_2_mas_per_day"

        elif compact.startswith("detectionflagu"):
            rename[col] = "DetectionFlag_u"
        elif compact.startswith("detectionflagg"):
            rename[col] = "DetectionFlag_g"
        elif compact.startswith("detectionflagr"):
            rename[col] = "DetectionFlag_r"
        elif compact.startswith("detectionflagi"):
            rename[col] = "DetectionFlag_i"
        elif compact.startswith("detectionflagz"):
            rename[col] = "DetectionFlag_z"
        elif compact.startswith("detectionflagy"):
            rename[col] = "DetectionFlag_y"

    return data.rename(columns=rename)


def load_raw_catalog(
    columns_file,
    data_file,
    nrows=None,
    catalog_row_start=0,
    catalog_row_stop=None,
):
    catalog_row_start, catalog_row_stop = resolve_catalog_row_window(
        catalog_row_start,
        catalog_row_stop,
    )

    if catalog_row_stop is not None:
        window_nrows = int(catalog_row_stop - catalog_row_start)
        if nrows is None:
            nrows = window_nrows
        else:
            nrows = min(int(nrows), window_nrows)

    log_step(f"[catalog] columns_file = {columns_file}")
    log_step(f"[catalog] data_file    = {data_file}")
    log_step(
        "[catalog] row window   = "
        f"[{catalog_row_start}, "
        f"{catalog_row_stop if catalog_row_stop is not None else 'EOF'})"
    )
    if nrows is not None:
        log_step(f"[catalog] read_nrows   = {nrows}")
    else:
        log_step("[catalog] read_nrows   = all")

    if not columns_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo de columnas: {columns_file}"
        )

    if not data_file.exists():
        raise FileNotFoundError(
            f"No existe el catálogo: {data_file}"
        )

    # Leer el archivo de columnas línea por línea. No usamos pd.read_csv
    # porque algunas descripciones contienen comas, por ejemplo "i.e.,".
    with open(columns_file, "r", encoding="utf-8") as file:
        column_names = [
            _normalized_column_name(line)
            for line in file
            if line.strip()
        ]

    log_step(f"[catalog] columns read = {len(column_names)}")

    n_data_columns = _count_data_columns(data_file)
    log_step(f"[catalog] data columns in first non-empty row = {n_data_columns}")

    # If the data file has one extra value with respect to the columns file,
    # assume the missing last column is the configured parallax angle column.
    # For the new Sedighe catalog this should normally be xi.
    if n_data_columns == len(column_names) + 1:
        column_names.append(PARALLAX_ANGLE_COLUMN)
        print(
            "[catalog] Added missing last-column name from YAML:",
            PARALLAX_ANGLE_COLUMN,
        )
    elif n_data_columns != len(column_names):
        raise ValueError(
            f"El catálogo tiene {n_data_columns} valores por fila, pero "
            f"columns contiene {len(column_names)} nombres."
        )

    log_step("[catalog] starting pd.read_csv ...")

    data = pd.read_csv(
        data_file,
        sep=r"\s+",
        header=None,
        names=column_names,
        engine="c",
        skiprows=catalog_row_start if catalog_row_start > 0 else None,
        nrows=nrows,
    )

    log_step(f"[catalog] finished pd.read_csv. shape = {data.shape}")

    # Exact configured angle-column name takes precedence.
    if PARALLAX_ANGLE_COLUMN in data.columns:
        data = data.rename(
            columns={PARALLAX_ANGLE_COLUMN: "xi_catalog"}
        )

    data = data.rename(columns=COLUMN_ALIASES)
    data = rename_sedighe_xi_columns(data)

    if "alpha_catalog" not in data.columns:
        data["alpha_catalog"] = np.nan

    log_step("[catalog] column renaming finished.")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "No pude identificar estas columnas:\n"
            + "\n".join(f"- {column}" for column in missing)
            + "\n\nColumnas disponibles:\n"
            + "\n".join(map(str, data.columns))
        )

    log_step("[catalog] required columns found.")
    return data


def _resolve_angle_unit(values):
    requested = PARALLAX_ANGLE_UNIT

    if requested in {"rad", "radian", "radians"}:
        return "radians"
    if requested in {"deg", "degree", "degrees"}:
        return "degrees"

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    if len(finite) == 0:
        raise ValueError("No hay valores finitos en la columna alpha/xi.")

    if np.nanmax(np.abs(finite)) <= 2.0 * np.pi + 0.1:
        return "radians"

    return "degrees"


def prepare_catalog(raw_catalog, max_base_events, catalog_row_offset=0):
    data = raw_catalog.copy().reset_index(drop=True)

    catalog_row_offset = int(catalog_row_offset)

    data["catalog_row"] = (
        catalog_row_offset
        + np.arange(
            len(data),
            dtype=int,
        )
    )

    numeric_columns = list(
        dict.fromkeys(
            REQUIRED_COLUMNS
            + [
                "mu_rel_catalog_masyr",
                "mu_rel_catalog_masday",
                "murel_1_mas_per_day",
                "murel_2_mas_per_day",
                "alpha_catalog",
                "xi_catalog",
                "n_data_catalog",
                "delta_chi2_catalog",
                "fwhm_catalog_days",
                "DetectionFlag_u",
                "DetectionFlag_g",
                "DetectionFlag_r",
                "DetectionFlag_i",
                "DetectionFlag_z",
                "DetectionFlag_y",
            ]
        )
    )

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # Explicit scientific assumption for this run:
    # use the catalog xi column as the relative trajectory angle.
    data["xi_catalog"] = data[
        "xi_catalog"
    ].to_numpy(dtype=float)

    resolved_angle_unit = _resolve_angle_unit(
        data["xi_catalog"].to_numpy(dtype=float)
    )

    if resolved_angle_unit == "degrees":
        data["xi_rad"] = np.deg2rad(
            data["xi_catalog"].to_numpy(dtype=float)
        )
    else:
        data["xi_rad"] = data[
            "xi_catalog"
        ].to_numpy(dtype=float)

    # Normalize only the angle used in trigonometric calculations.
    data["xi_rad"] = np.mod(
        data["xi_rad"].to_numpy(dtype=float),
        2.0 * np.pi,
    )

    data["xi_deg"] = np.rad2deg(
        data["xi_rad"].to_numpy(dtype=float)
    )

    # Backward-compatible aliases used by a few generic output routines.
    data["alpha_rad"] = data["xi_rad"]
    data["alpha_deg"] = data["xi_deg"]
    data["alpha_unit_resolved"] = resolved_angle_unit
    data["angle_semantics"] = "xi_catalog_column"

    # The test catalog had RA=Dec=0. Derive ICRS coordinates from l,b.
    coordinates = SkyCoord(
        l=data["l_deg"].to_numpy(dtype=float) * u.deg,
        b=data["b_deg"].to_numpy(dtype=float) * u.deg,
        frame="galactic",
    )

    data["ra"] = coordinates.icrs.ra.deg
    data["dec"] = coordinates.icrs.dec.deg

    # Preserve the catalog tE even if the column labeled mu_rel uses a
    # different unit/convention.
    data["mu_rel_for_pipeline_masyr"] = (
        data["thetaE_mas"]
        * 365.25
        / data["tE_catalog_days"]
    )

    # IMPORTANT:
    # t0_catalog_days is a relative time from Sedighe's catalog.
    # It must NOT be converted with a fixed global JD origin.
    # The absolute t0 is resolved per event after reading the relevant
    # OpSim/MAF dataSlice:
    #
    #     t0_jd = first_maf_timestamp_jd + t0_catalog_days
    #
    # Keep the old fixed-origin value only as a diagnostic.
    data["t0_jd_original_global_zero"] = (
        T0_ZERO_JD
        + data["t0_catalog_days"]
    )
    data["t0_reference_jd"] = np.nan
    data["t0_reference_mjd"] = np.nan
    data["t0_origin"] = "unresolved_requires_first_maf_timestamp"
    data["t0_jd"] = np.nan

    data["D_L"] = 1000.0 * data["lens_distance_kpc"]
    data["D_S"] = 1000.0 * data["source_distance_kpc"]
    data["D_L_kpc"] = data["lens_distance_kpc"]
    data["D_S_kpc"] = data["source_distance_kpc"]
    data["mu_rel"] = data["mu_rel_for_pipeline_masyr"]

    # Source radius/luminosity consistent with the catalog rho.
    theta_s_rad = (
        data["rho"].to_numpy(dtype=float)
        * data["thetaE_mas"].to_numpy(dtype=float)
        * u.mas
    ).to(u.rad).value

    source_radius = (
        theta_s_rad
        * data["source_distance_kpc"].to_numpy(dtype=float)
        * u.kpc
    ).to(u.R_sun)

    source_temperature = (
        10.0 ** data["logTe"].to_numpy(dtype=float)
    ) * u.K

    luminosity_ratio = (
        4.0
        * np.pi
        * sigma_sb
        * source_radius.to(u.m) ** 2
        * source_temperature ** 4
        / L_sun
    ).decompose().value

    data["source_radius_rsun_catalog"] = source_radius.value
    data["logL"] = np.log10(luminosity_ratio)
    data["W149"] = data["Y"]

    data["gall"] = data["l_deg"]
    data["galb"] = data["b_deg"]
    data["lens_ra"] = data["ra"]
    data["lens_dec"] = data["dec"]

    finite_required = [
        "ra",
        "dec",
        "D_L",
        "D_S",
        "mu_rel",
        "logTe",
        "logL",
        "lens_mass_msun",
        "u0",
        "t0_catalog_days",
        "piE",
        "rho",
        "alpha_rad",
        *SOURCE_MAG_COLUMNS,
        *BLEND_COLUMNS,
    ]

    valid = np.ones(len(data), dtype=bool)
    data["invalid_reason"] = ""

    def mark_invalid(mask, reason):
        """Attach a row-level invalid reason without aborting the full run."""

        mask = np.asarray(mask, dtype=bool)

        if len(mask) != len(data):
            raise ValueError(
                f"Máscara inválida para {reason}: "
                f"len(mask)={len(mask)}, len(data)={len(data)}"
            )

        if not np.any(mask):
            return

        previous = data.loc[mask, "invalid_reason"].astype(str)
        empty = previous.isin(["", "nan", "None"])

        data.loc[mask, "invalid_reason"] = np.where(
            empty,
            reason,
            previous + ";" + reason,
        )

    for column in finite_required:
        valid &= np.isfinite(
            data[column].to_numpy(dtype=float)
        )

    valid &= data["D_L"].to_numpy(dtype=float) > 0.0
    valid &= (
        data["D_S"].to_numpy(dtype=float)
        > data["D_L"].to_numpy(dtype=float)
    )
    valid &= data["mu_rel"].to_numpy(dtype=float) > 0.0
    valid &= data["lens_mass_msun"].to_numpy(dtype=float) > 0.0
    valid &= data["tE_catalog_days"].to_numpy(dtype=float) > 0.0
    valid &= data["t0_catalog_days"].to_numpy(dtype=float) >= 0.0
    valid &= data["piE"].to_numpy(dtype=float) >= 0.0
    valid &= data["rho"].to_numpy(dtype=float) > 0.0

    mark_invalid(
        ~valid,
        "invalid_basic_finite_or_physical_quantity",
    )

    blend_matrix = data[
        BLEND_COLUMNS
    ].to_numpy(dtype=float)

    blend_range_valid = np.all(
        np.isfinite(blend_matrix)
        & (blend_matrix >= 0.0)
        & (blend_matrix <= 1.0),
        axis=1,
    )

    if BAND_AVAILABILITY_MODE == "detection_flag":
        missing_flags = [
            column for column in DETECTION_FLAG_COLUMNS
            if column not in data.columns
        ]

        if missing_flags:
            raise KeyError(
                "sedighe.band_availability='detection_flag' requiere "
                f"estas columnas: {missing_flags}"
            )

        flag_matrix = data[
            DETECTION_FLAG_COLUMNS
        ].to_numpy(dtype=float)

        flag_finite = np.all(
            np.isfinite(flag_matrix),
            axis=1,
        )

        flag_binary = np.all(
            np.isin(
                np.rint(flag_matrix).astype(int),
                [0, 1],
            ),
            axis=1,
        )

        flag_valid = flag_finite & flag_binary
        availability_matrix = (
            np.rint(flag_matrix).astype(int) == 1
        )

    elif BAND_AVAILABILITY_MODE == "blend_positive":
        flag_valid = np.ones(len(data), dtype=bool)
        availability_matrix = blend_matrix > 0.0

    elif BAND_AVAILABILITY_MODE == "all":
        flag_valid = np.ones(len(data), dtype=bool)
        availability_matrix = np.ones_like(
            blend_matrix,
            dtype=bool,
        )

    else:
        raise ValueError(
            f"band_availability desconocido: {BAND_AVAILABILITY_MODE!r}"
        )

    visible_filter_count = np.sum(
        availability_matrix,
        axis=1,
    )

    data["catalog_visible_filter_count"] = (
        visible_filter_count.astype(int)
    )

    data["catalog_available_bands"] = [
        ",".join(
            band
            for band, is_available in zip(CATALOG_BANDS, row)
            if is_available
        )
        for row in availability_matrix
    ]

    for k, band in enumerate(CATALOG_BANDS):
        data[f"catalog_band_available_{band}"] = (
            availability_matrix[:, k].astype(int)
        )

    enough_visible_filters = (
        visible_filter_count
        >= BLENDING_MINIMUM_VISIBLE_FILTERS
    )

    # Only active/available bands need strictly positive source fraction.
    # Inactive bands may carry a catalog value, but it is not used.
    source_fraction_positive_in_available = np.ones(
        len(data),
        dtype=bool,
    )

    for k, band in enumerate(CATALOG_BANDS):
        source_fraction_positive_in_available &= (
            (~availability_matrix[:, k])
            | (blend_matrix[:, k] > 0.0)
        )

    mark_invalid(~blend_range_valid, "invalid_blend_source_fraction_range")
    mark_invalid(~flag_valid, "invalid_detection_flag")
    mark_invalid(~enough_visible_filters, "too_few_visible_filters")
    mark_invalid(
        ~source_fraction_positive_in_available,
        "visible_band_with_nonpositive_source_fraction",
    )

    blend_valid = (
        blend_range_valid
        & flag_valid
        & enough_visible_filters
        & source_fraction_positive_in_available
    )

    if BLENDING_STRICT and np.any(~blend_range_valid):
        bad = data.loc[
            ~blend_range_valid,
            ["catalog_row", "invalid_reason", *BLEND_COLUMNS],
        ].head(20)

        print(
            "[warning] Hay filas con factores de blending/source fraction "
            "fuera de rango o no finitos. Estas filas se marcarán como "
            "inválidas y no se simularán. Primeros casos:"
        )
        print(bad.to_string(index=False))

    if BLENDING_STRICT and np.any(~flag_valid):
        bad = data.loc[
            ~flag_valid,
            [
                "catalog_row",
                "invalid_reason",
                *[
                    column for column in DETECTION_FLAG_COLUMNS
                    if column in data.columns
                ],
            ],
        ].head(20)

        print(
            "[warning] Hay filas con DetectionFlag_* no finitas o no "
            "binarias. Estas filas se marcarán como inválidas y no se "
            "simularán. Primeros casos:"
        )
        print(bad.to_string(index=False))

    if BLENDING_STRICT and np.any(~enough_visible_filters):
        bad = data.loc[
            ~enough_visible_filters,
            [
                "catalog_row",
                "invalid_reason",
                "catalog_visible_filter_count",
                "catalog_available_bands",
                *[
                    column for column in DETECTION_FLAG_COLUMNS
                    if column in data.columns
                ],
                *BLEND_COLUMNS,
            ],
        ].head(20)

        print(
            "[warning] Hay eventos con menos filtros visibles que el "
            f"mínimo configurado ({BLENDING_MINIMUM_VISIBLE_FILTERS}) "
            f"usando band_availability={BAND_AVAILABILITY_MODE}. "
            "Estas filas se marcarán como inválidas y no se simularán. "
            "Primeros casos:"
        )
        print(bad.to_string(index=False))

    if BLENDING_STRICT and np.any(~source_fraction_positive_in_available):
        bad = data.loc[
            ~source_fraction_positive_in_available,
            [
                "catalog_row",
                "invalid_reason",
                "catalog_available_bands",
                *BLEND_COLUMNS,
            ],
        ].head(20)

        print(
            "[warning] Hay bandas catalogadas como visibles pero con "
            "source fraction <= 0. Para bandas activas se requiere "
            "0 < F_source/F_total <= 1. Estas filas se marcarán como "
            "inválidas y no se simularán. Primeros casos:"
        )
        print(bad.to_string(index=False))

    valid &= blend_valid

    invalid = data.loc[~valid].copy()
    data = data.loc[valid].copy().reset_index(drop=True)

    if (
        max_base_events is not None
        and len(data) > max_base_events
    ):
        selected_indices = (
            data.sample(
                n=max_base_events,
                random_state=RANDOM_SEED,
            )
            .index
            .sort_values()
        )

        data = (
            data.loc[selected_indices]
            .copy()
            .reset_index(drop=True)
        )

    data.attrs["xi_unit_resolved"] = resolved_angle_unit
    data.attrs["alpha_unit_resolved"] = resolved_angle_unit

    return data, invalid


def parallax_components_from_catalog(row):
    """
    Convert the catalog amplitude and xi into pyLIMA (North, East)
    parallax components.

    Assumption
    ----------
    The catalog column ``xi`` is used explicitly.  The column ``alpha``,
    if present, is retained only as metadata and is not used here.

    For the paper-local Galactic tangent basis:

        n1 = direction of increasing Galactic longitude
        n2 = direction of increasing Galactic latitude

    the vector is

        piE_n1 = piE * cos(xi)
        piE_n2 = piE * sin(xi)

    and is then rotated into the ICRS North/East basis expected by pyLIMA.
    """

    xi_rad = float(row["xi_rad"])
    piE = float(row["piE"])

    diagnostics = {
        "galactic_n1_basis_pa_icrs_deg": np.nan,
        "galactic_n2_basis_pa_icrs_deg": np.nan,
        # Retained aliases for compatibility with previous result readers.
        "galactic_l_basis_pa_icrs_deg": np.nan,
        "galactic_b_basis_pa_icrs_deg": np.nan,
    }

    if PARALLAX_ANGLE_BASIS == "icrs_en":
        if PARALLAX_COMPONENT_CONVENTION == "east_cos_north_sin":
            piEE = piE * np.cos(xi_rad)
            piEN = piE * np.sin(xi_rad)
        else:
            piEN = piE * np.cos(xi_rad)
            piEE = piE * np.sin(xi_rad)

        return float(piEN), float(piEE), diagnostics

    # Paper-local basis:
    # n1 = +l and n2 = +b on the sky.
    # Astropy position_angle is measured east of ICRS north.
    l_deg = float(row["l_deg"])
    b_deg = float(row["b_deg"])

    center_gal = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        frame="galactic",
    )
    center_icrs = center_gal.icrs

    epsilon = 1.0e-5 * u.deg
    cos_b = np.cos(np.deg2rad(b_deg))

    if abs(cos_b) < 1.0e-8:
        raise ValueError(
            "No se puede construir la base n1 cerca "
            "de un polo galáctico."
        )

    plus_n1 = SkyCoord(
        l=(l_deg * u.deg + epsilon / cos_b),
        b=b_deg * u.deg,
        frame="galactic",
    ).icrs

    plus_n2 = SkyCoord(
        l=l_deg * u.deg,
        b=(b_deg * u.deg + epsilon),
        frame="galactic",
    ).icrs

    pa_n1 = float(
        center_icrs.position_angle(plus_n1).to_value(u.rad)
    )
    pa_n2 = float(
        center_icrs.position_angle(plus_n2).to_value(u.rad)
    )

    piE_n1 = piE * np.cos(xi_rad)
    piE_n2 = piE * np.sin(xi_rad)

    # A tangent vector with position angle PA has components
    # North = amplitude*cos(PA), East = amplitude*sin(PA).
    piEN = (
        piE_n1 * np.cos(pa_n1)
        + piE_n2 * np.cos(pa_n2)
    )
    piEE = (
        piE_n1 * np.sin(pa_n1)
        + piE_n2 * np.sin(pa_n2)
    )

    pa_n1_deg = float(np.degrees(pa_n1))
    pa_n2_deg = float(np.degrees(pa_n2))

    diagnostics["galactic_n1_basis_pa_icrs_deg"] = pa_n1_deg
    diagnostics["galactic_n2_basis_pa_icrs_deg"] = pa_n2_deg
    diagnostics["galactic_l_basis_pa_icrs_deg"] = pa_n1_deg
    diagnostics["galactic_b_basis_pa_icrs_deg"] = pa_n2_deg

    recovered = float(np.hypot(piEN, piEE))

    if not np.isclose(
        recovered,
        piE,
        rtol=1.0e-7,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "La transformación de piE desde la base (n1,n2) "
            "a (North,East) no conservó la amplitud: "
            f"input={piE}, output={recovered}."
        )

    return float(piEN), float(piEE), diagnostics


def validate_t0_first_maf_timestamp(base_row, context=""):
    """
    Enforce the only allowed t0 convention for this runner.

    Sedighe's catalog t0 is relative to the first OpSim/MAF timestamp
    read for the selected field/source:

        t0_jd = t0_reference_jd + t0_catalog_days

    Any caller that tries to use the provisional/fixed-origin t0 must fail.
    """

    prefix = f"[{context}] " if context else ""

    origin = str(
        base_row.get("t0_origin", "")
    )

    if origin != "first_maf_timestamp":
        raise RuntimeError(
            prefix
            + "t0_jd todavía no fue resuelto desde el primer timestamp "
            "OpSim/MAF del campo. Llamá primero a "
            "apply_t0_from_first_maf_timestamp(base_row, config). "
            f"t0_origin actual={origin!r}."
        )

    t0_reference_jd = float(base_row.get("t0_reference_jd", np.nan))
    t0_catalog_days = float(base_row.get("t0_catalog_days", np.nan))
    t0_jd = float(base_row.get("t0_jd", np.nan))

    if not np.isfinite(t0_reference_jd):
        raise RuntimeError(
            prefix + "t0_reference_jd no es finito."
        )

    if not np.isfinite(t0_catalog_days):
        raise RuntimeError(
            prefix + "t0_catalog_days no es finito."
        )

    if t0_catalog_days < 0.0:
        raise RuntimeError(
            prefix
            + "t0_catalog_days es negativo. En este runner debe estar "
            "medido como días desde el primer timestamp OpSim/MAF. "
            f"Valor={t0_catalog_days}."
        )

    if not np.isfinite(t0_jd):
        raise RuntimeError(
            prefix + "t0_jd no es finito."
        )

    expected = t0_reference_jd + t0_catalog_days

    if not np.isclose(t0_jd, expected, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            prefix
            + "Convención temporal inválida para t0. Debe cumplirse "
            "t0_jd = t0_reference_jd + t0_catalog_days. "
            f"t0_jd={t0_jd:.9f}, "
            f"t0_reference_jd={t0_reference_jd:.9f}, "
            f"t0_catalog_days={t0_catalog_days:.9f}, "
            f"expected={expected:.9f}."
        )

    return True


def apply_t0_from_first_maf_timestamp(base_row, config):
    """
    Resolve t0_jd using the allowed convention:

        t0_jd = first relevant OpSim/MAF timestamp + t0_catalog_days

    Relevant means:
        - same coordinate / Rubin pointing mode as the simulation;
        - only bands available for this catalog event.

    With sedighe.band_availability='detection_flag', the reference timestamp
    is computed using only filters with DetectionFlag_band == 1.

    The photometric m5/5-sigma filter is not used to define this reference,
    because the simulated magnification is needed before that filter exists.
    """

    import set_telescopes_pyLIMA as stp

    base_row = base_row.copy()

    if not bool(config.get("use_rubin", True)):
        raise RuntimeError(
            "t0_origin=first_maf_timestamp requiere use_rubin=True."
        )

    if "t0_catalog_days" not in base_row.index:
        raise RuntimeError(
            "Falta t0_catalog_days; no puedo construir t0_jd desde el "
            "primer timestamp OpSim/MAF."
        )

    t0_catalog_days = float(base_row["t0_catalog_days"])

    if not np.isfinite(t0_catalog_days):
        raise RuntimeError(
            f"t0_catalog_days no finito: {t0_catalog_days}."
        )

    if t0_catalog_days < 0.0:
        raise RuntimeError(
            "t0_catalog_days es negativo. Para esta convención debe ser "
            "un tiempo positivo medido desde el primer timestamp OpSim/MAF "
            f"del campo. Valor={t0_catalog_days}."
        )

    source_ra = float(base_row["ra"])
    source_dec = float(base_row["dec"])

    visible_bands = _visible_bands_from_row(
        base_row,
        mode=BAND_AVAILABILITY_MODE,
    )

    if len(visible_bands) == 0:
        raise RuntimeError(
            "No hay bandas visibles según "
            f"band_availability={BAND_AVAILABILITY_MODE}."
        )

    tel_kwargs = {
        "path_ephemerides": config["path_ephemerides"],
        "time_window": None,
        "use_roman": bool(config.get("use_roman", False)),
        "use_rubin": bool(config.get("use_rubin", True)),
        "Ra": source_ra,
        "Dec": source_dec,
        "rubin_pointing_mode": config.get("rubin_pointing_mode", "source"),
        "rubin_cache_cell_deg": config.get("rubin_cache_cell_deg", None),
    }

    tel_parameters = inspect.signature(stp.tel_roman_rubin).parameters

    optional_rubin_path_kwargs = {
        "opsim_db_path": config.get("rubin_opsim_db_path", None),
        "rubin_sim_data_dir": config.get("rubin_sim_data_dir", None),
        "rubin_throughputs_dir": config.get("rubin_throughputs_dir", None),
    }

    for key, value in optional_rubin_path_kwargs.items():
        if key in tel_parameters and value not in (None, ""):
            tel_kwargs[key] = value

    _, dataSlice, _ = stp.tel_roman_rubin(**tel_kwargs)

    if dataSlice is None or len(dataSlice) == 0:
        raise RuntimeError(
            "No hay observaciones Rubin/MAF para calcular el primer timestamp."
        )

    if "observationStartMJD" not in dataSlice.dtype.names:
        raise RuntimeError(
            "dataSlice no contiene la columna observationStartMJD; "
            "no puedo definir el origen local de t0."
        )

    filter_column = _find_dataslice_filter_column(dataSlice)

    mjd = np.asarray(
        dataSlice["observationStartMJD"],
        dtype=float,
    )

    jd = mjd + 2400000.5

    filters = np.asarray(
        [
            _canonical_maf_filter(value)
            for value in dataSlice[filter_column]
        ],
        dtype=str,
    )

    finite_mask = np.isfinite(jd)
    finite = jd[finite_mask]

    if len(finite) == 0:
        raise RuntimeError(
            "dataSlice no contiene observationStartMJD finitos."
        )

    raw_first_timestamp_jd = float(np.min(finite))

    visible_mask = (
        finite_mask
        & np.isin(filters, visible_bands)
    )

    visible_jd = jd[visible_mask]
    visible_filters = filters[visible_mask]

    if len(visible_jd) == 0:
        raise RuntimeError(
            "No hay observaciones MAF en las bandas visibles del catálogo. "
            f"visible_bands={visible_bands}, "
            f"available filters in dataSlice={sorted(set(filters.tolist()))}."
        )

    first_index = int(np.argmin(visible_jd))
    first_timestamp_jd = float(visible_jd[first_index])
    first_filter = str(visible_filters[first_index])

    old_t0_jd = float(base_row.get("t0_jd", np.nan))

    legacy_t0_jd = float(
        base_row.get("t0_jd_original_global_zero", np.nan)
    )

    if not np.isfinite(legacy_t0_jd) and np.isfinite(old_t0_jd):
        legacy_t0_jd = old_t0_jd

    new_t0_jd = first_timestamp_jd + t0_catalog_days

    base_row["t0_jd_original_global_zero"] = legacy_t0_jd
    base_row["t0_reference_raw_first_maf_jd"] = raw_first_timestamp_jd
    base_row["t0_reference_jd"] = first_timestamp_jd
    base_row["t0_reference_mjd"] = first_timestamp_jd - 2400000.5
    base_row["t0_reference_source"] = (
        "min(dataSlice.observationStartMJD in catalog-visible bands)"
    )
    base_row["t0_reference_band_availability_mode"] = BAND_AVAILABILITY_MODE
    base_row["t0_reference_visible_bands"] = ",".join(visible_bands)
    base_row["t0_reference_first_filter"] = first_filter
    base_row["t0_reference_n_obs_raw"] = int(len(finite))
    base_row["t0_reference_n_obs_visible_bands"] = int(len(visible_jd))
    base_row["t0_reference_n_obs"] = int(len(visible_jd))
    base_row["t0_origin"] = "first_maf_timestamp"
    base_row["t0_jd"] = new_t0_jd

    validate_t0_first_maf_timestamp(
        base_row,
        context="apply_t0_from_first_maf_timestamp",
    )

    print(
        "t0 convention applied: "
        f"t0_jd={new_t0_jd:.8f} = "
        f"first_visible_maf_jd={first_timestamp_jd:.8f} + "
        f"t0_catalog_days={t0_catalog_days:.8f}. "
        f"visible_bands={visible_bands}, first_filter={first_filter}, "
        f"raw_first_maf_jd={raw_first_timestamp_jd:.8f}"
    )

    return base_row


# ============================================================================
# Configuration (same keys and defaults as the historical runner)
# ============================================================================

CONFIG_PATH = Path(os.environ["LRT_CONFIG"]).expanduser().resolve()
CONFIG_DIR = CONFIG_PATH.parent
CONFIG = _load_config(CONFIG_PATH)

PARALLAX_ANGLE_COLUMN = str(
    first_config_value(
        cfg("sedighe", "angle_column", None),
        cfg("parallax", "angle_column", None),
        default="xi",
    )
)

PARALLAX_ANGLE_UNIT = str(
    first_config_value(
        cfg("sedighe", "angle_unit", None),
        cfg("parallax", "angle_unit", None),
        default="auto",
    )
).lower()

PARALLAX_ANGLE_BASIS = str(
    first_config_value(
        cfg("sedighe", "angle_basis", None),
        cfg("parallax", "angle_basis", None),
        default="galactic_n1n2",
    )
).lower()

PARALLAX_COMPONENT_CONVENTION = str(
    first_config_value(
        cfg("sedighe", "component_convention", None),
        cfg("parallax", "component_convention", None),
        default="east_cos_north_sin",
    )
).lower()

T0_ZERO_JD = float(
    first_config_value(
        cfg("input", "t0_zero_jd", None),
        cfg("catalog", "t0_zero_jd", None),
        default=2460413.013828608,
    )
)

RANDOM_SEED = int(
    cfg("execution", "random_seed", 20260728)
)

BLENDING_STRICT = bool(
    cfg("blending", "strict", True)
)

BLENDING_MINIMUM_VISIBLE_FILTERS = int(
    first_config_value(
        cfg("sedighe", "minimum_visible_filters", None),
        cfg("blending", "minimum_visible_filters", None),
        default=3,
    )
)

BAND_AVAILABILITY_MODE = str(
    first_config_value(
        cfg("sedighe", "band_availability", None),
        cfg("selection", "band_availability", None),
        cfg("blending", "band_availability", None),
        default="detection_flag",
    )
).lower()

COLUMN_ALIASES = {
    "Number_lensing": "catalog_event_id",
    "Galactic latitude (degree)": "b_deg",
    "Galactic Longitude (degree)": "l_deg",
    "Right ascention (degree)": "ra_catalog_deg",
    "Right ascention  (degree)": "ra_catalog_deg",
    "Declination (degree)": "dec_catalog_deg",
    "Lens_mass (solar mass)": "lens_mass_msun",
    "Lens_Distance (kpc)": "lens_distance_kpc",
    "Source_distance (kpc)": "source_distance_kpc",
    "Log10[Teff_source]": "logTe",
    "Einstein crossing time (days)": "tE_catalog_days",
    "time of closest approach (days)": "t0_catalog_days",
    "lens-source relative angular velocity (mas/years)":
        "mu_rel_catalog_masyr",
    "lens-source relative angular velocity (mas/days)":
        "mu_rel_catalog_masday",
    "Lens impact parameter": "u0",
    "Angular Einstein radius (mas)": "thetaE_mas",
    "Normalized Parallax amplitude": "piE",
    "Normalized Parallax amplitude (piE)": "piE",
    "Normalized source radius": "rho",
    r"Normalized source radius (i.e., \rho*)": "rho",
    "Apparent magnitude of source star in u-band": "u",
    "Apparent magnitude of source star in g-band": "g",
    "Apparent magnitude of source star in r-band": "r",
    "Apparent magnitude of source star in i-band": "i",
    "Apparent magnitude of source star in z-band": "z",
    "Apparent magnitude of source star in y-band": "Y",
    "Blending factor in u-band": "blend_u",
    "Blending factor in g-band": "blend_g",
    "Blending factor in r-band": "blend_r",
    "Blending factor in i-band": "blend_i",
    "Blending factor in z-band": "blend_z",
    "Blending factor in y-band": "blend_y",
    "Number of data points": "n_data_catalog",
    "Delta chi2": "delta_chi2_catalog",
    "FWHM (days)": "fwhm_catalog_days",
    "alpha": "alpha_catalog",
    "Alpha": "alpha_catalog",
    "ALPHA": "alpha_catalog",
    r"\alpha(degree)": "alpha_catalog",
    r"\alpha (degree)": "alpha_catalog",
    "alpha (rad)": "alpha_catalog",
    "alpha(rad)": "alpha_catalog",
    "alpha (degree)": "alpha_catalog",
    "alpha (degrees)": "alpha_catalog",
    "trajectory angle": "xi_catalog",
    "parallax angle": "xi_catalog",
    "xi": "xi_catalog",
    "Xi": "xi_catalog",
    "XI": "xi_catalog",
    r"\xi": "xi_catalog",
    r"\xi (degree)": "xi_catalog",
    r"\xi(degree)": "xi_catalog",
    r"\xi (rad)": "xi_catalog",
    r"\xi(rad)": "xi_catalog",
    "murel_1 (mas/days): x-component of lens-source relative angular velocity":
        "murel_1_mas_per_day",
    "murel_2 (mas/days): x-component of lens-source relative angular velocity":
        "murel_2_mas_per_day",
    "DetectionFlag_u: if source is visible in u-band =1 and zero otherwise":
        "DetectionFlag_u",
    "DetectionFlag_g: if source is visible in g-band =1 and zero otherwise":
        "DetectionFlag_g",
    "DetectionFlag_r: if source is visible in r-band =1 and zero otherwise":
        "DetectionFlag_r",
    "DetectionFlag_i: if source is visible in i-band =1 and zero otherwise":
        "DetectionFlag_i",
    "DetectionFlag_z: if source is visible in z-band =1 and zero otherwise":
        "DetectionFlag_z",
    "DetectionFlag_y: if source is visible in y-band =1 and zero otherwise":
        "DetectionFlag_y",
}

REQUIRED_COLUMNS = [
    "catalog_event_id",
    "l_deg",
    "b_deg",
    "lens_mass_msun",
    "lens_distance_kpc",
    "source_distance_kpc",
    "logTe",
    "tE_catalog_days",
    "t0_catalog_days",
    "u0",
    "thetaE_mas",
    "piE",
    "rho",
    "u",
    "g",
    "r",
    "i",
    "z",
    "Y",
    "blend_u",
    "blend_g",
    "blend_r",
    "blend_i",
    "blend_z",
    "blend_y",
    "xi_catalog",
]

BLEND_COLUMNS = [
    "blend_u",
    "blend_g",
    "blend_r",
    "blend_i",
    "blend_z",
    "blend_y",
]

SOURCE_MAG_COLUMNS = ["u", "g", "r", "i", "z", "Y"]

CATALOG_BANDS = ["u", "g", "r", "i", "z", "y"]

DETECTION_FLAG_COLUMNS = [
    f"DetectionFlag_{band}"
    for band in CATALOG_BANDS
]

if PARALLAX_ANGLE_BASIS == "galactic_lb":
    PARALLAX_ANGLE_BASIS = "galactic_n1n2"
