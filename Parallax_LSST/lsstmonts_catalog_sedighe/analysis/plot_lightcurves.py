from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from detection_criteria import mag
# Si read_data está en otro módulo, importar como lo venías haciendo.
# from functions_roman_rubin import read_data

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------

run_dir = Path(
    "/home/anibal/ulensing_degenerate_models/Parallax_LSST/"
    "runs/LSSTMONTS_xi_catalogBlending_FSPLparallax_fitFSPLNoPiE_test5"
)

summary = pd.read_parquet(run_dir / "logs" / "run_summary.parquet")

ordered_bands = ["W149", "u", "g", "r", "i", "z", "y"]

colorbands = {
    "W149": "black",
    "u": "purple",
    "g": "green",
    "r": "red",
    "i": "orange",
    "z": "brown",
    "y": "gray",
}

# Zeropoints: usar los mismos que en tu pipeline
# Si ya los tenés definidos en tu notebook, borrá este bloque.
ZP = {
    "W149": 27.615,
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
}


def find_event_h5(model_dir):
    model_dir = Path(model_dir)

    files = sorted(model_dir.glob("Event_*.h5"))

    if len(files) == 0:
        files = sorted(model_dir.rglob("Event_*.h5"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No encontré Event_*.h5 dentro de {model_dir}"
        )

    if len(files) > 1:
        print("Warning: encontré más de un Event_*.h5. Uso el primero:")
        for f in files:
            print("  ", f)

    return files[0]


def band_detection_summary(bands, pyLIMA_parameters, sigma_threshold=3.0):
    """
    Diagnóstico simple:
    cuenta puntos con brightening significativo respecto al baseline.

    En magnitudes, el evento es más brillante si mag_obs < mag_baseline.
    Entonces:
        significance = (mag_baseline - mag_obs) / err_mag
    """

    rows = []

    for b in ordered_bands:
        if b not in bands or len(bands[b]) == 0:
            continue

        if f"ftotal_{b}" not in pyLIMA_parameters:
            baseline_mag = np.nan
        else:
            baseline_mag = mag(
                ZP[b],
                pyLIMA_parameters[f"ftotal_{b}"],
            )

        time = np.asarray(bands[b]["time"], dtype=float)
        m = np.asarray(bands[b]["mag"], dtype=float)
        merr = np.asarray(bands[b]["err_mag"], dtype=float)

        valid = np.isfinite(time) & np.isfinite(m) & np.isfinite(merr) & (merr > 0)

        if not np.isfinite(baseline_mag):
            n_above = 0
            max_sig = np.nan
        else:
            sig = (baseline_mag - m[valid]) / merr[valid]
            n_above = int(np.sum(sig > sigma_threshold))
            max_sig = float(np.nanmax(sig)) if len(sig) > 0 else np.nan

        rows.append(
            {
                "band": b,
                "n_points": int(np.sum(valid)),
                f"n_>{sigma_threshold:.0f}sigma": n_above,
                "max_sigma": max_sig,
                "baseline_mag": baseline_mag,
            }
        )

    return pd.DataFrame(rows)


def plot_sedighe_event(
    row_index=0,
    zoom_factor_tE=None,
    sigma_threshold=3.0,
):
    """
    row_index es el índice dentro de run_summary.parquet, no catalog_event_id.
    """

    row = summary.iloc[row_index]

    model_dir = Path(row["model_dir"])
    event_h5 = find_event_h5(model_dir)

    print("status:", row.get("status", ""))
    print("catalog_event_id:", row.get("catalog_event_id", ""))
    print("model_dir:", model_dir)
    print("event_h5:", event_h5)

    indices, strings, pyLIMA_parameters, TRILEGAL_params, bands, GENULENS_row, TRILEGAL_row = read_data(
        str(event_h5)
    )

    t0 = float(pyLIMA_parameters["t0"])
    tE = float(pyLIMA_parameters["tE"])

    det = band_detection_summary(
        bands,
        pyLIMA_parameters,
        sigma_threshold=sigma_threshold,
    )

    print()
    print(det.to_string(index=False))

    n_detected_bands = int(np.sum(det[f"n_>{sigma_threshold:.0f}sigma"] > 0))

    print()
    print(f"Bandas con al menos 1 punto > {sigma_threshold:.0f} sigma:", n_detected_bands)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 5))

    for b in ordered_bands:
        if b in bands and len(bands[b]) != 0:

            ax.errorbar(
                bands[b]["time"],
                bands[b]["mag"],
                bands[b]["err_mag"],
                linestyle="",
                marker="o",
                markersize=3,
                alpha=0.8,
                color=colorbands.get(b, None),
                label=b,
            )

            if f"ftotal_{b}" in pyLIMA_parameters:
                baseline_mag = mag(
                    ZP[b],
                    pyLIMA_parameters[f"ftotal_{b}"],
                )

                ax.axhline(
                    baseline_mag,
                    color=colorbands.get(b, None),
                    linestyle="--",
                    alpha=0.5,
                )

    ax.axvline(
        t0,
        color="k",
        linestyle="-",
        alpha=0.8,
        label="t0",
    )

    if zoom_factor_tE is not None:
        ax.set_xlim(
            t0 - zoom_factor_tE * tE,
            t0 + zoom_factor_tE * tE,
        )

    ax.invert_yaxis()
    ax.set_xlabel("time [JD]")
    ax.set_ylabel("magnitude")
    ax.set_title(
        f"catalog_event_id={row.get('catalog_event_id', '')} | "
        f"status={row.get('status', '')}"
    )
    ax.legend(loc=(1.02, 0.0))
    fig.tight_layout()
    plt.show()

    return det, row, pyLIMA_parameters, bands
