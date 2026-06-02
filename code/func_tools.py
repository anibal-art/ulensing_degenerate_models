import numpy as np
import pandas as pd
import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt

from pyLIMA import event, telescopes
from pyLIMA.models import PSPL_model
from pyLIMA.simulations import simulator
from pyLIMA.fits import TRF_fit

import numpy as np
import scipy.optimize as so
import numpy as np
import scipy.optimize as so
from pyLIMA.models import PSPL_model
def orbital_period_kepler(a_au, M_tot_Msun):
    """
    Compute the orbital period of a binary system using Kepler's third law
    in astronomical units.

    Parameters
    ----------
    a_au : float or array-like
        Semimajor axis in astronomical units (AU).
    M_tot_Msun : float or array-like
        Total mass of the system in solar masses (M_sun).

    Returns
    -------
    P_yr : float or ndarray
        Orbital period in years.
    """
    a_au = np.asarray(a_au, dtype=float)
    M_tot_Msun = np.asarray(M_tot_Msun, dtype=float)
    print("Period ", np.sqrt(a_au**3 / M_tot_Msun), "years")
    print("converting to ", np.sqrt(a_au**3 / M_tot_Msun)*365.25, "days (to use in pyLIMA)")
    return np.sqrt(a_au**3 / M_tot_Msun)*365.25*(1/u.day)

def build_case(case_name, DS, DL, rEhat, v_perp, a, M1, M2,
               t0=50, u0=0.1, xi_phase=0, xi_inclination=np.pi/2, flux_ratio=0.0):
    """
    Construye un diccionario con los parámetros de un caso de xallarap.
    """
    q_xi = (M1 / M2).decompose().value
    P = orbital_period_kepler(a, M1 + M2)

    # tE = (rEhat * DL / DS) / v_perp
    tE = (rEhat) / v_perp
    return {
        "case": case_name,
        "DS_kpc": DS.to(u.kpc).value,
        "DL_kpc": DL.to(u.kpc).value,
        "rEhat_AU": rEhat.to(u.AU).value,
        "v_perp_kms": v_perp.to(u.km/u.s).value,
        "a_AU": a.to(u.AU).value,
        "M1_Msun": M1.to(u.M_sun).value,
        "M2_Msun": M2.to(u.M_sun).value,
        "xi_mass_ratio": q_xi,
        "tE": tE.to(u.day).value,
        "t0": t0,
        "u0": u0,
        "xiE": (a / rEhat).decompose().value,
        "omega_xi_1_per_day": (2*np.pi / P).value,
        "xi_phase": xi_phase,
        "xi_inclination": xi_inclination,
        "flux_ratio": flux_ratio,
        "P": P.value,
    }

DS = 8 * u.kpc
DL = 4 * u.kpc
v_perp = 50 * u.km / u.s
a = 2 * u.AU

rows = []

# =========================
# rEhat = 5 AU
# =========================

rEhat = 5 * u.AU

# Case 1: face-on, P > tE
rows.append(build_case(
    "case1", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun
))

# Case 2: face-on, P < tE
rows.append(build_case(
    "case2", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun
))

# Case 3a: edge-on, low mass ratio
rows.append(build_case(
    "case3a", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun,
    xi_inclination=np.pi/2
))

# Case 3b: edge-on, high mass ratio
rows.append(build_case(
    "case3b", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun,
    xi_inclination=np.pi/2
))

# =========================
# rEhat = 2 AU  (Case 4)
# =========================
rEhat = 2 * u.AU

rows.append(build_case(
    "case4-1", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun,
    xi_inclination=0
))

rows.append(build_case(
    "case4-2", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun,
    xi_inclination=0
))

rows.append(build_case(
    "case4-3a", DS, DL, rEhat, v_perp, a,
    M1=2*u.M_sun, M2=1.4*u.M_sun,
    xi_inclination=np.pi/2
))

rows.append(build_case(
    "case4-3b", DS, DL, rEhat, v_perp, a,
    M1=1.4*u.M_sun, M2=100*u.M_sun,
    xi_inclination=np.pi/2
))

df_cases = pd.DataFrame(rows).set_index("case")




def chi2_theoretical(fit_params, your_model, use_magnification=False,
                     fs_fixed=1.0, ftotal_fixed=1.0):
    """
    SSE (sin pesos): compara data vs modelo.
    fit_params = [t0,u0,tE]
    Fija flujos: fsource=fs_fixed, ftotal=ftotal_fixed (porque blend_flux_parameter='ftotal' default).
    """
    fit_params = np.asarray(fit_params, dtype=float)
    full_params = np.concatenate([fit_params, [fs_fixed, ftotal_fixed]])

    py_params = your_model.compute_pyLIMA_parameters(full_params)

    sse = 0.0
    for telescope in your_model.event.telescopes:
        if telescope.lightcurve is None:
            continue

        data = telescope.lightcurve['flux'].value

        if use_magnification:
            model_pred = your_model.model_magnification(telescope, py_params)
        else:
            model_pred = your_model.compute_the_microlensing_model(
                telescope, py_params
            )['photometry']

        resid = data - model_pred
        sse += np.sum(resid**2)

    return float(sse)



def build_sim_event(time, mag0=19.0, emag=0.01, filt="G"):
    """
    Crea un Event con un Telescope con columnas time/mag/err_mag.
    simulator.simulate_lightcurve(...) llenará flux/err_flux (si lo usás).
    """
    ev = event.Event()
    ev.name = "Simulated"
    ev.ra = 170
    ev.dec = -70

    lightcurve_sim = np.c_[time, np.full_like(time, mag0), np.full_like(time, emag)]
    tel = telescopes.Telescope(
        name="Simulation",
        camera_filter=filt,
        lightcurve=lightcurve_sim.astype(float),
        lightcurve_names=["time", "mag", "err_mag"],
        lightcurve_units=["JD", "mag", "mag"],
        location="Earth",
    )
    ev.telescopes.append(tel)
    return ev




def a_from_P_kepler_days(P_days, Mtot_Msun):
    """
    Kepler: a^3 = Mtot * P^2, con P en años, a en AU, Mtot en Msun.
    Devuelve a en AU (float).
    """
    P_yr = np.asarray(P_days, dtype=float) / 365.25
    return (Mtot_Msun * P_yr**2)**(1.0/3.0)


import numpy as np
import pandas as pd


def detect_residual_structure_envelope(
    t,
    A_truth,
    A_fit,
    t0_ref,
    smooth_window=31,
    fraction_of_peak=0.05,
    min_points=5,
    pad_fraction=0.10
):
    """
    Detecta la región con estructura residual usando la envolvente suavizada
    de |A_truth - A_fit|.

    Parameters
    ----------
    t : array-like
        Tiempos.
    A_truth : array-like
        Magnificación verdadera.
    A_fit : array-like
        Magnificación del ajuste.
    t0_ref : float
        Tiempo de referencia del pico.
    smooth_window : int, optional
        Tamaño de la ventana para suavizar el residual.
    fraction_of_peak : float, optional
        Fracción del máximo de la envolvente usada como umbral.
    min_points : int, optional
        Número mínimo de puntos contiguos para aceptar un segmento.
    pad_fraction : float, optional
        Fracción del ancho del intervalo detectado para expandirlo.

    Returns
    -------
    result : dict
        Diccionario con:
        - residual
        - envelope
        - threshold
        - mask
        - intervals
        - best_interval
        - df_interval_truth
        - df_interval_fit
        - t0_interval
        - tE_interval
        - t_left
        - t_right
    """

    t = np.asarray(t, dtype=float)
    A_truth = np.asarray(A_truth, dtype=float)
    A_fit = np.asarray(A_fit, dtype=float)

    if not (len(t) == len(A_truth) == len(A_fit)):
        raise ValueError("t, A_truth y A_fit deben tener la misma longitud.")

    # ============================================================
    # [NUEVO BLOQUE 1] residual absoluto
    # ============================================================
    residual = np.abs(A_truth - A_fit)

    # ============================================================
    # [NUEVO BLOQUE 2] envolvente suavizada
    # ============================================================
    smooth_window = int(smooth_window)

    if smooth_window < 1:
        smooth_window = 1

    kernel = np.ones(smooth_window) / smooth_window
    envelope = np.convolve(residual, kernel, mode="same")

    # ============================================================
    # [NUEVO BLOQUE 3] umbral relativo al máximo
    # ============================================================
    threshold = fraction_of_peak * envelope.max()

    mask = envelope > threshold

    # ============================================================
    # [NUEVO BLOQUE 4] detectar intervalos contiguos
    # ============================================================
    intervals = []
    in_segment = False
    start_idx = None

    for i, flag in enumerate(mask):
        if flag and not in_segment:
            start_idx = i
            in_segment = True

        elif not flag and in_segment:
            end_idx = i - 1

            if end_idx - start_idx + 1 >= min_points:
                intervals.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "t_start": t[start_idx],
                    "t_end": t[end_idx],
                    "n_points": end_idx - start_idx + 1
                })

            in_segment = False

    if in_segment:
        end_idx = len(mask) - 1

        if end_idx - start_idx + 1 >= min_points:
            intervals.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "t_start": t[start_idx],
                "t_end": t[end_idx],
                "n_points": end_idx - start_idx + 1
            })

    # ============================================================
    # [NUEVO BLOQUE 5] elegir intervalo más cercano a t0_ref
    # ============================================================
    best_interval = None

    if len(intervals) > 0:
        centers = np.array([
            0.5 * (inter["t_start"] + inter["t_end"])
            for inter in intervals
        ])

        idx_best = np.argmin(np.abs(centers - t0_ref))
        best_interval = intervals[idx_best]

    # ============================================================
    # [NUEVO BLOQUE 6] si no hay intervalo
    # ============================================================
    if best_interval is None:
        return {
            "residual": residual,
            "envelope": envelope,
            "threshold": threshold,
            "mask": mask,
            "intervals": intervals,
            "best_interval": None,
            "df_interval_truth": None,
            "df_interval_fit": None,
            "t0_interval": None,
            "tE_interval": None,
            "t_left": None,
            "t_right": None
        }

    # ============================================================
    # [NUEVO BLOQUE 7] expandir intervalo
    # ============================================================
    t_start = best_interval["t_start"]
    t_end = best_interval["t_end"]

    width = t_end - t_start

    if width <= 0:
        width = np.median(np.diff(np.sort(t))) * max(min_points, 3)

    dt_pad = pad_fraction * width

    t_left = t_start - dt_pad
    t_right = t_end + dt_pad

    interval_mask = (t >= t_left) & (t <= t_right)

    df_interval_truth = pd.DataFrame({
        "t": t[interval_mask],
        "A": A_truth[interval_mask]
    })

    df_interval_fit = pd.DataFrame({
        "t": t[interval_mask],
        "A": A_fit[interval_mask]
    })

    # ============================================================
    # [NUEVO BLOQUE 8] t0 y tE efectivos
    # ============================================================
    t0_interval = 0.5 * (t_left + t_right)
    tE_interval = (t_right - t_left) / 6.0

    return {
        "residual": residual,
        "envelope": envelope,
        "threshold": threshold,
        "mask": mask,
        "intervals": intervals,
        "best_interval": best_interval,
        "df_interval_truth": df_interval_truth,
        "df_interval_fit": df_interval_fit,
        "t0_interval": t0_interval,
        "tE_interval": tE_interval,
        "t_left": t_left,
        "t_right": t_right
    }