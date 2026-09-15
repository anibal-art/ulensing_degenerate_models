"""Load one profiling-sample H5 (written by standalone_materialize.py)
into a meta dict compatible with run_bounds_audit_refit_core's
run_one_fit()/fit_rubin_roman() calling convention."""
import numpy as np


def load_new_case(h5_path, core):
    curves = core.load_h5_lightcurves(h5_path)

    import h5py
    with h5py.File(h5_path, "r") as f:
        pp = dict(f["pyLIMA_parameters"].attrs)
        tril = dict(f["TRILEGAL_row"].attrs)
        indices = np.asarray(f["indices"])

    truth = {
        "t0": float(pp["t0"]), "u0": float(pp["u0"]), "tE": float(pp["tE"]),
        "rho": float(pp["rho"]), "piEN": float(pp["piEN"]), "piEE": float(pp["piEE"]),
    }

    return {
        "sample": "profiling_sample",
        "row": int(indices[0]),
        "curves": curves,
        "truth": truth,
        "true_params": dict(truth),
        "Source": int(indices[0]),
        "rango": 1,
        "event_ra": float(tril["ra"]),
        "event_dec": float(tril["dec"]),
    }
