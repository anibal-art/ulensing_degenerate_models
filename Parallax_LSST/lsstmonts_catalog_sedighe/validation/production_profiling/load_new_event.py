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

    # An H0-generated event (simulated with truth_parallax=False, i.e.
    # parallax=["None",0.0] in the simulator) has no piEN/piEE in
    # pyLIMA_parameters at all. The keys are simply OMITTED below (not
    # set to None/0) -- no truth piE exists, and none is invented; a
    # dict lookup with .get("piEN") returns None either way, but an
    # omitted key also lets fit_lc.get_param's own required=False/
    # default=0.0 fallback apply cleanly wherever event_params is used
    # for bookkeeping, instead of crashing on float(None).
    truth = {
        "t0": float(pp["t0"]), "u0": float(pp["u0"]), "tE": float(pp["tE"]),
        "rho": float(pp["rho"]),
    }
    generating_model = "H1"
    if "piEN" in pp and "piEE" in pp:
        truth["piEN"] = float(pp["piEN"])
        truth["piEE"] = float(pp["piEE"])
    else:
        generating_model = "H0"

    return {
        "sample": "profiling_sample",
        "row": int(indices[0]),
        "curves": curves,
        "truth": truth,
        "true_params": dict(truth),
        "generating_model": generating_model,
        "Source": int(indices[0]),
        "rango": 1,
        "event_ra": float(tril["ra"]),
        "event_dec": float(tril["dec"]),
    }
