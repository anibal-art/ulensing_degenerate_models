"""
Fisher-information engine consistent with the frozen fitter.

DOCUMENTED TRANSFORM (fixed before looking at any result):

H0 dimensionless local coordinates at a solution (t0*,u0*,tE*,rho*):
    y1 = (t0 - t0*) / tE*      (=0 at the solution itself)
    y2 = u0
    y3 = log(tE)
    y4 = log(rho)

Local linear map x(y) around the solution (first order, adequate for
a Gauss-Newton/local-quadratic Fisher analysis):
    dt0  = tE* * dy1
    du0  = dy2
    dtE  = tE* * dy3   (from d(log tE) = dtE/tE, evaluated at tE*)
    drho = rho* * dy4

So D = diag(tE*, 1, tE*, rho*) satisfies dx = D dy, and
    F_y = D^T F_x D
where F_x is the Gauss-Newton Fisher matrix in raw physical units,
F_x = J_x^T J_x, J_x = d(profiled residuals)/d(x) (numerical, central
differences, fluxes profiled out via the exact bounded_flux_profile
objective -- i.e. J_x = Jacobian of fit.objective_function()).

H1 adds y5=piEN, y6=piEE with dx5/dy5=1, dx6/dy6=1 (already
dimensionless), so D_H1 = diag(tE*, 1, tE*, rho*, 1, 1).

Covariance in dimensionless coordinates: C_y = pinv(F_y) (SVD-based
pseudoinverse, singular values below a numerical-rank tolerance
zeroed out and flagged rather than hidden).
"""
import numpy as np


def numerical_jacobian(objective_fn, x0, rel_step, scale):
    """
    Central-difference Jacobian of objective_fn at x0.

    rel_step: relative step size (e.g. 1e-5).
    scale: per-parameter scale used for the step (e.g. tE for t0/tE,
        1 for u0, rho for rho, 1 for piE) -- NOT the dimensionless
        transform, just a sane finite-difference step.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    r0 = np.asarray(objective_fn(x0), dtype=float)
    m = len(r0)
    J = np.zeros((m, n))
    for k in range(n):
        h = rel_step * max(abs(scale[k]), 1e-12)
        xp = x0.copy()
        xp[k] += h
        xm = x0.copy()
        xm[k] -= h
        rp = np.asarray(objective_fn(xp), dtype=float)
        rm = np.asarray(objective_fn(xm), dtype=float)
        J[:, k] = (rp - rm) / (2.0 * h)
    return J, r0


def fisher_from_jacobian_physical(J_phys, scales_dimless):
    """
    scales_dimless: the diagonal of D = dx/dy (e.g. [tE,1,tE,rho] for
    H0, [tE,1,tE,rho,1,1] for H1).
    Returns F_y (dimensionless Fisher), F_phys.
    """
    F_phys = J_phys.T @ J_phys
    D = np.diag(scales_dimless)
    F_y = D.T @ F_phys @ D
    return F_y, F_phys


NUMERICAL_RANK_TOL = 1e-10


def fisher_diagnostics(F_y, weak_labels=None):
    """
    Full diagnostic bundle for a dimensionless Fisher matrix.
    Robust via SVD (F_y is symmetric PSD in theory; SVD handles any
    numerical asymmetry/negative-eigenvalue noise gracefully and lets
    us flag it rather than hide it).
    """
    n = F_y.shape[0]
    F_sym = 0.5 * (F_y + F_y.T)
    asym_norm = float(np.linalg.norm(F_y - F_sym))

    eigvals, eigvecs = np.linalg.eigh(F_sym)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    has_negative = bool(np.any(eigvals < -1e-8 * max(abs(eigvals.max()), 1.0)))

    scale = max(abs(eigvals.max()), 1e-300)
    numerical_rank = int(np.sum(eigvals > NUMERICAL_RANK_TOL * scale))
    full_rank = numerical_rank == n

    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    min_eig = float(eigvals_clipped[0])
    max_eig = float(eigvals_clipped[-1])
    cond_number = float(max_eig / min_eig) if min_eig > 0 else np.inf

    # Pseudoinverse (covariance), SVD-based, tolerance-flagged.
    pinv_tol = NUMERICAL_RANK_TOL * scale
    inv_eigs = np.array([1.0 / e if e > pinv_tol else 0.0 for e in eigvals_clipped])
    C_y = (eigvecs * inv_eigs) @ eigvecs.T

    sigmas = np.sqrt(np.clip(np.diag(C_y), 0.0, None))

    # correlations
    with np.errstate(divide="ignore", invalid="ignore"):
        outer_sig = np.outer(sigmas, sigmas)
        corr = np.where(outer_sig > 0, C_y / outer_sig, 0.0)
    np.fill_diagonal(corr, 1.0)
    off_diag_mask = ~np.eye(n, dtype=bool)
    max_abs_corr = float(np.max(np.abs(corr[off_diag_mask]))) if n > 1 else 0.0

    # logdet (only if full rank / all eigs > tol)
    if full_rank:
        logdet = float(np.sum(np.log(eigvals_clipped)))
    else:
        logdet = np.nan

    weakest_vec = eigvecs[:, 0]

    return {
        "F_y": F_sym,
        "C_y": C_y,
        "eigvals": eigvals_clipped,
        "eigvecs": eigvecs,
        "min_eig": min_eig,
        "max_eig": max_eig,
        "cond_number": cond_number,
        "numerical_rank": numerical_rank,
        "full_rank": full_rank,
        "has_negative_eig": has_negative,
        "asym_norm": asym_norm,
        "logdet": logdet,
        "sigmas": sigmas,
        "corr": corr,
        "max_abs_corr": max_abs_corr,
        "weakest_eigvec": weakest_vec,
    }
