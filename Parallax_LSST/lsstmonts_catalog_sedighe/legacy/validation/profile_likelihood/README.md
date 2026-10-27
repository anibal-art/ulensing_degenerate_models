# Hidden-parallax profile-likelihood validation

This directory contains the profile-likelihood diagnostics used to investigate
the local and global validity of the H1 covariance matrix.

The principal case studies were catalog rows 63218 and 72168.

The diagnostics showed that:

- the local quadratic covariance approximation can be accurate near an H1
  minimum;
- nuisance-parameter bounds, particularly rho in the original production
  domain, can invalidate an unconstrained covariance extrapolation;
- the global likelihood can contain substantially different valleys even when
  the local Hessian is well behaved.

These diagnostics motivated the subsequent `validation/bounds_audit/`
experiment.

The scripts stored here are snapshots of the scripts used during that
validation. Large H5 and NPY artifacts are intentionally not committed to Git.
