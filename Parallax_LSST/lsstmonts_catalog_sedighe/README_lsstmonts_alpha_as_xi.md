# LSSTMONTS runner: assume catalog alpha is xi

This version makes the provisional assumption explicit:

\[
\alpha_{\rm catalog} \equiv \xi.
\]

Here \(\xi\) is the angle of the relative source-lens trajectory used in
Equations 5-6 of Sajadian & Sahu (2023). It is not treated as the geometrical
\(\alpha\) from Appendix Equation A3.

The local tangent-plane convention is:

\[
\mathbf n_1 = +l,\qquad \mathbf n_2 = +b,
\]

so the catalog parallax amplitude is decomposed as

\[
\pi_{E,n_1}=\pi_E\cos\xi,
\qquad
\pi_{E,n_2}=\pi_E\sin\xi.
\]

The runner then rotates that vector into the ICRS North/East components
required by pyLIMA. The same \(\xi\) is passed to the simulated trajectory as
`theta_rad` and `traj_angle`.

The catalog column remains named `alpha` for input compatibility. Output files
record:

- `alpha_catalog`
- `alpha_interpretation = xi`
- `xi_catalog`
- `xi_rad`
- `xi_deg`
- `piEN`
- `piEE`

## Preparation test

```bash
python run_lsstmonts_catalog_sedighe_alpha_as_xi.py \
    --config lsstmonts_catalog_sedighe_alpha_as_xi.yaml \
    --prepare-only
```

The preparation summary should contain:

```text
Catalog angle column:   alpha
Angle interpretation:   alpha == xi
Xi tangent-plane basis: galactic_n1n2
```

## Run the 25-event test

```bash
python run_lsstmonts_catalog_sedighe_alpha_as_xi.py \
    --config lsstmonts_catalog_sedighe_alpha_as_xi.yaml
```

The existing zero-blending-factor handling is retained: filters with
`f_s == 0` are omitted and at least three positive catalog filters are
required.
