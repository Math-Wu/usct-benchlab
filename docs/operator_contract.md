# Forward and adjoint contract

## Arrays, variables and inner products

Images are `[row=y, col=x]`; coordinates, cell-edge origins and spacing are in
meters. The canonical pressure shapes are **`(time, tx, rx)`** and
**`(frequency, tx, rx)`**. External MATLAB/k-Wave axes must be declared and
converted, not inferred from square arrays. Pressure transforms use actual sample
times, including a nonzero start time:

$$
P(\omega)=\Delta t\sum_n p(t_n)e^{+i\omega t_n},
\qquad p(t)=\operatorname{Re}(P(\omega)e^{-i\omega t}).
$$

`forward(model)` predicts observations. `linearize(model)` returns the current
prediction, Jacobian, and a `derivative_kind` label. A Jacobian's `forward(dm)`
maps image perturbations to data perturbations. Its `adjoint(d)` is the transpose
under Euclidean array inner products, with real-model/complex-data pairing:

$$
\operatorname{Re}\langle J\delta m,z\rangle
=\langle\delta m,J^*z\rangle.
$$

Pixel area and propagation units are inside the operator. Statistical precision
weights and ROI restrictions are applied by solvers. An adjoint is not an inverse.

## Organization

| Physics | Forward module | Adjoint module | Model |
| --- | --- | --- | --- |
| Straight rays | `operators/forward/straight_ray.py` | `operators/adjoint/straight_ray.py` | Slowness perturbation, s/m |
| Refraction / Eikonal | `operators/forward/eikonal.py` | `operators/adjoint/eikonal.py` | Absolute slowness, s/m |
| Finite-frequency Ray-Born | `operators/forward/ray_born.py` | `operators/adjoint/ray_born.py` | Squared slowness or its perturbation, s^2/m^2 |
| Full-wave Helmholtz bridge | `operators/forward/fwi.py` | `operators/adjoint/fwi.py` | Squared slowness, s^2/m^2 |

The historical `algorithms.ray.StraightRayProjector` import remains compatible;
the numerical implementation is in the operator layer. Bent-ray and Ray-Born
do not use this projector to make their predictions or report residuals.

## Exact and approximate derivatives

Straight rays have an exact linear transpose. Eikonal differentiation follows
the causal active stencil; finite differences are checked away from first-arrival
branch changes. Ray-Born has an exact frozen complex transpose, but its continuum
Born derivative is only approximate for the optional discrete Eikonal/WKB background.
The `volume_integral` backend instead differentiates the discrete full Green
model, verified by finite differences as well as a conjugate-adjoint test.
The metadata explicitly distinguishes these claims. Native nonlinear Ray-Born
recomputes propagation for trial models; descent is measured on that nonlinear
training objective. Failure to find a descent step remains an explicit failure.

The finite-frequency correlation-delay composition additionally requires a
stationary, interior, locally resolved peak before exposing its implicit
derivative. For correlation $C(\tau)$, a bound
$M_3=\sum_f a_f|Q_f|\omega_f^3$ implies
$|C'''|\leq M_3$. Requiring $-C''(\tau)>2\Delta\tau M_3$ certifies strict
concavity in the neighborhood excluded from the sampled competitor search.
Clipped or unfinished peak refinement is invalid. This is conservative local
quality control, not a proof of a unique global maximum; distant peak switches
remain nonsmooth and require nonlinear acceptance checks.

## Full-wave reference bridge

The optional bridge calls an installed WaveformInversionUST `HelmholtzSolver.m`
and `stencilOptParams.m` to export the actual optimized nine-point operator. It
requires the constructor's fixed-velocity-bounds argument supported by the
validated external installation; arbitrary upstream versions are not certified.
It freezes those stencil weights during differentiation and includes PML and
off-diagonal mass terms. MATLAB column-major indexing is explicitly permuted to
NumPy row-major indexing. The external `sign_conv=-1` convention is conjugated
at the public pressure boundary when necessary.

This is a CPU sparse-LU reference interface for finite-difference, adjoint and
online-control tests. It does not replace the existing production CUDA solver,
filtered gradients, or PR/FR update trajectory. `controlled_operator: true`
explicitly selects its reference Gauss-Newton loop. Production artifact import
continues to work but rejects online stopping/holdout requests it cannot enforce.
It records an unavailable external stop reason instead of inventing one.

## Sources and independent references

Pressure cases can store `source_spectrum`, `water_reference_time`, and complex
`water_reference`. Ray-Born uses the analytic Green function's source convention;
the Helmholtz bridge uses a discrete right-hand-side convention. These source
numbers are not interchangeable. Each model fits its own calibration from an
independent water acquisition, or requires a supplied calibrated spectrum.
Training-only source fitting is a separate API and cannot borrow held-out
specimen data. An independent water acquisition at a held-out specimen frequency
is allowed only when explicitly declared independent.

Tests separate (1) executable plumbing, (2) adjoint/derivative identities,
(3) independent k-Wave image validation. Passing one does not establish the next.
