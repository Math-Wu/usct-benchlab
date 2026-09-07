# Native ray-Born reference operator

`operators.forward.ray_born.RayBornOperator` maps a **real squared-slowness
perturbation** to complex pressure with shape `(frequency, transmitter, receiver)`.
The outgoing 2-D Green function uses time dependence `exp(-i omega t)`:

```
G0(r) = i/4 H0^(1)(omega r / c0)
K dm = omega^2 pixel_area sum_x G(receiver,x) G(x,source) q_source dm(x)
```

A homogeneous background uses the analytic Hankel function. For a heterogeneous
background, fast-marching travel-time differences modify phase; a causal WKB
transport solve modifies geometrical spreading. Water-reference subtraction
reduces grid bias in both. The model is fixed during each linearization; its
real-model adjoint uses complex conjugates and the real part of the accumulated
gradient. Green fields are cached under a configurable byte budget. The dense
data-by-pixel Jacobian is never assembled.

This is an actual finite-frequency single-scattering implementation, **not** a
travel-time least-squares method relabeled r-Wave. It is also **not** a complete
port of upstream r-Wave: it does not reproduce paraxial ray shooting, caustics,
absorption/dispersion, the Hessian-free update, or all nonlinear continuation
choices. WKB transport is a first-order reference discretization; instability
raises an error rather than silently clipping amplitudes. The point-source
singularity uses an explicit equivalent-cell-radius cutoff.

## Numerical checkpoint

`pytest -q`: 45 passed after adding this core. Five added tests cover complex
adjoint products in homogeneous/heterogeneous media, an independent point
scatterer formula, refracted phase and nonconstant spreading, bounded cache
invariance, and quadratic Born truncation error against an independently
assembled discrete Lippmann-Schwinger multiple-scattering solve.

These tests verify numerical identities and weak-scattering behavior, not
full-scale breast image quality or clinical performance. References and the
upstream project are listed in [physics_validation.md](physics_validation.md).
