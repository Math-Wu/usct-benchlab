# Native ray-Born reference operator

`operators.forward.ray_born.RayBornOperator` maps a **real squared-slowness
perturbation** to complex pressure with shape `(frequency, transmitter, receiver)`.
The outgoing 2-D Green function uses time dependence `exp(-i omega t)`:

```
G0(r) = i/4 H0^(1)(omega r / c0)
K dm = omega^2 pixel_area sum_x G(receiver,x) G(x,source) q_source dm(x)
```

A homogeneous background uses the analytic Hankel function. The supplied config
uses `green_backend: volume_integral` for heterogeneous backgrounds: solve the
free-space Helmholtz Lippmann-Schwinger equation for every source and receiver.
FFT performs zero-padded linear convolution, and GMRES must meet an explicit
residual tolerance. The logarithmic singular self-cell is integrated using an
equal-area disk approximation; other cells use midpoint quadrature. This method
includes diffraction and multiple scattering in the background, with a local
distorted-Born update. It is not geometrical ray shooting.

The optional `eikonal_wkb` backend modifies phase using fast marching and spreading
using a positive finite-volume intensity transport solve. Water calibration
reduces its reference-grid bias but does not fix caustics, missing diffraction
or finite-frequency interference. The model is fixed during each linearization; its
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

The frozen-operator tests cover complex
adjoint products in homogeneous/heterogeneous media, an independent point
scatterer formula, refracted phase and nonconstant spreading, bounded cache
invariance, and quadratic Born truncation error against an independently
assembled discrete Lippmann-Schwinger multiple-scattering solve.

These tests verify numerical identities and weak-scattering behavior, not
full-scale breast image quality or clinical performance. References and the
upstream project are listed in [physics_validation.md](physics_validation.md).

## Nonlinear mode

`RayBornForward` rebuilds Green fields at each current squared-slowness model.
`solvers/nonlinear.py` computes truncated regularized Born steps and accepts only
decrease of the recomputed nonlinear training objective. It does not use validation
pressure in gradients or line searches. Full Green backgrounds have the derivative
label `discrete_volume_integral_born_derivative`, tested against central finite
differences and the complex adjoint identity. The WKB derivative is labelled
`continuum_born_approximation_not_discrete_wkb_derivative` to distinguish exact
frozen-transpose tests from a derivative of the WKB numerical discretization.

The upstream workflow was inspected at commit
`86fbd8a4cad17b5fc69909d687017ba24018adda`, particularly
`reconstructGreensImage.m` and `calcSoundSpeedUpdateDirection.m`.
Unlike that upstream implementation, this reference does not yet reproduce
frequency continuation, paraxial ray-linking, caustic phase, or absorption.
The [validation record](validation/2026-09-08_physics.md) separates the failed
coarse-grid/WKB tests from the corrected high-resolution full Green checks.
Passing small Born tests alone is not breast-image acceptance. No upstream MATLAB
source is vendored.

## Background-error regression

The NBP D case uses independent k-Wave object/water data, 128 transmitters,
128 receivers, and a 256 x 256 inverse grid. In a post-hoc forward check with
the evaluation sound-speed label, the original WKB pressure relative residual
was 0.4341, versus 0.2680 for uniform water. Conservative intensity transport
alone increased the error to 2.0046: removing a curvature-discretization artifact
does not remove geometrical caustics. Full Green propagation reduced the residual
to 0.00403. Its Born derivative agrees with finite differences to 8.9e-8, whereas
the original WKB/Born pair had a 0.972 relative discrepancy.

These labels are used only after inversion-independent source calibration to
test the forward model. They are not an initialization or stopping signal.
The full Green implementation is more expensive; source/GMRES loops cooperate
with the run deadline and retain the last complete accepted checkpoint.
