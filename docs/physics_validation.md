# Physics operators and validation

## Scope and conventions

Coordinates are `[y, x]` in meters. `GridSpec.origin_m` is the lower cell edge;
unknowns live at cell centers. A forward model predicts data, its Jacobian maps
model perturbations to data perturbations, and the adjoint is the transpose of
that Jacobian under the documented discrete inner product. An adjoint is not an
inverse reconstruction algorithm.

## Native bent-ray core

`operators.forward.eikonal.EikonalForward` solves the first-arrival Eikonal
problem `|grad T| = s`, where `s = 1/c` is absolute slowness in seconds/meter.
It uses causal first-order Godunov fast marching, with anisotropic Cartesian
spacing, off-grid source seeding, bilinear receiver sampling, and known-water
padding for transducers outside the image domain. The calibrated forward map is
`T_h(s) - T_h(s_water) + distance/c_water`; this removes the reference-medium
grid bias, not the heterogeneous discretization error.

The derivative follows the accepted upwind stencil. For each accepted node,
`dT_i = a_i ds_i + sum_j w_ij dT_j`; reverse accumulation through exactly those
same dependencies is implemented in `operators.adjoint.eikonal`. There is no
straight-ray projector in this forward or adjoint. The derivative is local to
the active stencil; changes of the first-arrival branch can be nonsmooth.

This is a numerically implemented first-arrival model, **not** a verbatim port
of r-Wave's off-grid Heun shooting/ray-linking implementation. Its limitations
include first-order grid error, first-arrival-only propagation, no multiple
arrivals and no diffraction. Install `.[performance]` for optional compiled
marching/tangent/adjoint loops; regression tests compare the same discretization
with the pure-Python reference, without fastmath or reduced precision.

### Tests completed at the first implementation checkpoint

- Homogeneous water with off-grid/exterior transducers and anisotropic spacing.
- Directional finite differences of the nonlinear discrete map.
- Inner-product test of its Jacobian and exact transpose.
- Flat-interface Snell/Fermat travel-time agreement and improvement on refinement.
- A slow inclusion produces a delay different from fixed straight integration.
- Rejection of zero, negative, NaN, and infinite slowness.

The tests above remain regression gates. Current A100 results, independent
k-Wave pressure data, eight breast cases and image evidence are recorded in
[validation/2026-09-08_physics.md](validation/2026-09-08_physics.md).
These are numerical/software and phantom tests, not clinical validation.

## Ray-Born and full-wave continuation

Native Born inversion relinearizes the background and line-searches on recomputed
pressure. The supplied config uses full Green volume-integral backgrounds.
The optional Eikonal/WKB model is not the full upstream ray-shooting/caustic
implementation: five of eight earlier coarse-grid inversions failed to find a
descent step. The high-resolution background-error check and reconstruction
results are separated in the validation record, not combined into one ranking.

The optional full-wave bridge exports the actual external WaveformInversionUST
Helmholtz matrix, with a separate real-model complex adjoint. Finite differences
and one controlled truth-free update have been tested on A100/MATLAB. This does
not certify the production driver's complete optimization trajectory; that
driver and its gradient choices are preserved.

## Research basis

- Javaherian, Lucka, Cox (2020), *Refraction-corrected ray-based inversion for
  three-dimensional ultrasound tomography of the breast*, Inverse Problems 36,
  125010. DOI: 10.1088/1361-6420/abc0fc.
- Javaherian and Cox (2021), *Ray-based inversion accounting for scattering for
  biomedical ultrasound tomography*, Inverse Problems 37, 115003.
  DOI: 10.1088/1361-6420/ac28ed.
- Javaherian (2025), *Introduction and Numerical Validation of an Open-Source
  MATLAB Package for Quantitative Ultrasound Tomography via Ray-Born Inversion*,
  arXiv:2511.18511, especially sections 3.1-3.5.
- Upstream r-Wave: https://github.com/Ash1362/ray-based-quantitative-ultrasound-tomography
- Production FWI upstream: https://github.com/rehmanali1994/WaveformInversionUST
  (`fwi_kwave_adapter`); `fwi_tiny` is only a plumbing/sanity test.

No upstream MATLAB source has been copied into this MIT-licensed package.
