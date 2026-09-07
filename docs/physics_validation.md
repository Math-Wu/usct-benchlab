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
arrivals, no diffraction, and pure-Python marching cost on large grids.

### Tests completed at the first implementation checkpoint

- Homogeneous water with off-grid/exterior transducers and anisotropic spacing.
- Directional finite differences of the nonlinear discrete map.
- Inner-product test of its Jacobian and exact transpose.
- Flat-interface Snell/Fermat travel-time agreement and improvement on refinement.
- A slow inclusion produces a delay different from fixed straight integration.
- Rejection of zero, negative, NaN, and infinite slowness.

Commands: `pytest -q` (40 passed at this checkpoint) and
`ruff check src/usctbench/operators tests/operators` (passed).
These are numerical/software tests, not clinical validation.

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
