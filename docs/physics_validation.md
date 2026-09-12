# Physics operators and validation

See the [independent calibration and model-matching report](validation/2026-09-09_calibration_and_model_matching_CN.md)
for measured timing accuracy, spatial-scale controls, second-order refinement,
finite-frequency observation matching, and frozen reconstruction comparisons.
The [multiband travel-time work log](validation/2026-09-09_multiband_traveltime_CN.md)
documents the nonlinear observation/Jacobian, optional CUDA backend, and ongoing
sampling/optimization checks. Forward agreement is not reconstruction acceptance.

## Scope and conventions

Coordinates are `[y, x]` in meters. `GridSpec.origin_m` is the lower cell edge;
unknowns live at cell centers. A forward model predicts data, its Jacobian maps
model perturbations to data perturbations, and the adjoint is the transpose of
that Jacobian under the documented discrete inner product. An adjoint is not an
inverse reconstruction algorithm.

## Native bent-ray core

`operators.forward.eikonal.EikonalForward` solves the first-arrival Eikonal
problem `|grad T| = s`, where `s = 1/c` is absolute slowness in seconds/meter.
It defaults to causal first-order Godunov fast marching. Optional `spatial_order=2`
(`eikonal_order: 2` in Bent configuration) uses mixed second-order upwinding with
its own four-parent tangent/adjoint tape; source/interface singularities still
limit global accuracy. Both support anisotropic Cartesian
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
include grid discretization error, first-arrival-only propagation, no multiple
arrivals and no diffraction. Install `.[performance]` for optional compiled
marching/tangent/adjoint loops; first-order regression tests compare the same
discretization with the independent pure-Python reference, without fastmath or
reduced precision. Second-order verification uses analytic cases, refinement,
and directional-derivative/adjoint checks.

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

The [recorded-medium ToF adaptation audit (Chinese)](validation/2026-09-09_tof_adaptation_CN.md)
compares original surrogate data with independently simulated k-Wave pressure,
checks image/element identity against the actual simulation input, and separates
finite-band picking from Eikonal discretization. It does not certify that these
observables are interchangeable or that reconstruction quality has reached a limit.

## Ray-Born and full-wave continuation

Native Born inversion relinearizes the background and line-searches on recomputed
pressure. The supplied config uses full Green volume-integral backgrounds.
The optional Eikonal/WKB model is not the full upstream ray-shooting/caustic
implementation: five of eight earlier coarse-grid inversions failed to find a
descent step. The high-resolution background-error check and reconstruction
results are separated in the validation record, not combined into one ranking.

Production `fwi_wust` calls the maintained WUST runtime directly. The previous
matrix-export reference bridge is removed. See [FWI deployment](fwi.md) for
the separate hardware-free, CPU-reference and real CUDA acceptance gates.

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
  (`fwi_wust` uses the maintained runtime); `fwi_tiny` is only a plumbing/sanity test.

No upstream MATLAB source has been copied into this MIT-licensed package.
