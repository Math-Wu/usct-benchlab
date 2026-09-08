# Algorithms

Algorithm names remain stable, but `bent_ray_gn` and `rwave_adapter` on the
physics-validation branch now use numerical physics operators, not the old
straight-ray substitutes. Historical README example images predate this change.

| Algorithm / registered name | Model and input | Output | Scope / limitation | Config |
| --- | --- | --- | --- | --- |
| CGLS / `straight_cgls` | Weighted regularized straight-ray delays | Sound speed | Krylov baseline; no refraction or diffraction | `configs/algorithms/cgls.yaml` |
| SIRT / `straight_sirt` | Simultaneous straight-ray updates | Sound speed | Same linear model, different iteration and smoothing | `configs/algorithms/sirt.yaml` |
| SART / `straight_sart` | Transmitter-group straight-ray updates | Sound speed | A sweep is not equivalent to a CGLS iteration | `configs/algorithms/sart.yaml` |
| Bent-ray / `bent_ray_gn` | Nonlinear fast-marching Eikonal ToF | Sound speed | First arrivals with refraction; no diffraction, caustics or multiple arrivals | `configs/algorithms/bent_ray.yaml` |
| Born / `rwave_adapter` | Complex pressure, current-background Green fields and Born updates | Sound speed | Full Green volume integral by config; optional Eikonal/WKB; not an upstream r-Wave port | `configs/algorithms/rwave.yaml` |
| FWI / `fwi_kwave_adapter` | External k-Wave/WaveformInversionUST driver or result | Sound speed | Production MATLAB path is retained; imported artifacts cannot provide online stopping | `configs/algorithms/fwi_kwave.yaml` |
| Controlled FWI / `fwi_kwave_adapter` with `controlled_operator: true` | External Helmholtz matrices and complex pressure | Sound speed | Optional CPU reference Gauss-Newton loop, not the production GPU optimization trajectory | `configs/algorithms/fwi_controlled.yaml` |
| Attenuation / `attenuation_sirt` | Straight-ray log-amplitude ratios | Attenuation | Requires calibrated units; ignores diffraction and refraction | `configs/algorithms/attenuation.yaml` |
| Tiny FWI / `fwi_tiny` | Small synthetic sanity model | Sound speed | Test helper, not production FWI | `configs/algorithms/fwi_tiny.yaml` |

The pre-existing `diffusion_fwi_kwave_adapter` artifact adapter is unchanged and
is outside this physics validation. No learned model is trained or integrated.

## Native nonlinear methods

Bent-ray solves the water-grid-bias-corrected Eikonal problem at the current
slowness on each outer iteration. Its Jacobian and adjoint differentiate the
accepted upwind stencil. This is refraction tomography, not a scattering model.

Ray-Born defaults to `mode: nonlinear`. It updates squared slowness, recomputes
background Green fields, and accepts a step only if the recomputed training
pressure objective decreases. `mode: fixed_background` retains an explicitly
linear single-scattering reference. Both use complex observations and an explicit
source spectrum or independent water calibration; travel-time-only cases are
rejected. The supplied config uses `green_backend: volume_integral`: a
free-space Helmholtz Lippmann-Schwinger solve with FFT linear convolution and
GMRES. Its full-field Born Jacobian is the derivative of that discrete model
up to the configured linear-solve tolerance. This is distorted Born, not a
geometrical ray method or a claim to reproduce the upstream r-Wave optimizer.
The explicit `eikonal_wkb` option has an exact frozen Born adjoint but only an
approximate derivative of its WKB pressure predictor. A line-search failure is
recorded as failure, never convergence.

## Parameters and evaluation

`scripts/validate_physics.py extract` accepts `--tof-method xcorr` (default) or
`envelope`. The latter uses a Hilbert-envelope early crossing relative to the
independent water trace, with `--envelope-fraction 0.1` and a minimum peak/noise
ratio of 5 in the Python API. The source duration and physical speed bounds
define its search window. Confidence describes signal quality, not proof of
first-arrival accuracy; finite bandwidth, pre-ringing and multipath can bias it.
Use a fresh `--case-file` to retain the original pressure-derived case.

The full Green implementation requires SciPy >= 1.12, matching the documented
[`gmres` rtol API](https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.sparse.linalg.gmres.html).

Resolved configs are saved with each result. Common stopping/evaluation settings
are documented in [agent_evaluation.md](agent_evaluation.md); all active rules
are OR conditions, and validation observations cannot enter an update.

| Setting | Type / legal range | Meaning |
| --- | --- | --- |
| `regularization_lambda` | finite float, >= 0 | Square root of the normal-equation penalty coefficient; not interchangeable across parameterizations |
| `inner_iterations` | integer, > 0 | Truncated linear solve cap per nonlinear outer step |
| `step_length` | finite float, > 0 | Initial step scale before backtracking |
| `smooth_sigma` | finite float, >= 0 | Update smoothing width in reconstruction pixels |
| `sound_speed_bounds_mps` | two finite positive numbers, increasing | Feasible sound-speed interval |
| `roi_update_only` | boolean | Keep the complement of the supplied ROI at the initial model |
| Ray-Born `mode` | `nonlinear` or `fixed_background` | Whether background propagation is updated |
| `green_backend` | `volume_integral` or `eikonal_wkb` | Full-frequency background or geometrical approximation; shipped config selects the former |
| `green_solver_rtol` | finite float, 0 < value < 1 | Relative GMRES tolerance, default 1e-7 |
| `green_solver_maxiter` | integer, > 0 | Maximum GMRES restart cycles per source; restart length is min(40, pixel count), default 20 cycles |
| `max_cache_bytes` | integer, >= 0 | Upper bound on cached complex Green fields; excludes working arrays, default config 512 MiB |
| `regularization_scaling` | `absolute` or `relative_jacobian_diagonal` | In relative mode, multiply lambda squared by the median positive training J*WJ diagonal |
| Born `regularization_length_wavelengths` | finite float, >= 0, default 0 | Optional physical Laplacian scale: fraction times c0/max(training frequencies). Zero preserves pixel-scale regularization; cannot be used with identity penalty |
| Born `initialization` | `configured` (default) or `phase_cgls` | The optional warm start uses independent water and at least three training frequencies; unavailable in fixed-background mode |
| Born `initialization_iterations` | integer, > 0, default 80 | Training-only CGLS initialization cap, charged to the shared work/time budget |
| Born `initialization_lambda` | finite float, >= 0, default 0.02 | Slowness Laplacian penalty for initialization, distinct from the pressure penalty |
| Born `initialization_smooth_mm` | finite float, >= 0, default 3 | Initialization smoothing in physical millimeters, converted separately along each grid axis |
| Born `initialization_max_phase_rms` | finite float, > 0, default 0.2 | Maximum phase-slope fit residual in radians |
| Born `initialization_min_amplitude_ratio` | finite float, > 0, default 0.05 | Reject a training frequency whose object amplitude is below this fraction of water amplitude |
| Bent `initialization` | `configured` (default) or `cgls` | Optional training-only straight-ray warm start; later updates always use the Eikonal model |
| Bent `initialization_iterations` | integer, > 0, default 80 | Warm-start linear solve cap within the shared budget |
| `allow_underresolved` | boolean, default false | Explicit override of inverse-grid minimum 4 pixels/wavelength; not suitable for quality claims |
| `use_feature_weights` | boolean, default false | Opt into broadband ToF weights; disallowed with pressure-frequency holdout |
| Ray-Born `max_update_mps` | finite float, > 0 | Per-accepted-step speed-change bound |
| Ray-Born `max_backtracks` | integer, > 0 | Maximum trial steps per outer iteration |
| Ray-Born `assume_unit_source` | boolean, default false | Explicit opt-in for dimensionless synthetic unit-source data only |
| `stopping.max_elapsed_s` | finite float, >= 0 | Wall-clock budget; native Green source/GMRES loops check cooperatively, external calls only between calls |

The straight-ray and Eikonal slowness penalties share units, but the squared-
slowness pressure penalty does not. Do not equate iteration counts or penalty
numbers across those models. A valid acquisition ROI must be supplied without
ground truth in truth-free deployment; this breast-map validation uses the
full image grid because no ROI was supplied. Setting `roi_update_only: true` alone
does not create a tissue support mask.

The optional phase seed estimates group delay, not an exact first arrival.
Aliasing and multipath can remain after its fit-quality checks. Its held-out
frequencies are not used for phase unwrapping, initialization or physical-length
selection. Independent water calibration is a separate acquisition, not specimen
validation data. No parameters are automatically selected with ground truth.
Bent-ray smoothing acts on the update once before backtracking, not repeatedly
on the accumulated image. Lower smoothing is not automatically better: the
measured validation record contains counterexamples with smaller residuals and
worse SSIM.

See [operator_contract.md](operator_contract.md) for numerical conventions and
[physics_validation.md](physics_validation.md) for tests, references and evidence.
