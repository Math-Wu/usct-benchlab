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
| FWI / `fwi_wust` | WUST frequency-domain total-pressure inversion | Sound speed | CUDA production; CPU reference; schedule/time budgets, not online convergence | `configs/algorithms/fwi_wust.yaml` |

The production FWI contract is documented in [fwi.md](fwi.md). WUST owns per-TX/per-frequency source-scale elimination; Born retains its distinct source calibration.

## Native nonlinear methods

Straight-ray delay cases must keep their recorded measurement reference speed;
changing only `reference_sound_speed_mps` is rejected. Rebase the observations
explicitly before changing that reference.

The current SIRT update includes row-length normalization, so its unprocessed
fixed-point objective uses precision `w_i / row_sum_i`, not CGLS's `w_i`.
SART with multiple subsets can cycle on inconsistent data. Its recorded global
objective is a monitor, not a guarantee of monotonic minimization. Reported data
residuals continue to use the common input precision. See the
[inverse-solver audit](validation/2026-09-09_inverse_solver_audit_CN.md) for independent references.

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
| Native GN `inner_solver` | `lsmr` (default), `lsqr`, `normal_cg` | Augmented least squares or explicit legacy normal-CG comparison; does not change the forward model |
| Native GN `inner_options` | mapping | `rtol`, `atol`, `btol`, `conlim` and optional column scaling; [definitions and numerical audit](validation/2026-09-10_augmented_least_squares_CN.md) |
| CGLS `gradient_rtol` | finite float, >= 0, default 1e-10 | Relative active-set KKT tolerance for quadratic/Huber, bounded/unbounded convergence; checkpoint-specific verification is recorded |
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
| Ray-Born `max_backtracks` | integer, > 0 | Maximum trials per direction proposal; guarded smoothing and projected-gradient fallbacks share the global work budget |
| Ray-Born `gradient_rtol` | finite float, >= 0, default 1e-8 | Training regularized gradient norm relative to its initial norm; not a global optimum certificate |
| Bent `gradient_rtol` | finite float, >= 0, default 1e-8 | Relative first-order check in the pixel or coefficient space; projected stationarity is recorded separately |
| Ray-Born `assume_unit_source` | boolean, default false | Explicit opt-in for dimensionless synthetic unit-source data only |
| `stopping.max_elapsed_s` | finite float, >= 0 | Wall-clock budget; native Green source/GMRES loops check cooperatively, external calls only between calls |

The straight-ray and Eikonal slowness penalties share units, but the squared-
slowness pressure penalty does not. Do not equate iteration counts or penalty
numbers across those models. A valid acquisition ROI must be supplied without
ground truth in truth-free deployment; this breast-map validation uses the
full image grid because no ROI was supplied. Setting `roi_update_only: true` alone
does not create a tissue support mask.

For native Bent, `line_search: false` disables repeated step halving, not the
descent acceptance rule. Gradient fallback trials still obey the global budget.

The native squared-slowness Gauss-Newton loop first projects a proposal onto its
speed and per-step bounds, then backtracks that actual feasible displacement.
Repeatedly clipping a large raw direction after each halving can otherwise
produce identical trial images. Logs separate the initial `step_length` from
`backtracking_fraction` and record displacement norms. Inner least-squares
accuracy does not certify an accepted outer step or a good reconstruction.
Likewise, `exact_data_fit` is a data-target stop, not a certificate that the
regularized objective gradient vanishes.

The legacy `normal_cg` comparison backend does not have the dynamic scaling of
the augmented solvers. A nonzero residual whose squared recurrence underflows
is reported as `arithmetic_precision_limit`, not convergence. Use the augmented
LSMR/LSQR path for scale-sensitive solves; an unchanged step with this flag is
not a solved subproblem.

The optional phase seed estimates group delay, not an exact first arrival.
Aliasing and multipath can remain after its fit-quality checks. Its held-out
frequencies are not used for phase unwrapping, initialization or physical-length
selection. Independent water calibration is a separate acquisition, not specimen
validation data. No parameters are automatically selected with ground truth.
Bent-ray smoothing acts on the update once before backtracking, not repeatedly
on the accumulated image. Lower smoothing is not automatically better: the
measured validation record contains counterexamples with smaller residuals and
worse SSIM.

### Optional finite-frequency TV penalty

The finite-frequency experiment can use `--optimizer trf` and
`--regularization-penalty smooth_tv`, with a positive
`--regularization-length-wavelengths` and damping.
This changes the inverse regularization, not the pressure data or registered ray
algorithms. The default remains the quadratic Laplacian. It minimizes

$$
\frac12\sum_{j\in\mathrm{train}}w_j|F_j(m)-d_j|^2
+\lambda\epsilon^2\sum_e\left(\sqrt{1+
\left(\frac{[D_\ell(m-m_0)]_e}{\epsilon}\right)^2}-1\right).
$$

Here $m=c^{-2}$, $D_\ell$ is the first difference multiplied by
$\ell/\Delta y$ or $\ell/\Delta x$, and only interior edges are included.
This is a discrete **smooth anisotropic TV** penalty, not post-filtering and not
a claim of grid-independent regularization. A reduced model retains this same
fine-grid penalty and its exact transpose. No GT support is inferred.

`--tv-transition-mps` (positive float, default 5) specifies
$\epsilon=2v_{\mathrm{transition}}/1500^3$: a reference-speed conversion for the
scaled gradient, not an exact nonlinear speed difference. Changing it also
changes the large-gradient penalty strength. The Laplacian and TV coefficients
are not interchangeable tuning optima. Actual penalty parameters and objective
values are recorded; data rows remain quadratic. The implementation uses
[SciPy's TRF loss interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
TV also appears in [finite-frequency USCT reconstruction](https://arxiv.org/html/1908.03302v1),
but this option is not a reproduction of that paper's full acquisition or solver.

See [operator_contract.md](operator_contract.md) for numerical conventions and
[physics_validation.md](physics_validation.md) for tests, references and evidence.
