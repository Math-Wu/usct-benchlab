# WUST full-wave inversion

`fwi_wust -> pinned WUST runtime -> MATLAB -> CUDA Block-LU` is the only
production FWI path. CPU is reference/debug only, never an automatic fallback.
Ordinary installation and CI require neither MATLAB nor GPU. Reconstruction is
2-D sound-speed-only; no diffusion runtime or attenuation reconstruction is included.

`TinyFWIAlgorithm` remains directly importable from
`usctbench.algorithms.fwi.tiny` for mathematical regression tests only. It is
not registered as a CLI algorithm or exposed through Agent discovery.

## Deployment

Use clean `lucian-dw/WaveformInversionUST` at exactly
`79e347015be64cca88bacf591b4eed0952398800` (version `0.2.0-dev.1`, schema 1).
SHA and verified source manifest are authoritative, not a moving branch.
Build the maintained CUDA MEX on the deployment host. This development pin is
not itself final release certification.

```bash
export USCT_WUST_ROOT=/path/to/approved/WaveformInversionUST
python "$USCT_WUST_ROOT/Runtime/python/wust_runtime.py" describe --json
usct describe-algorithm fwi_wust --json
usct run fwi_wust \
  --case /path/to/frequency_case.h5 \
  --config configs/algorithms/fwi_wust.yaml \
  --out runs/single_fwi
```

The example demonstrates syntax, not calibrated settings. Explicitly resolve
sound-speed bounds and PML thickness for your grid. Expert YAML may configure
`matlab_executable`, `cuda_visible_devices`, `runtime_root`, `scratch_root`,
and reference-only `backend: cpu`. These controls are hidden from Agent admission.

## Existing observations

Supply complex total pressure `freq_data[frequency,tx,rx]`, positive unique
`frequencies_hz`, and an explicit binary `valid_mask[tx,rx]` or
`valid_mask[frequency,tx,rx]`. Valid zeros remain data. Travel times,
scattered-only fields, water ratios and phase-only measurements are not accepted.

Case metadata must include `pressure_contract` with `fourier_sign` (-1 or +1),
`real_pressure`, `pressure_type: total_pressure`, `data_units`, and
`spectrum_normalization`, using WUST-supported declarations. The documented
pressure-preserving k-Wave importer supplies an explicit positive-DTFT quadrature;
a time-harmonic sign alone cannot establish the transform convention.

BenchLab writes generic frequency input with image axes [y,x], spacing [dy,dx],
pixel-edge origin, and exactly one physical-coordinate swap from public [y,x]
to [x,y]. WUST ingestion owns sorting, pressure canonicalization, Fourier
conversion, snapping and MATLAB indexing. GT never creates/modifies observations.
WUST fits complex source scale per TX/frequency; neither water reference nor
Born source spectrum is required.

## Initialization and update region

Agent initialization is `reference` (requires case
`reference_sound_speed_mps`) or `scalar` with `initial_sound_speed_mps`.
Experts can supply `initialization: map` and `initial_map_mps`.
Maps must match exact grid/bounds without resizing or clipping. No arbitrary
artifact resolver is advertised.

The default scientific update mask excludes pixels within PML thickness plus
one maximum grid spacing of each image edge. Experts may override `update_mask`.
WUST independently validates numerical/PML constraints and never silently crops
an invalid request. GT never defines the mask.

## Schedules and completion

`frequency_schedule_hz` uses exact matches against WUST's canonical frequencies.
Null means one update per available frequency; repetitions repeat updates.
Empty schedule or zero iteration budget yields validated initialization only.

Executed updates are the minimum of schedule length, requested max_iterations,
and budget cap max_iterations. One required monotonic elapsed-time budget covers
serialization, discovery/probe, ingestion, reconstruction and parsing.
Timeout terminates the runtime and returns failure with budget semantics, not
a success image. Forward/adjoint-call caps, non-null update_rtol, legacy stopping,
online numerical convergence and validation selection are unsupported.

Optimization variable: slowness. Update variable: full slowness in s/m.
Relative L2 change over the declared update mask is normalized by the previous
slowness norm and is diagnostic only. One frequency-schedule item equals one
complete update. Schedule completion is not convergence. Truncation and zero
updates report max_iterations with budget semantics.

## Results and validation

Authoritative float64 c_mps[y,x] maps directly to the result. Per-update loss
and residual are **before-update diagnostics**, not final-model residuals.
Unavailable final residual remains null. GT is used only for post-run metrics.

Results retain records, resolved config, schedules, budgets, exact source
identity, MATLAB/GPU/CUDA/MEX facts, and input/output hashes. Interchange files
and logs stay in the run's wust_directory, outside source-control assets.

`scripts/validate_wust_integration.py` runs the real cross-repository gate on an
independently generated coherent fixture. Normal tests use protocol stubs.
Fresh-MEX provenance, GPU numerical checks, CPU/GPU comparisons and representative
deployment-size reconstruction are release gates; CPU and tiny runs alone do
not establish production certification.
