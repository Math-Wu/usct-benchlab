# Model mismatch versus inversion limitations: frozen diagnostic protocol

Status: protocol, NOT completed results. Start: 8be7229352588c13522b881b1b8d13e23e9c3de9. Local recovered source tree 84476fa0463ebe8149158078b4ff75bc4e45c465 matches that commit. The supplied 64-channel package has just passed all 61 file hashes and 12 HDF5 array/schema checks. CPU Python is available; command-line Git cannot resolve github.com, so publishing uses the connected GitHub API. No new branch or production FWI changes.

Question: distinguish implementation/optimization limits, spatial information limits, ray-model mismatch, and finite-band arrival-feature mismatch. Good matched-model reconstructions are not evidence of full-wave performance.

## Controls frozen before seeing new reconstruction results

1. Re-run all four full high-band D baselines with the supplied configurations; compare arrays, evaluation splits and stopping records. Archive current test output, not historical counts.
2. With the same geometry, valid channels, precision weights, initialization and receiver validation policy, generate diagnostic straight-Siddon and Eikonal delays from the provided property map. Invert these and compare with the k-Wave-derived delay. Label GT-generated control observations explicitly; never use their scores to claim performance on k-Wave/clinical data. Do not feed GT to the reconstruction initialization or validation selection.
3. Diagnose travel-time consistency at GT for high D and at least low D and low OpenBreastUS. Record straight/eikonal discrepancy, feature discrepancy, reciprocity and known-water controls. Do not fit GT offsets to repair observations.
4. On the high-D raw object/water acquisition, run picker translation-equivariance controls using digitally shifted WATER traces with known shifts, and waveform-distortion diagnostics. A shift-only control has no scattering; it checks the picker and sign/time convention, not the physical model. Any alternative picker policy must be frozen independently of GT and keep the same channels/weights or report its changed coverage separately.
5. Separate spatial representation from information: analyze the rank/spectrum of the fixed 32x32 bilinear parameterization, exact-model least-squares controls and analytic null-space ambiguity where feasible. An oracle model-space projection is only a labelled diagnostic and not a reconstruction result.
6. Bound Eikonal discretization effects by forward predictions on refined grids for the same fixed piecewise field. Use a streaming diagnostic to avoid retaining every source tape at large grids. A finite grid study is not a proof of continuum accuracy.

## Reporting rules

Save protocol, inputs' hashes, actual commands, per-run outputs and checkpoints incrementally outside Git. Only code, tests and aggregate evidence go to Git; do not upload the private data package. Preserve all negative results. Do not equate CGLS steps, SART sweeps or Bent outer iterations, or directly rank ToF and pressure residuals. Validation used for stopping is not independent testing. The intended outcome is a evidence-based development decision, not a guarantee that all four methods improve. No A100/MATLAB/full 128-channel/clinical run will be claimed unless actually executed. Final report will enumerate which controls completed and which remain blocked.
