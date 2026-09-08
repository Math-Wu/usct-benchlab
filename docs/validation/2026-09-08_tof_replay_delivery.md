# ToF model audit: verified replay delivery

This is a freshly executed CPU replay after runtime recovery, not an aggregate of historical test claims. The numerical package was unchanged throughout all final reconstructions. Source snapshot 307ba1f19160245111461cb18555e23f9652598b has tree eb4852ba4e7b66925d9796504dd0fce27caff2ca, verified against the GitHub Actions source archive. Later documentation commits do not change those experiments.

## Execution and scope

- 61 supplied file hashes and 12 HDF5 array hashes/schemas passed, again checked at delivery. Original private data were not modified or uploaded to Git.
- Four full 64-channel baselines passed the array, split, metric-policy and stopping comparator. This is not a quick three-iteration replay.
- 68 final successful reconstructions, all finite, with no ground-truth stopping and no runtime-package changes. They represent two breast anatomies (high/low D share anatomy), three acquisitions, and one smooth analytic control, not 68 independent cases.
- Final local delivery suite: **236 passed, 2 external-MATLAB skips**. Black, Ruff, compileall and release audit passed. GitHub tests and physics-validation for 307ba1f succeeded.
- An additional 15 primary attribution runs were repeated after copying non-training observations verbatim rather than reconstructing them by subtract/add arithmetic. All five final controls passed bit-exact validation/unused-data checks. The earlier 15 are retained separately and excluded from the 68 final comparison rows.
- No new k-Wave simulation, production FWI, MATLAB, A100, original 128-channel campaign, clinical data or remaining six low-band cases was executed.

The final 68 rows comprise high-D S/E/K x four methods (12), low-D/OB S/E/K x three straight methods (18), smooth positive controls (4), training-CGLS-initialized Bent (2), primary five-seed attribution (15), secondary five-seed attribution (15), and two coarse CGLS runs compared with direct quadratic solutions (2). Independent reference arrays are not counted as new patients.

## Matched-model versus measured-feature controls

S = A(1/c_GT - 1/c_water), E = T_h(1/c_GT) - T_h(1/c_water), K = the supplied k-Wave-derived envelope delay. S and E are explicitly GT-generated, same-discrete-model diagnostics with inverse-crime risk; never substitute them for realistic K rankings. Geometry, valid channels, weights, split and original solver settings are fixed. No GT initialization, inversion ROI, offset fitting or stopping is used.

High-D tissue RMSE (m/s) / SSIM:

| Method | S | E | K |
|---|---:|---:|---:|
| CGLS | 23.32377 / .210493 | 27.59828 / .127440 | 27.19321 / .114274 |
| SIRT | 24.33914 / .194902 | 27.67864 / .124976 | 27.00560 / .121096 |
| SART | 24.45367 / .190073 | 27.66338 / .124135 | 26.92774 / .122694 |
| Bent | 25.73943 / .177183 | 26.27108 / .139334 | 27.08777 / .130122 |

Own-model data improve some results but do not recover the anatomical fine detail. Conversely, smooth analytic matched controls give RMSE 1.256/1.578/1.453/.931 and SSIM .987/.983/.984/.991. This is a conditional software positive control, not proof of k-Wave compatibility.

## Separating possible causes

**Picker implementation versus observable:** repicking all 2752 original valid high-D envelope delays is bit-exact. For 128 fixed water traces digitally shifted by +/-250 ns, envelope RMS error is about .0825 ns (sampling interval 16.1317 ns). This checks digital translation behavior, not physical first arrivals. Actual envelope/xcorr delays differ by RMS 64.23 ns on 2746 common channels; six invalid xcorr channels are excluded. Shift-plus-scalar-gain waveform residual medians remain 20.55%/16.73%. Actual object pulses are not merely translated/scaled water pulses.

**GT forward consistency:** high-D S/K and E/K RMS discrepancies are 84.57/87.42 ns, about 43.6%/45.1% of differential-delay signal norm. Low-D values are 120.05/114.31 ns; low-OB 458.59/499.80 ns. These mix observable, physical and discretization error; they are not measured noise standard deviations.

**Discretization:** on eight fixed transmitters and 344 valid channels, piecewise-field-preserving 256/512/1024 refinement gives E/K RMS 84.54/109.32/137.08 ns, with consecutive changes 35.95/36.30 ns. A recorded-interpolation 1024/2048 probe changes 31.57 ns. Neither certifies convergence or byte-exact recovery of the unavailable original simulation_input.mat. Better continuum accuracy need not fit a finite-band feature more closely.

**Error cancellation:** K-S = (E-S)+(K-E). High-D weighted component RMS values are 110.34 and 87.43 ns, but total RMS is 84.58 ns; component cosine is -.6564. Do not drop the cross term or call K-E pure picker error.

**Fixed-objective optimization:** independent fine-grid LSMR, same training mask, lambda=.02 and Laplacian, reaches the specified 1e-10 tolerance after 3101/3214 iterations (stop code 2). Both solutions satisfy speed bounds. S tissue RMSE changes 23.32377 to 23.28540; K 27.19321 to 27.05793. Thus longer optimization of this fixed objective gives little image improvement, without proving a universal prior/model ceiling. A 32x32 coefficient quadratic is also solved directly; its Hessian is numerically positive definite and relative normal residual is about 4e-16. Direct/CGLS RMSE agree within .0005 m/s on S and .0002 m/s on K. These are floating-point, fixed-objective references, not interval proofs or SSIM bounds.

**Identifiability:** 2752 directed rays contain 1376 reciprocal pairs; training has 1049 pairs versus 65536 pixel unknowns. A geometry-only null-space witness differs from uniform water by RMS 10.0047 m/s in a fixed disk, stays within 1465.11-1536.52 m/s, yet has maximum recomputed all-valid Siddon delay below 1e-11 ns. A prior may exclude the witness. It is not a full-wave/Bent null vector or a universal image-quality bound.

## Equal-energy, different-structure experiment

Primary seeds **0..4** shuffle only training reciprocal-pair errors relative to S. Weights, weighted mean/energy and antisymmetric error are conserved. Actual validation and unused observations are copied verbatim. GT assists construction of these oracle attribution controls, never a deployable data correction. This is not iid Gaussian noise.

| Method | Original K tissue RMSE | Shuffled mean +/- sample SD | Original water RMSE -> shuffled mean |
|---|---:|---:|---:|
| CGLS | 27.19321 | 24.26581 +/- .13500 | 3.93543 -> 7.97438 |
| SIRT | 27.00560 | 24.73788 +/- .13459 | 2.98990 -> 6.80286 |
| SART | 26.92774 | 24.83586 +/- .11780 | 2.84870 -> 6.38628 |

The same error energy produces different tissue degradation; error structure matters. Water gets worse, so this is not an overall imaging improvement. Five seeds are not five patients. Bent attribution was not run. Supplementary seeds 20260908..20260912 shuffle within train/validation/unused separately; their validation values change, so those runs are reported separately.

## Consequence for the library and agent

Keep the four traditional methods and the common raw-pressure k-Wave benchmark. Maintain separate tracks for matched-discrete solver checks, independent geometric/ToF accuracy, and end-to-end pressure-acquisition performance. CGLS/SIRT/SART share one physics model. Common-acquisition image-quality/cost comparison with FWI is meaningful; mixed ToF/pressure residual rankings and equal-iteration budgets are not.

Prioritize observable-specific forward/Jacobian definitions, independent water/known-phantom calibration, Eikonal convergence, and finite-frequency traveltime sensitivity before more blind iteration/regularization sweeps. Finite-frequency methods must be registered as distinct models, not silently substituted for traditional Bent. Frequency holdouts must occur before any feature extraction that could use those frequencies. Tiny residuals, reciprocity and high SNR do not certify image quality.

Not all input adaptation has been resolved: continuous first-arrival error, actual fine simulation-field correspondence and finite-band effects remain incompletely separated. Raw low-band data are absent. No universal four-method accuracy ceiling or clinical conclusion is claimed.

The conversation delivery contains the detailed Chinese report, 68-row CSV/JSON, finite-array/split/stopping gates, three uniformly scaled figure panels, configurations, result HDF5s, diagnostics, test logs, checksums, and a relocatable command driver. Driver commands were executed individually; the convenience shell driver itself is syntax-checked, not claimed as an additional end-to-end run.
