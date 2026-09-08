# Agent evaluation and stopping contract

## Tissue image scores after reconstruction

`usct run` and `usct bench` evaluate the selected final reconstruction, after the
solver returns. Their primary `rmse`, `psnr`, and `ssim` now refer to **non-water
tissue**. The same values are retained as `tissue_*`; `full_image_*` and
`water_background_rmse` / `water_background_bias` remain separate. This is not
water-value subtraction or background correction of the reconstructed image.
Plots keep the full field of view and a common color scale.

The deterministic GT mask removes water connected to the image boundary using
four-neighbor connectivity and `abs(GT - c_water) <= tolerance`. Enclosed pixels
equal to water speed remain in tissue. All methods on a case share this mask;
its SHA-256, pixel count, water speed and tolerance are recorded in
`image_evaluation` in both metrics and metadata. The default water speed follows
the configured/case reference speed (otherwise 1500 m/s); tolerance is 0.1 m/s.
This rule assumes a known water background. Use a protocol-appropriate policy
for other acquisitions; it is not a general-purpose anatomical segmenter.

PSNR and SSIM use one range from the finite **full GT**, shared across methods
and regions, or an explicit positive protocol range. They never use each
reconstruction's contrast range. SSIM averages complete valid windows of up to
7 x 7 pixels, with a documented global-SSIM fallback for very small/thin masks.
Thus exterior pixels cannot alter tissue scores. Nonfinite reconstruction
values in the water are still a numerical failure, not hidden by the mask.

No GT mask is passed to initialization, the inversion ROI, source calibration,
line search, holdout selection or stopping. An absent GT or empty tissue gives
null primary scores and an explicit status, not a silent full-image fallback.
Standalone low-level `algorithm.run` image metrics retain their existing
full-image/explicit-ROI behavior; the common regional policy is applied by the
benchmark runner. Historical run files are not rewritten. Re-evaluations create
new reports with a region/version label; do not mix these with old scores.

Optional per-algorithm configuration:

```yaml
parameters:
  image_evaluation:
    primary_region: tissue  # or full_image for an explicitly declared protocol
    water_speed_mps: 1500.0
    water_tolerance_mps: 0.1
    # data_range_mps: 200.0  # optional fixed protocol range, not a fitted value
```

## Data separation

`make_data_split` accepts real `(tx, rx)` observations or complex
`(frequency, tx, rx)` observations. Receiver and frequency holdouts are whole
acquisition groups, not randomly interleaved scalar entries. With reciprocal
acquisitions, a transmitter colocated with a held-out receiver is removed from
training as well. Reverse-only channels are marked unused; receiver, frequency,
and joint validation masks are disjoint. Empty training sets, duplicate/out-of-
range indices, invalid weights and misspelled options fail explicitly.

The split records indices, seed, counts, and a SHA-256 of the masks. A split used
for model selection or early stopping is **validation**, not an untouched test
set. Independent final testing requires a separately reserved acquisition/case.
In particular, fitting a source spectrum or warm start on the entire measured
dataset before splitting would leak validation data; callers must not do that.

## Residuals without ground truth

`residual_statistics` computes real/complex L2 norm, observed norm, relative L2
residual, magnitude RMSE/MAE, sample counts, and precision-weighted versions.
Weights enter as `sqrt(w)`. It never discards imaginary components. Missing
observations are excluded; a non-finite prediction on an active observation is a
numerical failure. Empty holdouts and zero observed norms produce explicit null
metrics, not a fabricated zero score. No ground-truth image is needed.

Travel-time, scattered-pressure and full-pressure residuals have different units
and denominators. Even a dimensionless relative residual is not automatically
comparable across these observation models. Compare algorithms on the same
held-out measurement representation and preprocessing/calibration when possible;
also report raw residual norms and acquisition counts. Full-pressure residuals
can be dominated by the incident field, so report scattered/contrast residuals
separately when a calibrated background is available.

## OR stopping policy

`StopPolicy` / `StopMonitor` support an iteration cap, elapsed-time budget,
forward/adjoint-call budgets, a target relative residual, the discrepancy
principle (`||sqrt(W) r|| <= factor * noise_norm`), a small relative model update,
objective plateau with patience, validation plateau with patience, and an
explicitly opted-in ground-truth RMSE target. These are OR rules. Safety budgets
and exact data fit do not wait for `min_iterations`. All simultaneous triggers
are recorded, with a deterministic primary reason.

Noise norms must be estimated in the **same training domain and weighting** as
the residual; an unweighted per-sample noise standard deviation is not the same
quantity. A quality target is disabled by default and requires
`allow_ground_truth_stopping: true`; a truth-free deployment must not enable it.

Validation checkpoint restoration records both the last complete iteration and
the selected iteration. Numerical failure retains the last complete finite
state. Algorithm loops can record additional reasons such as `line_search_failed`
or `stationary_gradient`. A model-capacity plateau is a valid termination, not
proof of adequate image quality.

`WorkLedger` checks budgets before operator calls. Time limits are enforced
between calls; an already-running PDE solve cannot be preempted by this Python
API. Counts include evaluation and line-search calls, and must be accompanied by
model-specific source-solve/subset counts. One CGLS step, one SART sweep, one
Gauss-Newton outer step, and one external FWI iteration are **not** equivalent work.

## Integration and evidence

CGLS, SIRT, SART, Eikonal, native Ray-Born and the opt-in controlled Helmholtz
bridge call this monitor inside their loops. Completed checkpoints, not partially
computed updates, are returned after budget exhaustion. Ray-Born setup and water
calibration are also counted in current code. The production external FWI result
adapter rejects unsupported online-control requests and marks its external stop
reason unavailable. Importing a finished artifact never retroactively stops it.

`evaluation.receiver`, `evaluation.frequency` and `evaluation.joint` contain
disjoint holdout statistics. Full-pressure and contrast-pressure residuals are
both retained for Ray-Born. Image metrics reject nonfinite reconstructions rather
than hiding failed pixels. The post-reconstruction image policy above is separate
from the inversion ROI: a missing inversion ROI does not infer a breast support.
A uniform returned image can have a moderate SSIM, so no single score is an
acceptance certificate.

See [the independent eight-case report](validation/2026-09-08_physics.md) for
measured results and remaining limitations, including a validation-selected
initial model. Holdouts used to stop/select are not an independent test set.
