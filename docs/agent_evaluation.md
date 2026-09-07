# Agent evaluation and stopping contract

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

## Low-level validation checkpoint

The initial API checkpoint passes 67 tests across the repository, including
complex residuals, invalid predictions, zero norms, deterministic grouped masks,
reciprocity exclusion, OR priority, budget enforcement, best-checkpoint recovery,
and explicit oracle-only stopping. Algorithm-loop integration is documented in
the subsequent implementation report; these APIs do not retroactively stop an
already-completed external reconstruction artifact.
