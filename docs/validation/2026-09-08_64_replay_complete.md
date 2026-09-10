# Completed recovery replay of the 64-channel campaign

Historical record for 2026-09-08. Test counts, run status and timings below
belong to that recovery replay and its recorded source versions, not a fresh
verification of the current working tree.

This records actual commands executed after restoring the published source,
not planned runs or results copied from the historical 128-channel collection.
Reconstruction code was frozen at `2de5aea8f556bdecd071dfb7f624b77fcdd135b7`;
the four high-band pixel baselines used the unchanged original `78dea025`
source. The acceptance gate and ten additional tests are in `3bedb101`.
No reconstruction code or image defaults changed during this recovery replay.

## Baseline acceptance and recovery provenance

The rebuilt CPU workspace recovered commit
`2de5aea8f556bdecd071dfb7f624b77fcdd135b7` from its GitHub Actions source archive,
with recovered tree `b498a46277c32bfd21f1a9b0892b49c132518524`.
Before the acceptance-gate additions, the replay verified 61 file hashes and
all 12 HDF5 array/schema checks, reran the 199-test suite (199 passed, 2 external
MATLAB tests skipped), and completed all four original baselines. The original
source tree was kept separate and unchanged during those baseline runs.

`check_64_baselines.py` compares saved speed arrays, tissue/water/residual
scores, evaluation mask policy, receiver split, status, actual termination
reason, selected/completed iterations, and simultaneous stop triggers. It
records file hashes and explicit failed checks. Missing, corrupt, failed or
nonfinite output cannot pass. Its default absolute tolerances are **1e-9 m/s
for arrays and 1e-9 for numeric metrics**: floating-point reproducibility, not
a clinical quality threshold. Runtime equality is not required across CPU hosts.

The gate passed on the four newly replayed baselines in that historical run.
The campaign shell driver invokes it after the high-band pixel baselines and
before the basis experiment. Its ten added regression tests cover mismatched
arrays, nonfinite values, changed policy/split/selection, failed/missing results,
and metric changes; the resulting suite recorded 209 passes and the same two
MATLAB skips. The gate did not introduce a new imaging algorithm or change
reconstruction defaults.

```bash
python scripts/check_64_baselines.py \
  --reference /path/usct_handoff_64/baselines_64 \
  --actual /path/reproduced_64 \
  --out /path/fresh_baseline_acceptance.json
```

Original acquisition files were neither modified nor uploaded to GitHub.
Recovery logs, reconstructed arrays and timing curves were produced by that
replay, not borrowed from the first CPU run; timing reflects its concurrent
CPU load. The shell wrapper was syntax-checked, while its constituent commands
were executed individually, not as an additional end-to-end wrapper run.

## Completed outputs

- Package verification rerun after all experiments: 61 file hashes and all 12
  HDF5 array/schema checks pass. Original input files were not modified.
- Four complete high-band baselines pass the new array/metric/split/stop gate.
  Straight image max differences are at most 4.548e-13 m/s; Bent is identical.
- 28 reconstructed result sets were regenerated: three acquisitions times two
  parameterizations times four algorithms, plus four high-band phase controls.
  This includes only two different anatomies (high/low NBP D share anatomy).
- Every result has success status, a fixed implementation digest during its run,
  and ground_truth_used_for_stopping=false. Seven COMPLETE.json stage markers
  and all per-run metrics, arrays, resolved configs and iteration histories exist.
- Rendering completed with 28 CSV/JSON metric rows and 27 figures: three common-
  grayscale GT comparisons and 24 iteration/elapsed-time convergence plots.
  The renderer verifies identical data splits within each acquisition.
- Full local suite after the acceptance addition: 209 passed, 2 optional MATLAB
  skips. Black, Ruff, compilation, smoke and release audit pass. The GitHub
  tests job for 3bedb101 (run 34200946736) also completed successfully, including
  its smoke and release gates. No MATLAB, A100 or production FWI run is claimed.

## Numerical diagnostics from the recovery replay

Siddon adjoint relative error: 7.130e-16. Eikonal adjoint: 4.627e-16.
Eikonal directional derivative: 2.828e-9. Reduced-space straight and Eikonal
adjoint errors: 0 and 3.325e-16. Flat-interface Snell absolute error decreases
from 1.242 to 0.813 to 0.506 microseconds on 41/81/161 grids. Calibrated water
consistency is not treated as independent heterogeneous-model validation.

Full raw DTFT verification gives object/water relative errors 1.814e-17 and
1.237e-18. A separate time-slab recomputation also agrees to roundoff. Post-hoc
GT forward RMS misfit is 0.08457 us (straight) and 0.08742 us (Eikonal), combining
feature, physical-model and discretization error; no fitted correction was used.

An alternating five-batch, 30-calls-per-batch CPU microbenchmark gives CSR median
speedups of 7.52x forward and 9.58x adjoint. Reference/CSR relative differences
are 6.01e-16 and 2.05e-18. This excludes setup and is not an end-to-end speedup;
other CPU jobs were active. CSR adds 12,956,968 bytes and retains the ray lists.

All paired image scores reproduce the published results to the reported
precision, including the low-band OpenBreastUS Bent regression (35.469 to
39.239 m/s tissue RMSE). Basis32 remains opt-in; the single-phase feature is a
negative control. No consistent SIRT/SART image-quality improvement is claimed.

## Failures and execution scope retained

The original numerical-failure injection test was rerun against the unchanged
baseline source: all eight cases fail, reproducing the erroneous success/status
behavior. The corrected-source suite passes those same cases. One diagnostic
invocation initially imported the old editable installation and failed; after
installing the recovered source explicitly, the diagnostic rerun passed. An
interrupted full-test command was replaced by a complete logged 209-test run.
These failures are retained in delivery logs rather than counted as successes.

The campaign constituents were executed individually with per-stage source
snapshots. The shell wrapper was syntax-checked, not separately run as a second
full campaign. Recovery timing curves reflect the actual concurrent CPU load.
See the [results report](2026-09-08_64_channel_results.md) for all paired scores,
mathematical model, approximation limits and reproducible commands.
