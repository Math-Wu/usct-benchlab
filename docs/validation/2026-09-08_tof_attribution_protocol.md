# Current-session ToF error attribution protocol

Start commit: fe505f3b626a8999333222df534ddacc81dfb23e. This is a NEW replay, not the previous checkpoint results. Source recovered from its GitHub Actions archive; local tree fb56e63d0f42578a89f7995bfe4516d54e910193 matches. Command-line Git DNS failed; connected API reads/writes are available. Input package stays private, outside Git.

Completed before reconstruction comparisons: 61 file hashes and 12 HDF5 schemas/array hashes passed; 218 tests passed and 2 external MATLAB tests skipped. Offline editable installation succeeded. CPU only, approximately 4 cores / 4 GiB. Full high-D baseline reproduction is running under unchanged 120/200/100/20 caps. Historical audit JSON is NOT accepted as current evidence.

Frozen experiment policy:
- Re-execute matched straight / matched discrete Eikonal / k-Wave-ToF controls for high D, low D, low OpenBreastUS. Same 64 TX/RX, 256-square grid, original valid mask, weights, seed-42 receiver validation, water initialization and original per-method budgets. All synthetic ToFs carry oracle diagnostic labels; no change to the private source data.
- Repeat known digital water-trace shifts, actual raw repicking, waveform shift/gain fit and fixed-field Eikonal refinement using committed diagnostics. No low-band raw extraction claim.
- Repeat the geometry-only null-space witness and independent augmented LSMR optimization diagnostics. Neither a finite iteration count nor one null-space witness proves a universal image-quality ceiling.
- NEW: decompose the vector residual at GT into straight-to-Eikonal and Eikonal-to-observable terms, including their cross term. Never subtract RMSEs to assign causal percentages. On reciprocal pairs, report the weighted irreducible residual of *any exactly reciprocal* prediction.
- NEW: compare the real high-D straight-model discrepancy with five seeded, pair-preserving shuffled controls of equal weighted energy. Shuffle whitened *unordered-pair* residuals separately inside train / validation / unused groups. Preserve each group's norm and observed reciprocity defect; do not move a validation error into a training pair. No GT offset correction is applied to measured data. Invert with CGLS/SIRT/SART under the same caps. These are oracle attribution controls, not new acquired data or a performance improvement.
- Repeat smooth analytic matched-model controls (not derived from breast GT) and two data-only Bent warm-start controls if resources permit. Every completed run gets output arrays, metrics, stop reason and work counts; remaining steps are marked not executed.

No default production reconstruction parameters/operators or FWI code are changed in this study. No larger sweep, postprocessing or GT-driven early stopping. All development GT has been viewed: this is validation/research, not blind testing. Final source/results/test evidence will be pushed incrementally on the same branch and archived outside the repository.
