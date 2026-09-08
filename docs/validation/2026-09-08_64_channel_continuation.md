# 64-channel continuation: executed checkpoints

Base: `78dea0254e6e4eff9977e2be78f898f7c135f3df`, branch
`work/physics-agent-validation`. These are new CPU-session results, not the
historical 128-channel experiments or A100 measurements. No pressure data is
committed. The user-supplied package is an observation subset of an existing
128-channel k-Wave simulation, not a new 64-channel simulation.

## Checkpoint 1: data and baseline, 2026-09-08

The attachment was extracted. `tools/verify.py --repo usct-benchlab` passed
61 file hashes and all array hashes/schema checks for 12 HDF5 files. The source
archive, after restoring its executable modes, has tree
`603d496bf533f7a1ac6448842b7f79761a6f98f2`, identical to the base GitHub tree.
The raw high-band object/water tensors are (6381,64,64), float32; the other
files do not provide new raw time traces. The original files remain unchanged.

Online pip and command-line Git failed DNS resolution. Editable installation
succeeded offline using the existing exact Black/Ruff wheels and installed
numerical dependencies. Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Numba 0.65.1.
After creating a local Git index for the source snapshot, the unchanged full
suite returned **161 passed, 2 skipped**. Before creating that index, the two
release tests requiring `git ls-files` failed; those were environment failures.
The skipped external MATLAB tests were not executed.

The full frozen high-band D baselines below were run without `--quick`:

| Method | Tissue RMSE (m/s) | Tissue SSIM | Selected/completed | Stop |
| --- | ---: | ---: | --- | --- |
| CGLS | 27.193208708966395 | 0.114274175695250 | 120/120 | max_iterations |
| SIRT | 27.005598960685713 | 0.121096114777781 | 200/200 | max_iterations |
| SART | 26.927737346553016 | 0.122694304335006 | 100/100 | max_iterations |

Each saved speed array differs from the packaged baseline by at most
4.548e-13 m/s; SSIM differences are below 1.5e-12. Evaluation splits and stop
reasons match exactly. The Bent run was still executing when this checkpoint
was saved; no result is asserted for it here.

Validation here is the receiver holdout used for stopping/selection, not an
independent test. All image scores are post-reconstruction tissue scores using
the package evaluation contract. No algorithm improvement is claimed by this
baseline checkpoint. Later executed checkpoints will be appended separately.
