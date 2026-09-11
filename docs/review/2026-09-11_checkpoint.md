# Scientific review checkpoint — 2026-09-11

Base: `3af32a3f5653f5644e39e1f081b5dfd6cbef3978` on `work/physics-agent-validation`.
Review branch: `review/numerics-contracts-20260911`.
The consumer `zoe5xy/inverse_problem_agent`, branch `codex/linear`, is read-only.

The Actions source archive was extracted and indexed locally. Its Git tree is
`356b123930ffebcf263fbed0606b5c14ea9e7d5c`, identical to the base commit tree.
Fresh pre-change pytest: **548 passed, 3 skipped** (25.90 s, Python 3.13.5).
`ruff check .` passed with the pinned 0.16.6 tool. `black --check .` failed only
on the pre-existing `.github/scripts/complete_waveform_stage.py` formatting.
Pinned Black 26.5.1 / Ruff 0.16.6 were installed from available offline wheels.
CLI Git clone failed with DNS resolution; GitHub connector reads/writes work.

A new numerical counterexample reproduces an unconditional exact-data-fit stop:
`min_x 0.5 ||x-1||^2 + 0.5 ||x||^2`, starting at `x=1`, returns `x=1` in the base
although the unique minimizer is `x=0.5`. The data residual vanishes but the
regularization gradient does not. A locally tested correction requires the
complete nonnegative objective to vanish before this shortcut is accepted.
Fresh suite after that correction: **549 passed, 3 skipped** (20.62 s).
The correction and tests will be included in a separate code commit; this
checkpoint is not a claim that the whole refactor has been completed.

Process logs and numerical fixtures are saved outside the repository under the
session review directory. No private datasets are committed.
