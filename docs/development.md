# Development

This document records the release hygiene commands for `usct-benchlab`.

## Format and Lint

```bash
black src tests scripts
ruff check src tests scripts --fix
```

## Compile and Test

```bash
python -m compileall src tests
pytest -q
```

Tests are organized by behavior, not by experiment date. Keep independent
forward/adjoint checks, dense solver references, real bug regressions, data-split
isolation, stopping/budget behavior, and CLI/release contracts. Parameterize only
when a case exercises a distinct code path or numerical failure mode; avoid full
Cartesian products of methods, units, penalties, and invalid arguments when
representative combinations cover the same contract. Do not replace removed
parameter combinations with hidden loops merely to lower the displayed count.

One-off sweeps, run launchers, copied source snapshots, and their logs belong in
the external run workspace, not in `src/` or `tests/`. Preserve raw observations,
final results, source identities, and confirmed failure evidence. Delete only
regenerable caches or verified redundant/failed startup files; never clean an
active run's source or checkpoints. Consolidate superseded progress notes into
the final report instead of accumulating another archive tree.

## CLI Smoke

```bash
usct --help
usct list-algorithms
bash scripts/run_smoke.sh
```

## Release Audit

```bash
python scripts/audit_release.py
```

Additional checks:

```bash
git ls-files | grep -E '\.(h5|hdf5|mat|npy|npz|zarr|pt|pth|ckpt|pkl)$' || true
grep -RIn "/home/example\|/Users/example" README.md docs configs scripts src tests .env.example || true
```

The first command should print nothing. The second command is an example path
pattern check; adapt the user names for local audits.

## Git Hygiene

Commit source, configs, tests, docs, scripts, and small README figures. Do not
commit raw datasets, generated benchmark runs, external repositories, or model
checkpoints.
