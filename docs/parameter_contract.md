# Parameter contract

Algorithm parameter definitions live in `algorithms/parameters.py` and
`algorithms/fwi/parameters.py`. YAML files are experiment overrides, not default
definitions. Constraints describe numerical admissibility, not experimentally
validated search ranges. No weak/moderate/strong presets are provided.

## Shared validation

`validate_algorithm_config(id, config)` is the common boundary for YAML loading
and direct algorithm `run()` calls. It returns a new config and never modifies
the caller's dictionary. Explicit validation raises `ValueError`; the algorithm
result API returns `status=failed` with an `invalid configuration` reason before
running any solver. Unknown and unused fields fail, including nested inner
solver/evaluation settings. YAML top-level keys also fail closed.

```python
from usctbench.algorithms.parameters import CGLSParameters
from usctbench.algorithms.configuration import validate_algorithm_config
from usctbench.core.schema import AlgorithmConfig

config = validate_algorithm_config(
    "straight_cgls",
    AlgorithmConfig(
        parameters=CGLSParameters(regularization="laplacian"),
        run_controls={"max_iterations": 30},
    ),
)
```

Each field provides a description, JSON type, default, numerical constraints,
units where established and `exposure`: `agent`, `advanced` or `internal`.
Null means case-dependent or delegated; it is not substituted with an invented
physical constant. Penalty amplitudes have objective/operator-dependent units.

## Legacy migration

| Legacy input | Canonical representation / policy |
| --- | --- |
| `lambda` | `regularization_lambda` |
| `damping` | `regularization_lambda = sqrt(damping)`; damping is the squared coefficient |
| `l2`, `roughness` regularizers | `identity`, `laplacian` |
| `irls`, `huber_irls` robust loss | `huber` |
| `ray_weight_min`, `ray_weight_threshold` | `min_ray_weight` |
| `roi_aware_laplacian` | `roi_laplacian` |
| `iterations`, `outer_iterations`, Tiny `steps` | Checked legacy run budgets, not typed algorithm hyperparameters |
| Fixed Born `inner_iterations` | Checked total iteration budget, not a nonlinear inner solve |
| `parameters.stopping` | Checked legacy policy; remains opt-in compatibility, not the Agent interface |
| SIRT/SART `regularization_lambda`, CGLS `subsets`, straight-ray `inner_iterations` | Reject: these settings never controlled those solvers; remove from shared dictionaries |
| Bent `roi_laplacian` | Reject: it was only recorded in metadata, not applied as a separate Bent regularizer switch |

Equivalent native aliases are accepted; conflicting aliases/budgets fail. WUST uses only its canonical typed parameters, with no legacy execution aliases.

Fixed Born normalizes `inner_iterations` before checking `iterations`, legacy
`stopping.max_iterations` and `run_controls.max_iterations`; repeated validation
has the same admission decision. Tiny rejects legacy `stopping` and fitting/split
`evaluation`, including empty dictionaries: neither is implemented by that solver.
Its `steps` loop and post-hoc image evaluation remain unchanged.

Both Born models allow `max_cache_bytes=0` (no Green cache) and require
`0 < green_solver_rtol < 1`, matching the numerical operator.

New callers put budgets in `run_controls` / `budget_caps`. The loader now
preserves these top-level fields (previously it discarded them). Existing YAML
iteration and stopping dictionaries remain supported without changing their
legacy stopping defaults. There is no new universal `update_rtol`. The default
iteration table in `algorithms/configuration.py` records existing kernel
defaults, not benchmark YAML choices. Time/call caps retain cooperative semantics.

Native default parameters are resolved before execution. Fixed-background Born
has a smaller model without nonlinear update controls. Tests compare arrays and
residual curves before/after validation, not only successful process exits.

## Supported configurations and boundaries

All algorithm YAML files use validated typed models. Production `fwi_wust` uses the small `WUSTParameters` model. Bounds and PML must be explicitly supplied for execution. Reference initialization requires declared case reference speed; there is no injected 1500 m/s fallback. Initialization maps and deployment paths are expert/internal only.

WUST accepts only iteration and elapsed-time budgets. One schedule entry is one update; repeated Hz values repeat updates. Schedule truncation occurs after WUST ingestion. A single hard deadline includes all subprocesses and interchange work. Call caps, non-null `update_rtol` and legacy stopping dictionaries are rejected. CPU is reference-only; production requires the pinned CUDA runtime. See [fwi.md](fwi.md).

Calibration arrays are validated runtime inputs, never Agent hyperparameters.
Initial/background image arrays remain expert-only; default Agent interfaces
must not accept arbitrary artifact paths. Attenuation reconstruction and its typed fields have been removed.

## Intentional breaking change

Round 3 removes `attenuation_sirt`, canonical `log_amp` measurements, attenuation
GT/results, and their HDF5 read/write paths. No compatibility or migration is
provided for attenuation-bearing BenchLab files. Sound-speed-only cases/results
remain supported. Raw dataset maps outside this scope are ignored; no frequency
or unit conversion is inferred. Removed joint-FWI schedules now fail validation.
