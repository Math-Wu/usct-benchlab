# Research Agent algorithm API

This package is the authority for physical variants and parameter permissions.
Consumers must not infer physics from a registered name or load arbitrary YAML
to decide which controls an autonomous Agent may submit. The API supports
research reconstruction, not clinical decision-making.

## Static discovery

```bash
usct list-algorithms --json
usct describe-algorithm straight_cgls --json
usct describe-algorithm bent_ray_gn --json
usct describe-algorithm rwave_adapter --variant full_green_nonlinear --json
usct describe-algorithm fwi_kwave_adapter --variant controlled --json
usct describe-algorithm fwi_kwave_adapter --variant external_pipeline --json
```

Stdout is a single JSON value. Errors go to stderr and return a nonzero status.
The schema version is `usct.algorithm.v1`. An entry supplies:

- Stable algorithm id and explicit family/variant/model.
- Required observation domains and fields (`all_of`, plus alternative groups).
- Geometry, runtime requirements, limitations and deterministic status.
- `parameter_model`, filtered `config_schema`, `allowed_parameters` and
  `default_parameters`, all derived from the same typed definitions.
- Separate `compute_budget` request/cap schemas and actual iteration units.

`rwave_adapter` is the legacy command name for native Ray-Born implementations:
`wkb_nonlinear`, `full_green_nonlinear`, `wkb_fixed`, `full_green_fixed`.
This does not claim an upstream r-Wave reproduction. WKB sensitivity is not
the exact derivative of the discretized WKB prediction.

FWI variants are `import_result` (the historical default), `external_pipeline`,
and `controlled`. Import does not reconstruct and exposes no autonomous
hyperparameters. External pipeline configuration does not imply availability or
online stopping support. Controlled FWI has a separate parameter model and
discrete pressure/source requirements.

The existing diffusion adapter is listed with `typed_interface_available=false`:
it remains expert-only and cannot be admitted through this Agent API. Neither
that integration nor `inverse_problem_agent` was modified in these rounds.

## Enforced admission

```python
from usctbench.cli import register_builtin_algorithms
from usctbench.core.algorithm_specs import make_agent_config
from usctbench.core.registry import get_algorithm

register_builtin_algorithms()
config = make_agent_config(
    "straight_cgls",
    {"regularization": "laplacian", "sound_speed_bounds_mps": [1300, 1700]},
    run_controls={"max_iterations": 30},
    budget_caps={"max_iterations": 20},
)
# With a prepared USCTCase:
# result = get_algorithm("straight_cgls").run(case, config)
```

These bounds and budgets illustrate syntax, not calibrated prescriptions.
Only canonical Agent fields/values are accepted. Unknown keys, advanced/internal
fields, legacy aliases, nested runtime objects and conflicting variant selectors
fail before executing an algorithm. Advanced parameters remain available through
the expert Python/YAML interface documented in [parameter_contract.md](parameter_contract.md).

`trusted_parameters` in `make_agent_config` is a **deployment-owned** mapping for
approved runtime configuration and artifacts. Never populate it from model output.
AlgorithmConfig/direct Python is an expert API, not a sandbox for untrusted Agent
arguments. The `run_controls` argument of the Agent factory permits only budget
fields; advanced stopping tolerances require a separate expert policy. No
universal update tolerance or uncalibrated semantic preset is introduced.

## Case-bound capabilities

```bash
usct describe-algorithm straight_cgls --json --case /path/to/prepared_case.h5
```

The optional `case_capabilities` object uses `usct.case_capabilities.v1` and
reports available frequencies, calibration presence, runtime availability (null
until checked by the deployment), approved initialization artifact ids, resolved
budgets and policy. These are not static algorithm properties. Artifact ids are
opaque identifiers from a deployment-verified allowlist, never filesystem paths.

Python callers can use `case_capabilities(id, case, variant=..., config=...)` with
their resolved config. Frequency/calibration presence is not a physical
compatibility certificate: axis, unit, Fourier, source and operator checks still
apply during execution. Validation observations are not mislabeled independent
test data. No MATLAB process is started by discovery.

## Compatibility and limits

Legacy YAML stopping behavior is preserved; new Agent admission uses RunControls
with no default update tolerance. Budgets cap work but do not promise equal FLOPs
or equal reconstruction quality across models. External deadlines remain the
supervisor's responsibility. Do not send online budget requests to variants that
declare them unsupported, or reinterpret imported results as budget-controlled
reconstructions.

Schema generation prunes hidden properties and unreachable `$defs`. Nested
object-valued Agent parameters currently fail schema generation until an explicit
nested exposure policy is defined; the library never silently exports an arbitrary
runtime object. Consumers should version-check schemas and reject unsupported
versions rather than reconstructing a second parameter table.
