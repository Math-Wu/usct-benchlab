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
usct describe-algorithm fwi_wust --json
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

`fwi_wust` has one production variant, family `full_wave`. It requires existing total complex pressure, explicit axes/Fourier semantics and validity mask. WUST eliminates complex source scale per TX/frequency; it does not require Born source calibration. Runtime root, MATLAB executable, CUDA device and CPU reference selection are never Agent parameters. Deployment must separately verify GPU production availability.

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

Native legacy YAML stopping behavior remains available. New Agent admission uses RunControls with no default update tolerance. WUST supports only schedule truncation and a shared hard elapsed-time deadline, not numerical convergence. CPU availability must not be used as a production certificate.

Schema generation prunes hidden properties and unreachable `$defs`. Nested
object-valued Agent parameters currently fail schema generation until an explicit
nested exposure policy is defined; the library never silently exports an arbitrary
runtime object. Consumers should version-check schemas and reject unsupported
versions rather than reconstructing a second parameter table.

## Reconstruction scope

The canonical reconstruction target is 2-D sound speed. Attenuation is not an
estimated quantity or Agent capability. Production WUST performs reconstruction through its versioned runtime rather than importing historical images.
