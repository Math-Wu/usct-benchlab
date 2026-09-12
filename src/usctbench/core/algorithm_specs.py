"""Static, versioned research algorithm descriptions and Agent admission.

Registry consumers use explicit physical variants, never naming heuristics.
Static schemas contain no case data or deployment-specific paths.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import re

from pydantic import ConfigDict, create_model

from usctbench.algorithms.configuration import (
    default_iterations,
    parameter_model,
    validate_algorithm_config,
)
from usctbench.core.run_controls import BudgetCaps, RunControls
from usctbench.core.schema import AlgorithmConfig
from usctbench.core.stopping import StopPolicy

BUDGET_FIELDS = (
    "max_iterations",
    "max_elapsed_s",
    "max_forward_calls",
    "max_adjoint_calls",
)
AgentRunControls = create_model(
    "AgentRunControls",
    __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    **{
        name: (
            RunControls.model_fields[name].annotation,
            RunControls.model_fields[name],
        )
        for name in BUDGET_FIELDS
    },
)


@dataclass(frozen=True)
class Variant:
    id: str
    model: str
    selectors: dict = field(default_factory=dict)
    observation_domains: tuple[str, ...] = ("features",)
    required_observations: tuple[str, ...] = ("delta_tof_s",)
    observation_alternatives: tuple[tuple[str, ...], ...] = ()
    runtime_requirements: tuple[str, ...] = ("numpy", "scipy")
    limitations: tuple[str, ...] = ()
    iteration_unit: str = "iteration"
    deterministic: str = (
        "deterministic_for_fixed_input_config_and_numerical_environment"
    )
    online_controls: bool = True
    agent_parameters: bool = True


@dataclass(frozen=True)
class AlgorithmSpecification:
    algorithm_id: str
    family: str
    description: str
    variants: tuple[Variant, ...]
    supported_geometries: tuple[str, ...] = ("ring", "linear", "custom")

    def variant(self, variant_id=None):
        if variant_id is None:
            return self.variants[0]
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise ValueError(f"unknown variant {variant_id!r} for {self.algorithm_id}")

    def describe(self, variant_id=None):
        v = self.variant(variant_id)
        model = parameter_model(self.algorithm_id, v.selectors)
        schema, defaults = agent_schema(model, v.selectors, enabled=v.agent_parameters)
        description = {
            "schema_version": "usct.algorithm.v1",
            "algorithm_id": self.algorithm_id,
            "family": self.family,
            "variant": v.id,
            "variants": [
                {"id": item.id, "model": item.model} for item in self.variants
            ],
            "description": self.description,
            "mathematical_model": v.model,
            "required_observation_domains": list(v.observation_domains),
            "required_observations": {
                "all_of": list(v.required_observations),
                "any_of": [list(group) for group in v.observation_alternatives],
            },
            "supported_geometries": list(self.supported_geometries),
            "parameter_model": model.__name__,
            "config_schema": schema,
            "allowed_parameters": sorted(schema["properties"]),
            "default_parameters": defaults,
            "compute_budget": {
                "separate_from_parameters": True,
                "default_max_iterations": default_iterations(
                    self.algorithm_id, v.selectors
                ),
                "online_controls_supported": v.online_controls,
                "run_controls_schema": (
                    AgentRunControls.model_json_schema() if v.online_controls else None
                ),
                "budget_caps_schema": (
                    BudgetCaps.model_json_schema() if v.online_controls else None
                ),
                "deadline_semantics": (
                    "cooperative; launching process must enforce a hard deadline"
                    if v.online_controls
                    else "external supervisor required; online library controls unavailable"
                ),
            },
            "runtime_requirements": list(v.runtime_requirements),
            "runtime_availability": "not_checked_static_description",
            "limitations": [
                "Research-grade within tested models/ranges; not clinical validation.",
                "Geometry acceptance does not establish imaging performance for every layout.",
                *v.limitations,
            ],
            "iteration_unit": v.iteration_unit,
            "deterministic_status": v.deterministic,
        }
        if self.algorithm_id == "fwi_wust":
            for key in ("run_controls_schema", "budget_caps_schema"):
                properties = description["compute_budget"][key]["properties"]
                for unsupported in ("max_forward_calls", "max_adjoint_calls"):
                    properties.pop(unsupported, None)
            description["compute_budget"][
                "deadline_semantics"
            ] = "hard global deadline including admission, ingestion, reconstruction and parsing"
            description["compute_budget"][
                "numerical_convergence_stopping_supported"
            ] = False
            controls = description["compute_budget"]["run_controls_schema"][
                "properties"
            ]
            controls["max_iterations"][
                "description"
            ] = "Truncate the resolved frequency schedule; null runs the full schedule, zero returns validated initialization."
            controls["max_elapsed_s"][
                "description"
            ] = "One hard global deadline in seconds; an explicit request or deployment timeout cap is required."
        return description


def agent_schema(model, selectors=None, *, enabled=True):
    """Filter properties AND reachable definitions, defaults and permissions.

    No internal field survives through a $defs side channel. Object-valued Agent
    parameters require an explicit nested policy rather than allowing arbitrary
    dictionaries or external runtime command maps.
    """
    schema = deepcopy(model.model_json_schema())
    defaults = model.model_validate(selectors or {}).model_dump(mode="json")
    properties = {
        key: value
        for key, value in schema["properties"].items()
        if enabled and value.get("exposure") == "agent"
    }
    definitions = schema.get("$defs", {})
    reachable = {}

    def inspect(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                raise ValueError(
                    "nested Agent object requires an explicit exposure policy"
                )
            if "$ref" in node:
                name = node["$ref"].removeprefix("#/$defs/")
                if name not in reachable:
                    reachable[name] = definitions[name]
                    inspect(definitions[name])
            for child in node.values():
                inspect(child)
        elif isinstance(node, list):
            for child in node:
                inspect(child)

    for key, value in properties.items():
        inspect(value)
        value.update(getattr(model, "agent_schema_overrides", {}).get(key, {}))
        value["default"] = defaults[key]
        if key in (selectors or {}):
            value["const"] = selectors[key]
    result = {"type": "object", "additionalProperties": False, "properties": properties}
    if reachable:
        result["$defs"] = reachable
    required = [key for key in schema.get("required", []) if key in properties]
    if required:
        result["required"] = required
    return result, {key: defaults[key] for key in properties}


def _born_variant(identifier, mode, backend):
    return Variant(
        identifier,
        (
            "distorted_born_volume_integral"
            if backend == "volume_integral"
            else "eikonal_wkb_ray_born"
        ),
        selectors={"mode": mode, "green_backend": backend},
        observation_domains=("frequency",),
        required_observations=("freq_data", "frequencies_hz"),
        observation_alternatives=(("source_spectrum",), ("water_reference",)),
        iteration_unit=(
            "Ray-Born outer step" if mode == "nonlinear" else "Ray-Born CGLS step"
        ),
        limitations=(
            "Not a reproduction of the upstream r-Wave MATLAB package.",
            "Requires calibrated complex pressure, Fourier convention and adequate PPW.",
            (
                "WKB sensitivity approximates continuous physics, not the exact discrete WKB derivative."
                if backend == "eikonal_wkb"
                else "Full-Green accuracy depends on the volume-integral solve tolerance and grid."
            ),
            (
                "Fixed-background Born linearization only."
                if mode == "fixed_background"
                else "Background Green fields are updated during inversion."
            ),
        ),
    )


SPECS = {
    "straight_cgls": AlgorithmSpecification(
        "straight_cgls",
        "straight_ray",
        "Bounded weighted straight-ray least squares with optional regularization/IRLS.",
        (
            Variant(
                "cgls",
                "A delta_slowness = delta_tof",
                iteration_unit="cgls_step",
                limitations=("Assumes straight travel paths; not a waveform model.",),
            ),
        ),
    ),
    "straight_sirt": AlgorithmSpecification(
        "straight_sirt",
        "straight_ray",
        "Simultaneous algebraic travel-time reconstruction.",
        (
            Variant(
                "sirt",
                "A delta_slowness = delta_tof",
                iteration_unit="full_sweep",
                limitations=(
                    "Smoothing is an iterative filter, not an explicit lambda penalty.",
                ),
            ),
        ),
    ),
    "straight_sart": AlgorithmSpecification(
        "straight_sart",
        "straight_ray",
        "Ordered-subset algebraic travel-time reconstruction.",
        (
            Variant(
                "sart",
                "A delta_slowness = delta_tof",
                iteration_unit="full_sweep",
                limitations=(
                    "One iteration includes all subsets; do not compare with one subset update.",
                ),
            ),
        ),
    ),
    "bent_ray_gn": AlgorithmSpecification(
        "bent_ray_gn",
        "eikonal",
        "Native fast-marching travel-time Gauss-Newton reconstruction.",
        (
            Variant(
                "eikonal_gn",
                "|grad T| = slowness",
                required_observations=(),
                observation_alternatives=(("delta_tof_s",), ("tof_s",)),
                iteration_unit="Gauss-Newton outer step",
                limitations=(
                    "First-arrival geometrical acoustics; no pressure/diffraction modeling.",
                    "Eikonal differentiation is local to the active numerical stencil.",
                ),
            ),
        ),
    ),
    "rwave_adapter": AlgorithmSpecification(
        "rwave_adapter",
        "ray_born",
        "Native complex-pressure Born variants; legacy registered name retained.",
        tuple(
            _born_variant(identifier, mode, backend)
            for identifier, mode, backend in (
                ("wkb_nonlinear", "nonlinear", "eikonal_wkb"),
                ("full_green_nonlinear", "nonlinear", "volume_integral"),
                ("wkb_fixed", "fixed_background", "eikonal_wkb"),
                ("full_green_fixed", "fixed_background", "volume_integral"),
            )
        ),
    ),
    "fwi_wust": AlgorithmSpecification(
        "fwi_wust",
        "full_wave",
        "2-D frequency-domain sound-speed FWI through the pinned WUST CUDA runtime; no attenuation reconstruction.",
        (
            Variant(
                "wust",
                "source_projected_Helmholtz_frequency_continuation_NCG",
                observation_domains=("frequency",),
                required_observations=(
                    "freq_data",
                    "frequencies_hz",
                    "valid_mask",
                    "declared_pressure_convention",
                ),
                runtime_requirements=(
                    "MATLAB",
                    "approved_pinned_WUST_CUDA_Block_LU",
                ),
                limitations=(
                    "GPU is production; CPU is reference/debug only. No CPU fallback.",
                    "Schedule completion is not convergence; no final-model residual evaluation.",
                    "Only iteration and global elapsed-time budgets are supported.",
                    "Scalar/reference initialization available to Agent; map is expert-only.",
                ),
                iteration_unit="frequency_schedule_update",
                deterministic="fixed_runtime_and_linear_solver_environment",
            ),
        ),
    ),
}


def make_agent_config(
    algorithm_id,
    parameters=None,
    *,
    variant=None,
    run_controls=None,
    budget_caps=None,
    trusted_parameters=None,
):
    """Backend-enforced Agent boundary; trusted_parameters are deployment-owned.

    Never fill trusted_parameters from Agent output. They supply approved runtime
    artifacts/executables separately from autonomous numerical choices.
    """
    spec = SPECS[algorithm_id]
    selected = spec.variant(variant)
    description = spec.describe(selected.id)
    supplied = dict(parameters or {})
    if algorithm_id == "fwi_wust" and supplied.get("initialization") == "map":
        raise ValueError(
            "Map initialization is expert-only; no Agent artifact resolver exists yet"
        )
    forbidden = supplied.keys() - set(description["allowed_parameters"])
    if forbidden:
        raise ValueError(f"parameters not exposed to Agent: {sorted(forbidden)}")
    model = parameter_model(algorithm_id, selected.selectors)
    agent_model = create_model(
        f"{model.__name__}AgentInput",
        __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
        **{
            key: (model.model_fields[key].annotation, model.model_fields[key])
            for key in description["allowed_parameters"]
        },
    )
    # Validate JSON types/enums without expert-only alias normalization. Cross-field
    # physical checks still run in the full model at the common boundary below.
    supplied = agent_model.model_validate_json(
        json.dumps(supplied, allow_nan=False), strict=True
    ).model_dump(exclude_unset=True)
    values = {**(trusted_parameters or {}), **supplied}
    for key, value in selected.selectors.items():
        if key in values and values[key] != value:
            raise ValueError(f"parameter {key} conflicts with selected variant")
        values[key] = value
    if not selected.online_controls and (
        run_controls is not None or budget_caps is not None
    ):
        raise ValueError("selected variant does not enforce online run controls")
    return validate_algorithm_config(
        algorithm_id,
        AlgorithmConfig(
            name=algorithm_id,
            parameters=values,
            run_controls=(
                RunControls.model_validate(
                    AgentRunControls.model_validate(run_controls or {}).model_dump()
                )
                if selected.online_controls
                else None
            ),
            budget_caps=budget_caps,
        ),
    )


def case_capabilities(
    algorithm_id,
    case,
    *,
    variant=None,
    config=None,
    runtime_available=None,
    initialization_artifact_ids=(),
):
    """Case facts, not a clinical/physical validity certificate or static schema."""
    spec = SPECS[algorithm_id]
    selected = spec.variant(variant)
    resolved = validate_algorithm_config(
        algorithm_id,
        config or AlgorithmConfig(name=algorithm_id, parameters=selected.selectors),
    )
    for key, value in selected.selectors.items():
        if resolved.parameters.get(key) != value:
            raise ValueError("case-bound config does not match requested variant")
    ids = list(initialization_artifact_ids)
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in ids):
        raise ValueError(
            "initialization artifacts must be approved opaque ids, not paths"
        )
    measurement = case.measurement
    frequencies = measurement.frequencies_hz
    controls = None
    policy = None
    if algorithm_id == "fwi_wust":
        schedule = resolved.parameters.get("frequency_schedule_hz")
        count = len(
            ([] if frequencies is None else frequencies)
            if schedule is None
            else schedule
        )
        policy = (
            (resolved.run_controls or RunControls())
            .capped(resolved.budget_caps, default_iterations=count)
            .model_dump()
        )
        policy["max_iterations"] = min(count, policy["max_iterations"])
        controls = {key: policy[key] for key in BUDGET_FIELDS}
    elif selected.online_controls:
        policy = (
            (resolved.run_controls or RunControls()).to_stop_policy(
                default_iterations=resolved.parameters["iterations"],
                budget=resolved.budget_caps,
            )
            if resolved.run_controls is not None or resolved.budget_caps is not None
            else StopPolicy.from_parameters(resolved.parameters)
        )
        policy = asdict(policy)
        controls = {key: policy[key] for key in BUDGET_FIELDS}
    return {
        "schema_version": "usct.case_capabilities.v1",
        "algorithm_id": algorithm_id,
        "variant": selected.id,
        "case_id": case.case_id,
        "available_frequencies_hz": [] if frequencies is None else frequencies.tolist(),
        "calibration_availability": {
            "source_spectrum_present": measurement.source_spectrum is not None,
            "water_reference_present": measurement.water_reference is not None,
            "discrete_source_spectrum_present": "discrete_source_spectrum"
            in resolved.parameters,
            "meaning": "presence only; operator-specific units/convention checks still required",
        },
        "runtime_available": runtime_available,
        "production_runtime_available": (
            None if algorithm_id == "fwi_wust" else runtime_available
        ),
        "runtime_availability_source": (
            "not_checked" if runtime_available is None else "deployment_reported"
        ),
        "compatible_initialization_artifact_ids": ids,
        "initialization_compatibility_source": "deployment_verified_allowlist",
        "resolved_budgets": controls,
        "resolved_run_policy": policy,
        "validation_is_independent_test": False,
    }
