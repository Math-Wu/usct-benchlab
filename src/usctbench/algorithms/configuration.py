"""One validation boundary for direct calls, YAML and registered execution.

The transport remains AlgorithmConfig for compatibility. The parameter object
has no run budget or runtime data arrays. Legacy keys are checked separately,
then supplied to the existing kernels without changing their stopping policy.
"""

from __future__ import annotations

from functools import wraps

import numpy as np
from pydantic import Field

from usctbench.algorithms.parameters import (
    BentParameters,
    BornParameters,
    CGLSParameters,
    FixedBornParameters,
    Nonnegative,
    NonnegativeInt,
    Parameters,
    Positive,
    SARTParameters,
    SIRTParameters,
    TinyFWIParameters,
    parameter,
)
from usctbench.core.config import coerce_bool, expand_config_value
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, ResultStatus
from usctbench.core.stopping import StopPolicy

PARAMETER_MODELS = {
    "straight_cgls": CGLSParameters,
    "straight_sirt": SIRTParameters,
    "straight_sart": SARTParameters,
    "bent_ray_gn": BentParameters,
    "rwave_adapter": BornParameters,
    "fwi_tiny": TinyFWIParameters,
}

# These are existing kernel defaults, not calibrated runtime presets.
DEFAULT_ITERATIONS = {
    "straight_cgls": 30,
    "straight_sirt": 50,
    "straight_sart": 10,
    "bent_ray_gn": 4,
    "fwi_tiny": 20,
}


def default_iterations(name, parameters=None):
    p = parameters or {}
    if name == "rwave_adapter":
        return 30 if p.get("mode") == "fixed_background" else 4
    if name == "fwi_kwave_adapter":
        return 5 if coerce_bool(p.get("controlled_operator", False)) else None
    return DEFAULT_ITERATIONS.get(name)


def parameter_model(name, parameters=None):
    from usctbench.algorithms.fwi.parameters import (
        ControlledFWIParameters,
        ExternalFWIParameters,
    )

    if name == "fwi_kwave_adapter":
        return (
            ControlledFWIParameters
            if coerce_bool((parameters or {}).get("controlled_operator", False))
            else ExternalFWIParameters
        )
    if name == "rwave_adapter" and (parameters or {}).get("mode") == "fixed_background":
        return FixedBornParameters
    if name not in PARAMETER_MODELS:
        raise ValueError(f"{name}: no approved typed parameter interface")
    return PARAMETER_MODELS[name]


class EvaluationParameters(Parameters):
    receiver_indices: list[NonnegativeInt] | None = parameter(
        None, "Explicit held-out receivers."
    )
    frequency_indices: list[NonnegativeInt] | None = parameter(
        None, "Explicit held-out frequencies."
    )
    receiver_fraction: float = parameter(0.0, "Held-out receiver fraction.", ge=0, lt=1)
    frequency_fraction: float = parameter(
        0.0, "Held-out frequency fraction.", ge=0, lt=1
    )
    seed: NonnegativeInt = parameter(0, "Split seed.")
    exclude_reciprocal: bool = parameter(
        True, "Exclude reciprocal partners from fitting."
    )


class ImageEvaluationParameters(Parameters):
    primary_region: str = parameter(
        "tissue", "Post-hoc primary image region.", pattern="^(tissue|full_image)$"
    )
    water_speed_mps: Positive = parameter(1500.0, "Post-hoc water reference.", "m/s")
    water_tolerance_mps: Nonnegative = parameter(
        0.1, "Post-hoc water classification tolerance.", "m/s"
    )
    data_range_mps: Positive | None = parameter(
        None, "Explicit PSNR/SSIM data range; null uses GT range.", "m/s"
    )


class LegacyIterations(Parameters):
    iterations: NonnegativeInt | None = Field(None)
    outer_iterations: NonnegativeInt | None = Field(None)
    steps: NonnegativeInt | None = Field(None)


def validate_algorithm_config(name, config: AlgorithmConfig) -> AlgorithmConfig:
    """Raise on invalid input; return an independent, resolved compatibility object."""
    from usctbench.algorithms.fwi.parameters import (
        ControlledFWIParameters,
        ExternalFWIParameters,
    )

    if config.name is not None and config.name != name:
        raise ValueError(f"algorithm/config mismatch: {name!r} != {config.name!r}")
    values = expand_config_value(dict(config.parameters))
    model = parameter_model(name, values)
    # Canonicalize the whole-solve alias before any legacy or typed budget checks.
    if model is FixedBornParameters and "inner_iterations" in values:
        legacy = values.pop("inner_iterations")
        LegacyIterations(iterations=legacy)
        if "iterations" in values and values["iterations"] != legacy:
            raise ValueError("conflicting fixed-background iteration budgets")
        values["iterations"] = legacy
    if name == "fwi_tiny" and {"stopping", "evaluation"}.intersection(values):
        raise ValueError(
            "fwi_tiny does not support stopping or fitting/split evaluation; use steps"
        )
    auxiliary = {}
    budget_keys = {
        key: values.pop(key)
        for key in ("iterations", "outer_iterations", "steps")
        if key in values
    }
    LegacyIterations.model_validate(budget_keys)
    valid_budget_keys = {"iterations"}
    if (
        name in {"bent_ray_gn", "rwave_adapter"} and model is not FixedBornParameters
    ) or model is ControlledFWIParameters:
        valid_budget_keys.add("outer_iterations")
    if name == "fwi_tiny":
        valid_budget_keys = {"steps"}
    if model is ExternalFWIParameters:
        valid_budget_keys = set()
    if budget_keys.keys() - valid_budget_keys:
        raise ValueError(
            f"unused iteration controls for {name}: {sorted(budget_keys.keys() - valid_budget_keys)}"
        )
    if len(set(budget_keys.values())) > 1:
        raise ValueError("conflicting legacy iteration budgets")
    if (
        config.run_controls is not None
        and config.run_controls.max_iterations is not None
        and budget_keys
    ):
        if config.run_controls.max_iterations != next(iter(budget_keys.values())):
            raise ValueError("conflicting run_controls and legacy iteration budget")
    auxiliary.update(budget_keys)
    if "stopping" in values:
        stopping = values.pop("stopping")
        StopPolicy.from_parameters(
            {"stopping": stopping},
            default_iterations=default_iterations(name, values) or 50,
        )
        if (
            budget_keys
            and "max_iterations" in stopping
            and stopping["max_iterations"] != next(iter(budget_keys.values()))
        ):
            raise ValueError("conflicting stopping and legacy iteration budget")
        auxiliary["stopping"] = dict(stopping)
    for key, auxiliary_model in (
        ("evaluation", EvaluationParameters),
        ("image_evaluation", ImageEvaluationParameters),
    ):
        if key in values:
            auxiliary[key] = auxiliary_model.model_validate(values.pop(key)).model_dump(
                exclude_unset=True
            )
    if "_run_output_dir" in values:
        path = values.pop("_run_output_dir")
        if not isinstance(path, str):
            raise ValueError("_run_output_dir must be a deployment-owned string")
        auxiliary["_run_output_dir"] = path
    for key, allowed in (
        ("source_spectrum", name == "rwave_adapter"),
        ("discrete_source_spectrum", model is ControlledFWIParameters),
    ):
        if key in values and allowed:
            array = np.asarray(values.pop(key), dtype=complex)
            if array.ndim not in (0, 1, 2) or not np.isfinite(array).all():
                raise ValueError(
                    f"{key} must be finite scalar/frequency/frequency-source calibration"
                )
            auxiliary[key] = array
    for key in ("initial_sound_speed_mps", "background_sound_speed_mps"):
        if isinstance(values.get(key), np.ndarray):
            values[key] = values[key].tolist()
    typed = model.model_validate(values)
    backend = typed.backend_parameters()
    if model is ExternalFWIParameters and (
        config.run_controls is not None or config.budget_caps is not None
    ):
        raise ValueError(
            "external FWI cannot enforce run_controls/budget_caps; use controlled_operator or a deployment-level supervisor"
        )
    if name == "fwi_tiny" and (
        config.run_controls is not None or config.budget_caps is not None
    ):
        raise ValueError(
            "fwi_tiny only supports its legacy steps budget, not online RunControls"
        )
    if default_iterations(name, backend) is not None:
        key = "steps" if name == "fwi_tiny" else "iterations"
        count = auxiliary.get("outer_iterations", default_iterations(name, backend))
        count = auxiliary.get("stopping", {}).get("max_iterations", count)
        if (
            config.run_controls is not None
            and config.run_controls.max_iterations is not None
        ):
            count = config.run_controls.max_iterations
        auxiliary.setdefault(key, count)
    return config.model_copy(
        update={"name": name, "parameters": {**backend, **auxiliary}}
    )


def validated_run(function):
    """Preserve the algorithm result/failure API while validating direct calls."""

    @wraps(function)
    def run(self, case, config):
        try:
            resolved = validate_algorithm_config(self.name, config)
        except (ValueError, TypeError) as exc:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=f"invalid configuration: {exc}",
            )
        return function(self, case, resolved)

    return run
