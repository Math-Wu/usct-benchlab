"""Typed admission, legacy migrations and unchanged numerical defaults."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from usctbench.algorithms.configuration import (
    parameter_model,
    validate_algorithm_config,
)
from usctbench.algorithms.parameters import CGLSParameters, SIRTParameters
from usctbench.algorithms.fwi.parameters import ExternalFWIParameters
from usctbench.benchmark.runner import load_algorithm_config
from usctbench.cli import register_builtin_algorithms
from usctbench.core.registry import get_algorithm
from usctbench.core.schema import AlgorithmConfig
from usctbench.data.synthetic import make_sound_speed_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name",
    [
        "straight_cgls",
        "straight_sirt",
        "straight_sart",
        "bent_ray_gn",
        "rwave_adapter",
        "attenuation_sirt",
        "fwi_kwave_adapter",
        "fwi_tiny",
    ],
)
def test_models_have_schema_metadata_and_reject_unknown(name):
    model = parameter_model(name)
    schema = model.model_json_schema()
    for field in schema["properties"].values():
        assert field["description"]
        assert field["exposure"] in {"agent", "advanced", "internal"}
        assert "units" in field
    with pytest.raises(ValidationError):
        model.model_validate({"typo": 1})


@pytest.mark.parametrize(
    "values",
    [
        {"lambda": 0.2, "regularization_lambda": 0.3},
        {"damping": 0.04, "regularization_lambda": 0.3},
        {"damping": -1},
        {"damping": True},
        {"damping": float("nan")},
        {"regularization_lambda": -1},
        {"regularization_lambda": float("inf")},
        {"sound_speed_bounds_mps": [1500, 1400]},
        {"ray_weight_min": 0.1, "min_ray_weight": 0.2},
        {"inner_iterations": 10},
    ],
)
def test_invalid_and_conflicting_cgls_parameters(values):
    with pytest.raises(ValueError):
        CGLSParameters.model_validate(values)


def test_alias_normalization_and_unused_parameters():
    model = CGLSParameters.model_validate(
        {
            "damping": 0.04,
            "lambda": 0.2,
            "regularization": "roughness",
            "robust_loss": "irls",
        }
    )
    assert model.regularization_lambda == pytest.approx(0.2)
    assert model.regularization == "laplacian"
    assert model.robust_loss == "huber"
    assert {"lambda", "damping"}.isdisjoint(model.model_dump())
    for field in ("regularization_lambda", "subsets", "damping"):
        with pytest.raises(ValueError):
            SIRTParameters.model_validate({field: 0.1})


def test_nested_fields_fail_closed():
    for parameters in (
        {"inner_options": {"atol": 1e-8, "matlab_bin": "bad"}},
        {"evaluation": {"reciever_fraction": 0.1}},
        {"stopping": {"unknown": 1}},
    ):
        with pytest.raises(ValueError):
            validate_algorithm_config(
                "bent_ray_gn", AlgorithmConfig(parameters=parameters)
            )


def test_repository_configs_and_idempotent_resolution():
    for path in sorted((ROOT / "configs/algorithms").glob("*.yaml")):
        if path.name == "diffusion_fwi_kwave.yaml":
            # Existing optional integration is explicitly outside this typed API.
            continue
        config = load_algorithm_config(path)
        again = validate_algorithm_config(config.name, config)
        assert again.model_dump() == config.model_dump(), path


def test_yaml_python_and_typed_object_share_validation(tmp_path):
    parameters = {"lambda": 0.02, "regularization": "laplacian"}
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "straight_cgls",
                "parameters": parameters,
                "run_controls": {"max_iterations": 2},
                "budget_caps": {"max_iterations": 1},
            }
        )
    )
    loaded = load_algorithm_config(path)
    direct = validate_algorithm_config(
        "straight_cgls",
        AlgorithmConfig(
            name="straight_cgls",
            parameters=CGLSParameters.model_validate(parameters),
            run_controls={"max_iterations": 2},
            budget_caps={"max_iterations": 1},
        ),
    )
    assert loaded.model_dump() == direct.model_dump()
    assert loaded.run_controls.max_iterations == 2
    assert loaded.budget_caps.max_iterations == 1
    path.write_text(
        yaml.safe_dump({"name": "straight_cgls", "parameters": {"typo": 1}})
    )
    with pytest.raises(ValueError):
        load_algorithm_config(path)


@pytest.mark.parametrize(
    "name", ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"]
)
def test_typed_boundary_preserves_default_reconstruction(name):
    register_builtin_algorithms()
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8, inclusion_mps=1490)
    algorithm = get_algorithm(name)
    config = AlgorithmConfig(
        parameters={
            "iterations": 2,
            "stopping": {"update_rtol": None, "objective_rtol": None},
        }
    )
    before = algorithm.run.__wrapped__(algorithm, case, config)
    after = algorithm.run(case, config)
    assert before.status == after.status == "success", after.failure_reason
    np.testing.assert_array_equal(after.sound_speed_mps, before.sound_speed_mps)
    np.testing.assert_array_equal(
        after.metrics["residual_curve"], before.metrics["residual_curve"]
    )
    assert after.metrics["stopping"]["policy"] == before.metrics["stopping"]["policy"]
    bad = algorithm.run(case, AlgorithmConfig(parameters={"unknown": 1}))
    assert bad.status == "failed" and "invalid configuration" in bad.failure_reason


def test_fwi_aliases_delegate_unspecified_runtime_defaults():
    p = ExternalFWIParameters.model_validate(
        {"c_init": 1500, "velocity_bounds": [1400, 1600], "sos_freqs_mhz": [0.3, 0.4]}
    )
    assert p.initial_sound_speed_mps == 1500
    assert p.sos_frequencies_hz == [300000, 400000]
    assert p.max_update_mps is None
    assert "max_update_mps" not in p.backend_parameters()
    assert p.backend_parameters()["sos_freqs_mhz"] == pytest.approx([0.3, 0.4])
    with pytest.raises(ValueError, match="conflicting"):
        ExternalFWIParameters.model_validate(
            {"c_init": 1500, "initial_sound_speed_mps": 1400}
        )
    with pytest.raises(ValueError, match="cannot enforce"):
        validate_algorithm_config(
            "fwi_kwave_adapter", AlgorithmConfig(run_controls={"max_iterations": 2})
        )


def test_fixed_born_budget_is_not_a_nonlinear_inner_parameter():
    config = validate_algorithm_config(
        "rwave_adapter",
        AlgorithmConfig(parameters={"mode": "fixed_background", "inner_iterations": 7}),
    )
    assert config.parameters["iterations"] == 7
    assert "inner_iterations" not in config.parameters
    assert config.parameters["regularization"] == "identity"
    with pytest.raises(ValueError):
        validate_algorithm_config(
            "rwave_adapter",
            AlgorithmConfig(
                parameters={"mode": "fixed_background", "max_update_mps": 3}
            ),
        )
