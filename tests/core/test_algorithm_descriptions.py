"""Agent schemas are executable permissions, not just a filtered help page."""

import json

import numpy as np
import pytest
from pydantic import create_model

from usctbench.algorithms.parameters import InnerOptions, Parameters, parameter
from usctbench.cli import main, register_builtin_algorithms
from usctbench.core.algorithm_specs import (
    SPECS,
    agent_schema,
    case_capabilities,
    make_agent_config,
)
from usctbench.core.io import write_case_hdf5
from usctbench.core.registry import get_algorithm, get_algorithm_entry
from usctbench.core.schema import AlgorithmConfig
from usctbench.data.synthetic import make_sound_speed_case


@pytest.mark.parametrize("algorithm_id", sorted(SPECS))
def test_schema_defaults_permissions_and_variant_identity(algorithm_id):
    register_builtin_algorithms()
    entry = get_algorithm_entry(algorithm_id)
    for variant in entry.specification.variants:
        description = entry.describe(variant.id)
        schema = description["config_schema"]
        defaults = description["default_parameters"]
        assert (
            set(description["allowed_parameters"])
            == set(defaults)
            == set(schema["properties"])
        )
        assert schema["additionalProperties"] is False
        assert description["algorithm_id"] == algorithm_id
        assert description["family"] == SPECS[algorithm_id].family
        for name, value in schema["properties"].items():
            assert value["exposure"] == "agent"
            assert defaults[name] == value["default"]
        payload = json.dumps({"schema": schema, "defaults": defaults}, allow_nan=False)
        for hidden in (
            "matlab_bin",
            "pipeline_module",
            "pipeline_args",
            "scratch_dir",
            "functions_path",
            "inner_options",
            "regularization_lambda",
            "gradient_rtol",
            "timeout_s",
        ):
            assert hidden not in payload
        config = make_agent_config(algorithm_id, defaults, variant=variant.id)
        assert config.name == algorithm_id
        assert "iterations" not in description["allowed_parameters"]
        assert "update_rtol" not in json.dumps(description["compute_budget"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("regularization_lambda", 0.1),
        ("lambda", 0.1),
        ("damping", 0.1),
        ("inner_options", {"atol": 0.1}),
        ("iterations", 10),
        ("_run_output_dir", "/tmp/not-allowed"),
        ("matlab_bin", "anything"),
        ("pipeline_args", ["--something"]),
        ("source_spectrum", [1]),
    ],
)
def test_agent_cannot_submit_hidden_or_unknown_controls(field, value):
    with pytest.raises(ValueError, match="not exposed"):
        make_agent_config("straight_cgls", {field: value})


def test_agent_types_enums_cross_checks_and_budget_boundary():
    for values in (
        {"regularization": "roughness"},
        {"robust_loss": "irls"},
        {"sound_speed_bounds_mps": [1600, 1400]},
    ):
        with pytest.raises(ValueError):
            make_agent_config("straight_cgls", values)
    with pytest.raises(ValueError):
        make_agent_config("straight_sirt", {"relaxation": "0.3"})
    with pytest.raises(ValueError):
        make_agent_config("straight_cgls", run_controls={"update_rtol": 0.1})
    config = make_agent_config(
        "straight_cgls",
        run_controls={"max_iterations": 3},
        budget_caps={"max_iterations": 1},
    )
    register_builtin_algorithms()
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    result = get_algorithm("straight_cgls").run(case, config)
    assert result.status == "success", result.failure_reason
    assert result.metrics["iterations"] <= 1
    assert result.metrics["stopping"]["policy"]["update_rtol"] is None


def test_definitions_do_not_leak_and_nested_objects_fail_closed():
    hidden = create_model(
        "HiddenNested",
        __base__=Parameters,
        runtime=(
            InnerOptions,
            parameter(InnerOptions(), "hidden", exposure="internal"),
        ),
    )
    schema, defaults = agent_schema(hidden)
    assert "$defs" not in schema
    assert defaults == {}
    exposed = create_model(
        "UnsafeNested",
        __base__=Parameters,
        runtime=(InnerOptions, parameter(InnerOptions(), "unsafe", exposure="agent")),
    )
    with pytest.raises(ValueError, match="nested Agent object"):
        agent_schema(exposed)


def test_observation_and_runtime_variants_are_not_guessed_from_names():
    born = SPECS["rwave_adapter"].describe("full_green_nonlinear")
    assert born["family"] == "ray_born"
    assert born["required_observation_domains"] == ["frequency"]
    assert "freq_data" in born["required_observations"]["all_of"]
    assert born["mathematical_model"] == "distorted_born_volume_integral"
    fixed = SPECS["rwave_adapter"].describe("wkb_fixed")
    assert "max_update_mps" not in fixed["allowed_parameters"]
    assert fixed["compute_budget"]["default_max_iterations"] == 30
    imported = SPECS["fwi_kwave_adapter"].describe()
    controlled = SPECS["fwi_kwave_adapter"].describe("controlled")
    assert imported["allowed_parameters"] == []
    assert imported["compute_budget"]["online_controls_supported"] is False
    assert controlled["compute_budget"]["online_controls_supported"] is True
    with pytest.raises(ValueError, match="conflicts"):
        make_agent_config(
            "rwave_adapter",
            {"green_backend": "volume_integral"},
            variant="wkb_nonlinear",
        )
    with pytest.raises(ValueError, match="does not enforce"):
        make_agent_config(
            "fwi_kwave_adapter",
            variant="external_pipeline",
            run_controls={"max_iterations": 1},
        )


def test_case_bound_facts_and_policy_are_separate():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    case.measurement.frequencies_hz = np.array([150000.0, 250000.0])
    static = SPECS["straight_cgls"].describe()
    config = AlgorithmConfig(
        parameters={
            "iterations": 2,
            "stopping": {"update_rtol": None, "min_iterations": 1},
        }
    )
    bound = case_capabilities(
        "straight_cgls",
        case,
        config=config,
        initialization_artifact_ids=["water-init-1"],
    )
    assert "available_frequencies_hz" not in static
    assert bound["available_frequencies_hz"] == [150000, 250000]
    assert bound["runtime_available"] is None
    assert bound["resolved_budgets"]["max_iterations"] == 2
    assert bound["resolved_run_policy"]["min_iterations"] == 1
    assert bound["compatible_initialization_artifact_ids"] == ["water-init-1"]
    with pytest.raises(ValueError, match="opaque ids"):
        case_capabilities(
            "straight_cgls", case, initialization_artifact_ids=["/secret/init.h5"]
        )


def test_json_stdout_cli_and_case_capabilities(capsys, tmp_path):
    assert main(["list-algorithms", "--json"]) == 0
    output = capsys.readouterr()
    entries = json.loads(output.out)
    assert output.err == ""
    assert len(entries) == 9
    legacy = next(
        row for row in entries if row["algorithm_id"] == "diffusion_fwi_kwave_adapter"
    )
    assert legacy["typed_interface_available"] is False
    path = tmp_path / "case.h5"
    write_case_hdf5(make_sound_speed_case(shape=(8, 8), n_transducers=8), path)
    assert (
        main(["describe-algorithm", "straight_cgls", "--json", "--case", str(path)])
        == 0
    )
    assert "case_capabilities" in json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        main(["describe-algorithm", "unknown", "--json"])
    output = capsys.readouterr()
    assert output.out == ""
    assert "unknown algorithm" in output.err
