"""Admission and stopping semantics, independent of image quality."""

import numpy as np
import pytest
from pydantic import ValidationError

from usctbench.core.run_controls import BudgetCaps, RunControls
from usctbench.core.schema import AlgorithmConfig
from usctbench.core.stopping import StopMonitor, WorkLedger


def test_minimal_policy_disables_optional_quality_rules():
    policy = RunControls().to_stop_policy(default_iterations=30)
    assert policy.max_iterations == 30
    assert policy.update_rtol is None
    assert policy.min_iterations == 3
    assert policy.update_patience == 2
    assert policy.objective_rtol is None
    assert policy.validation_patience is None
    assert policy.target_rmse_mps is None
    assert not policy.restore_best_validation
    assert not policy.allow_ground_truth_stopping


def test_caps_intersect_and_zero_is_not_missing():
    requested = RunControls(max_iterations=30, max_elapsed_s=2, max_forward_calls=10)
    actual = requested.capped(
        BudgetCaps(
            max_iterations=0, timeout_s=4, max_forward_calls=3, max_adjoint_calls=0
        )
    )
    assert actual.max_iterations == 0
    assert actual.max_elapsed_s == 2
    assert actual.max_forward_calls == 3
    assert actual.max_adjoint_calls == 0
    assert requested.max_iterations == 30


@pytest.mark.parametrize(
    "values",
    [
        {"max_iterations": True},
        {"max_iterations": 1.5},
        {"max_elapsed_s": float("nan")},
        {"max_elapsed_s": float("inf")},
        {"update_rtol": -1},
        {"update_patience": 0},
        {"unexpected": 2},
    ],
)
def test_invalid_controls_fail_closed(values):
    with pytest.raises(ValidationError):
        RunControls(**values)


def test_one_small_step_or_pre_minimum_steps_do_not_stop():
    policy = RunControls(
        min_iterations=2, update_patience=2, update_rtol=0.01
    ).to_stop_policy(default_iterations=10)
    monitor = StopMonitor(policy, WorkLedger(policy))
    updates = [None, 0.001, 0.001, 0.5, 0.001, 0.001]
    for k, update in enumerate(updates):
        reason = monitor.observe(
            k,
            np.ones(2),
            residual_norm=1,
            observed_norm=2,
            objective=0.5,
            update_relative=update,
        )
        assert reason == ("small_model_update" if k == 5 else None)
    assert monitor.record()["completed_iterations"] == 5


def test_controls_round_trip_and_no_ambiguous_legacy_stopping():
    config = AlgorithmConfig(run_controls=RunControls(max_iterations=7))
    assert (
        AlgorithmConfig.model_validate_json(
            config.model_dump_json()
        ).run_controls.max_iterations
        == 7
    )
    with pytest.raises(ValidationError, match="not both"):
        AlgorithmConfig(run_controls=RunControls(), parameters={"stopping": {}})


@pytest.mark.parametrize("native", [4, 5, 10, 30, 50, 120, 200])
@pytest.mark.parametrize("partial", [{"max_elapsed_s": 60}, {"update_rtol": 1e-5}, {}])
def test_partial_controls_inherit_method_iterations(native, partial):
    assert (
        RunControls(**partial).to_stop_policy(default_iterations=native).max_iterations
        == native
    )


@pytest.mark.parametrize("cap, expected", [(100, 30), (10, 10), (0, 0)])
def test_cap_cannot_increase_inherited_default(cap, expected):
    policy = RunControls().to_stop_policy(
        default_iterations=30, budget={"max_iterations": cap}
    )
    assert policy.max_iterations == expected
    assert (
        RunControls()
        .capped({"max_iterations": cap}, default_iterations=30)
        .max_iterations
        == expected
    )


def test_explicit_request_overrides_default_before_cap():
    assert (
        RunControls(max_iterations=60)
        .to_stop_policy(default_iterations=30, budget={"max_iterations": 40})
        .max_iterations
        == 40
    )
    with pytest.raises(ValueError, match="default_iterations"):
        RunControls().capped({"max_iterations": 100})


def test_inversion_control_resolves_partial_controls_and_caps():
    from usctbench.algorithms._control import InversionControl
    from usctbench.core.schema import GeometrySpec, GridSpec, MeasurementSpec, USCTCase

    case = USCTCase(
        case_id="controls",
        grid=GridSpec(shape=(2, 2), spacing_m=(1, 1)),
        geometry=GeometrySpec(tx_pos_m=[[0, 0]], rx_pos_m=[[1, 1]]),
        measurement=MeasurementSpec(domain="features", delta_tof_s=[[1.0]]),
    )
    config = AlgorithmConfig(
        run_controls=RunControls(max_elapsed_s=60),
        budget_caps=BudgetCaps(max_iterations=100),
    )
    for default in (4, 10, 30, 50):
        control = InversionControl(
            case, config, np.ones((1, 1)), default_iterations=default
        )
        assert control.policy.max_iterations == default
    config.parameters["iterations"] = 7
    assert (
        InversionControl(
            case, config, np.ones((1, 1)), default_iterations=50
        ).policy.max_iterations
        == 7
    )


def test_stage_change_resets_update_patience_not_global_iteration_budget():
    policy = RunControls(
        max_iterations=4, min_iterations=1, update_rtol=0.01
    ).to_stop_policy()
    monitor = StopMonitor(policy, WorkLedger(policy))
    for k, stage in enumerate(("a", "a", "b", "c", "d")):
        monitor.set_stage(stage)
        reason = monitor.observe(
            k,
            np.ones(2),
            residual_norm=1,
            observed_norm=2,
            objective=0.5,
            update_relative=None if k == 0 else 0.001,
        )
        assert reason == ("max_iterations" if k == 4 else None)
    record = monitor.record()
    assert record["stage_id"] == record["selected_stage_id"] == "d"
    assert record["resolved_policy"]["max_iterations"] == 4
    assert [x["stage_id"] for x in monitor.history] == ["a", "a", "b", "c", "d"]


def test_attenuation_zero_reference_does_not_create_a_physical_scale():
    from usctbench.algorithms.attenuation import AttenuationSIRTAlgorithm
    from usctbench.data.synthetic import make_sound_speed_case

    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    case.measurement.log_amp = np.full((8, 8), -0.02)
    config = AlgorithmConfig(
        run_controls=RunControls(max_iterations=3, min_iterations=1, update_rtol=1.0)
    )
    result = AttenuationSIRTAlgorithm().run(case, config)
    assert result.status == "success", result.failure_reason
    history = result.metrics["iteration_history"]
    assert history[1]["relative_update"] is None
    assert history[2]["relative_update"] is not None
    assert result.metrics["iterations"] >= 2
    assert "update_reference_scale_np_per_m" not in result.metrics
    assert "frequency_unspecified" in result.metrics["stopping"]["update_units"]
