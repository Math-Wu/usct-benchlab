"""Admission and stopping semantics, independent of image quality."""

import numpy as np
import pytest
from pydantic import ValidationError

from usctbench.core.run_controls import BudgetCaps, RunControls
from usctbench.core.schema import AlgorithmConfig
from usctbench.core.stopping import StopMonitor, WorkLedger


def test_minimal_policy_disables_optional_quality_rules():
    policy = RunControls().to_stop_policy()
    assert policy.max_iterations == 50
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
    ).to_stop_policy()
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
