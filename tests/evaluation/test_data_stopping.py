import json

import numpy as np
import pytest

from usctbench.core.stopping import BudgetExhausted, StopMonitor, StopPolicy, WorkLedger
from usctbench.evaluation import make_data_split, residual_statistics


def test_complex_residual_retains_imaginary_component():
    stats = residual_statistics(np.array([1 + 3j]), np.array([1 + 1j]))
    assert stats["residual_norm"] == 2
    assert stats["rmse"] == 2
    assert stats["relative_residual"] == pytest.approx(np.sqrt(2))
    json.dumps(stats, allow_nan=False)


def test_zero_norm_empty_and_bad_prediction_are_distinct():
    assert residual_statistics(np.ones(2), np.zeros(2))["relative_residual"] is None
    assert (
        residual_statistics(np.ones(2), np.ones(2), mask=[False, False])["status"]
        == "empty"
    )
    with pytest.raises(FloatingPointError):
        residual_statistics(np.array([np.nan, 1]), np.ones(2))
    assert (
        residual_statistics(np.array([np.nan, 1]), np.array([np.nan, 1]))["num_samples"]
        == 1
    )
    with pytest.raises(ValueError):
        residual_statistics(np.ones(2), np.ones(2), weights=[-1, 1])


def test_group_holdouts_disjoint_and_reciprocity_safe():
    positions = np.column_stack([np.arange(5), np.zeros(5)])
    obs = np.ones((3, 5, 5), dtype=complex)
    split = make_data_split(
        obs,
        receiver_indices=[1],
        frequency_indices=[2],
        tx_positions=positions,
        rx_positions=positions,
    )
    assert not split.train[:, :, 1].any()
    assert not split.train[:, 1, :].any()
    assert not split.train[2].any()
    assert split.receiver[:2, :, 1].all()
    assert split.joint[2, :, 1].all()
    masks = [split.train, split.receiver, split.frequency, split.joint, split.unused]
    np.testing.assert_array_equal(sum(m.astype(int) for m in masks), np.ones(obs.shape))
    assert split.metadata["reciprocal_tx_excluded"] == [1]
    with pytest.raises(ValueError):
        split.train[0, 0, 0] = False
    json.dumps(split.metadata, allow_nan=False)


def test_random_split_reproducible_and_no_truth_argument():
    kwargs = dict(
        receiver_fraction=0.25,
        frequency_fraction=0.25,
        seed=82,
        exclude_reciprocal=False,
    )
    a = make_data_split(np.ones((4, 6, 8)), **kwargs)
    b = make_data_split(np.ones((4, 6, 8)), **kwargs)
    assert a.metadata == b.metadata
    assert a.receiver.any() and a.frequency.any() and a.joint.any()
    assert not (a.train & a.validation).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"receiver_indices": [0, 0]},
        {"receiver_indices": [9]},
        {"receiver_fraction": 1},
        {"receiver_indices": [0, 1, 2]},
        {"receiver_indices": [0.5]},
        {"frequency_indices": [0]},
        {"receiver_indices": [0], "receiver_fraction": 0.2},
    ],
)
def test_invalid_splits_fail(kwargs):
    with pytest.raises(ValueError):
        make_data_split(np.ones((3, 3)), exclude_reciprocal=False, **kwargs)


def test_weighted_residual_and_missing_data():
    obs = np.array([[1, 2, np.nan], [3, 4, 5]])
    split = make_data_split(obs, weights=[[4, 0, 1], [1, 1, 1]])
    stats = split.evaluate(np.nan_to_num(obs) + 1, obs)["train"]
    assert stats["num_samples"] == 4
    assert stats["weighted_residual_norm"] == pytest.approx(np.sqrt(7))


def monitor(**kwargs):
    policy = StopPolicy(**kwargs)
    return StopMonitor(policy, WorkLedger(policy, clock=lambda: 0.0))


def observe(m, i, **kwargs):
    args = dict(residual_norm=1.0, observed_norm=2.0, objective=0.5)
    args.update(kwargs)
    return m.observe(i, np.array([i + 1.0]), **args)


def test_or_rules_record_all_triggers_with_priority():
    m = monitor(max_iterations=1, target_relative_residual=0.6, noise_norm=1.0)
    assert observe(m, 0) is None
    assert observe(m, 1) == "target_residual"
    assert m.record()["triggered_rules"] == [
        "target_residual",
        "noise_discrepancy",
        "max_iterations",
    ]
    assert m.record()["quality_target_met"] is True
    assert m.record()["selected_iterate_quality_target_met"] is True
    assert m.record()["terminated_iterate_quality_target_met"] is True
    assert m.history[0]["quality_triggered_rules"] == []
    assert m.history[1]["quality_triggered_rules"] == [
        "target_residual",
        "noise_discrepancy",
    ]
    json.dumps(m.record(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs,second,reason",
    [
        ({"max_iterations": 0}, {}, "max_iterations"),
        ({"update_rtol": 0.01}, {"update_relative": 0.001}, "small_model_update"),
        ({"objective_patience": 1}, {}, "objective_plateau"),
        ({"max_iterations": 5}, {"residual_norm": 0, "objective": 0}, "exact_data_fit"),
    ],
)
def test_stop_reasons(kwargs, second, reason):
    m = monitor(**kwargs)
    if kwargs.get("max_iterations") == 0:
        assert observe(m, 0) == reason
    else:
        observe(m, 0)
        assert observe(m, 1, **second) == reason
    record = m.record()
    assert record["quality_target_met"] is (reason == "exact_data_fit")
    assert record["selected_iterate_quality_target_met"] is (reason == "exact_data_fit")
    assert record["terminated_iterate_quality_target_met"] is (
        reason == "exact_data_fit"
    )


def test_validation_restores_best_not_last_state():
    m = monitor(validation_patience=2, objective_rtol=None)
    observe(m, 0, validation_relative=0.9)
    observe(m, 1, validation_relative=0.5)
    observe(m, 2, validation_relative=0.7)
    assert observe(m, 3, validation_relative=0.8) == "validation_plateau"
    state, index = m.selected_state()
    assert index == 1
    np.testing.assert_array_equal(state, [2.0])
    assert m.record()["completed_iterations"] == 3


@pytest.mark.parametrize("restore_best_validation", [True, False])
def test_quality_target_tracks_selected_checkpoint_not_termination(
    restore_best_validation,
):
    m = monitor(
        max_iterations=1,
        target_rmse_mps=1,
        allow_ground_truth_stopping=True,
        restore_best_validation=restore_best_validation,
    )
    assert observe(m, 0, validation_relative=0.1, quality_rmse_mps=2) is None
    assert (
        observe(m, 1, validation_relative=0.2, quality_rmse_mps=0.5)
        == "oracle_quality_target"
    )
    state, index = m.selected_state()
    assert index == (0 if restore_best_validation else 1)
    np.testing.assert_array_equal(state, [index + 1.0])
    record = m.record()
    assert record["quality_target_met"] is (not restore_best_validation)
    assert record["selected_iterate_quality_target_met"] is (
        not restore_best_validation
    )
    assert record["terminated_iterate_quality_target_met"] is True
    assert record["selected_iteration"] == index
    assert record["reason"] == "oracle_quality_target"
    assert record["triggered_rules"] == ["oracle_quality_target", "max_iterations"]
    assert m.history[0]["quality_triggered_rules"] == []
    assert m.history[1]["quality_triggered_rules"] == ["oracle_quality_target"]
    json.dumps({"record": record, "history": m.history}, allow_nan=False)


def test_quality_target_requires_a_complete_checkpoint():
    m = monitor(target_relative_residual=0.1)
    assert m.record()["quality_target_met"] is False
    assert m.finish("target_residual") == "target_residual"
    assert m.selected_state() == (None, None)
    record = m.record()
    assert record["quality_target_met"] is False
    assert record["selected_iterate_quality_target_met"] is False
    assert record["terminated_iterate_quality_target_met"] is False
    assert record["has_complete_checkpoint"] is False
    assert record["selected_iteration"] is None
    assert record["reason"] == "target_residual"
    assert record["triggered_rules"] == ["target_residual"]


def test_budgets_checked_before_operator_call():
    policy = StopPolicy(max_forward_calls=1)
    ledger = WorkLedger(policy)
    assert ledger.call("linear_forward", lambda: 7) == 7
    with pytest.raises(BudgetExhausted, match="forward_calls_budget"):
        ledger.call("linear_forward", lambda: pytest.fail("budget overrun"))
    assert ledger.counts["forward_calls"] == 1
    with pytest.raises(BudgetExhausted, match="time_budget"):
        WorkLedger(StopPolicy(max_elapsed_s=0)).call("adjoint", lambda: 0)


def test_numerical_failure_preserves_last_complete_state():
    m = monitor()
    observe(m, 0)
    assert observe(m, 1, objective=np.nan) == "numerical_failure"
    state, index = m.selected_state()
    assert index == 0
    assert state[0] == 1


def test_oracle_stopping_is_explicit_and_missing_truth_is_not_ignored():
    with pytest.raises(ValueError, match="explicit"):
        StopPolicy(target_rmse_mps=1)
    m = monitor(target_rmse_mps=1, allow_ground_truth_stopping=True)
    with pytest.raises(ValueError, match="no quality metric"):
        observe(m, 0)
    observe(m, 0, quality_rmse_mps=2)
    assert observe(m, 1, quality_rmse_mps=0.5) == "oracle_quality_target"


def test_policy_rejects_typos_nan_and_fractional_counts():
    for kwargs in (
        {"max_iterations": -1},
        {"max_iterations": 1.5},
        {"update_rtol": np.nan},
    ):
        with pytest.raises(ValueError):
            StopPolicy(**kwargs)
    with pytest.raises(ValueError, match="unknown"):
        StopPolicy.from_parameters({"stopping": {"max_iteration": 4}})
    assert not StopPolicy.from_parameters(
        {"stopping": {"restore_best_validation": "false"}}
    ).restore_best_validation


def test_termination_is_sticky_and_cannot_replace_checkpoint():
    m = monitor(max_iterations=1)
    observe(m, 0)
    observe(m, 1)
    before = m.record()
    assert m.finish("numerical_failure") == "max_iterations"
    assert observe(m, 2, objective=np.nan) == "max_iterations"
    assert m.record() == before
    assert m.record()["termination_category"] == "budget"
    assert m.record()["quality_target_met"] is False


def test_control_rejected_objective_cannot_mismatch_image_and_prediction():
    from usctbench.algorithms._control import InversionControl
    from usctbench.core.schema import AlgorithmConfig
    from usctbench.data.synthetic import make_sound_speed_case

    case = make_sound_speed_case(shape=(4, 4), n_transducers=4)
    obs = np.ones((4, 4))
    control = InversionControl(case, AlgorithmConfig(), obs, default_iterations=4)
    initial = np.zeros((4, 4))
    control.observe(0, initial, initial, objective=1)
    control.observe(1, np.ones((4, 4)), obs, objective=np.nan)
    selected, metrics = control.output(initial)
    np.testing.assert_array_equal(selected, initial)
    assert metrics["data_residual_norm"] == 4
    assert metrics["stop_reason"] == "numerical_failure"
    assert metrics["stopping"]["termination_category"] == "failure"
