from __future__ import annotations

import numpy as np
import pytest

from usctbench.algorithms.ray import (
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
    StraightRayProjector,
)
from usctbench.core.schema import AlgorithmConfig, GroundTruthSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.evaluation.data import residual_statistics, make_data_split
from usctbench.operators.base import adjoint_error
from usctbench.operators.straight_ray import (
    StraightRayProjector as ForwardProjector,
)

ALGORITHMS = [
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
]


def small_case():
    return make_sound_speed_case(shape=(8, 8), n_transducers=8, inclusion_mps=1470)


def config(algorithm=None, **parameters):
    return AlgorithmConfig(
        parameters={
            "iterations": 4,
            **({"subsets": 3} if algorithm is StraightRaySARTAlgorithm else {}),
            "stopping": {
                "update_rtol": None,
                "objective_rtol": None,
                "restore_best_validation": False,
            },
            **parameters,
        }
    )


def test_operator_extraction_preserves_alias_and_discrete_adjoint():
    assert StraightRayProjector is ForwardProjector
    case = small_case()
    op = ForwardProjector.from_case(case)
    rng = np.random.default_rng(73)
    error = adjoint_error(
        op, rng.normal(size=case.grid.shape), rng.normal(size=op.n_rays)
    )
    assert error < 1e-12
    assert op.__class__.__module__ == "usctbench.operators.straight_ray"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_training_holdout_isolation_without_truth(algorithm):
    case = small_case().model_copy(update={"ground_truth": GroundTruthSpec()})
    cfg = config(
        algorithm, evaluation={"receiver_indices": [1], "exclude_reciprocal": True}
    )
    first = algorithm().run(case, cfg)
    altered = case.measurement.delta_tof_s.copy()
    altered[:, 1] = altered[:, 1] * -8 + 4e-6
    changed = case.model_copy(
        update={
            "measurement": case.measurement.model_copy(update={"delta_tof_s": altered})
        }
    )
    second = algorithm().run(changed, cfg)
    assert first.failure_reason is None, first.failure_reason
    assert second.failure_reason is None, second.failure_reason
    np.testing.assert_allclose(
        first.sound_speed_mps, second.sound_speed_mps, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        first.metrics["residual_curve"],
        second.metrics["residual_curve"],
        rtol=0,
        atol=0,
    )
    assert first.metrics["stop_reason"] in {"max_iterations", "stationary_gradient"}
    if first.metrics["stop_reason"] == "stationary_gradient":
        assert first.metrics["solver_optimality"]["verified"]
        assert first.metrics["solver_optimality"]["converged"]
    assert (
        first.metrics["evaluation"]["receiver"]["relative_residual"]
        != second.metrics["evaluation"]["receiver"]["relative_residual"]
    )
    assert first.metrics["data_residual_reduction"] > 0
    stop = first.metrics["stopping"]
    assert stop["optimization_variable"] == "delta_slowness"
    assert stop["update_variable"] == "full_slowness"
    assert stop["update_units"] == "s/m"
    assert stop["update_norm_scope"] == "full_physical_grid"
    assert stop["resolved_policy"] == stop["policy"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "budget,limit",
    [("max_forward_calls", 0), ("max_forward_calls", 2), ("max_adjoint_calls", 2)],
)
def test_budgets_return_consistent_complete_checkpoints(algorithm, budget, limit):
    case = small_case()
    cfg = config(
        algorithm, stopping={budget: limit, "update_rtol": None, "objective_rtol": None}
    )
    result = algorithm().run(case, cfg)
    assert result.failure_reason is None, result.failure_reason
    stop = result.metrics["stopping"]
    assert stop["work"][budget.removeprefix("max_")] <= limit
    assert stop["reason"] == budget.removeprefix("max_") + "_budget"
    if "evaluation" in result.metrics:
        op = ForwardProjector.from_case(case)
        prediction = op.forward(1 / result.sound_speed_mps - 1 / 1500).reshape(
            op.ray_shape
        )
        split = make_data_split(
            case.measurement.delta_tof_s, valid_mask=case.measurement.valid_mask
        )
        norm = residual_statistics(
            prediction, case.measurement.delta_tof_s, mask=split.train
        )["residual_norm"]
        np.testing.assert_allclose(
            norm, result.metrics["data_residual_norm"], rtol=1e-9, atol=1e-16
        )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_bound_projected_metrics_correspond_to_returned_image(algorithm):
    case = small_case()
    cfg = config(algorithm, sound_speed_bounds_mps=[1499, 1501])
    result = algorithm().run(case, cfg)
    assert result.failure_reason is None, result.failure_reason
    assert np.min(result.sound_speed_mps) >= 1499 - 1e-10
    assert np.max(result.sound_speed_mps) <= 1501 + 1e-10
    op = ForwardProjector.from_case(case)
    residual = (
        op.forward(1 / result.sound_speed_mps - 1 / 1500).reshape(op.ray_shape)
        - case.measurement.delta_tof_s
    )
    np.testing.assert_allclose(
        np.linalg.norm(residual[case.measurement.valid_mask]),
        result.metrics["data_residual_norm"],
        rtol=1e-9,
        atol=1e-16,
    )


def test_huber_preconditioning_uses_only_training_data_and_global_budget():
    case = small_case()
    cfg = config(
        evaluation={"receiver_indices": [2]},
        robust_loss="huber",
        huber_delta_s=1e-8,
        irls_iterations=3,
        coverage_preconditioning=True,
    )
    first = StraightRayCGLSAlgorithm().run(case, cfg)
    changed_data = case.measurement.delta_tof_s.copy()
    changed_data[:, 2] *= 30
    second = StraightRayCGLSAlgorithm().run(
        case.model_copy(
            update={
                "measurement": case.measurement.model_copy(
                    update={"delta_tof_s": changed_data}
                )
            }
        ),
        cfg,
    )
    assert first.failure_reason is None, first.failure_reason
    np.testing.assert_array_equal(first.sound_speed_mps, second.sound_speed_mps)
    assert first.metrics["iterations"] <= 4
    costs = [row["objective"] for row in first.metrics["iteration_history"]]
    assert np.all(np.diff(costs) <= 1e-25)
    assert first.metrics["objective_name"] == "weighted_huber"


def test_attenuation_holdout_budget_and_returned_residual():
    from usctbench.algorithms.attenuation import AttenuationSIRTAlgorithm
    from usctbench.data.synthetic import make_attenuation_case

    case = make_attenuation_case(shape=(8, 8), n_transducers=8)
    cfg = config(evaluation={"receiver_indices": [1]}, attenuation_upper_np_per_m=0.1)
    first = AttenuationSIRTAlgorithm().run(case, cfg)
    altered = case.measurement.log_amp.copy()
    altered[:, 1] += 100
    changed = case.model_copy(
        update={"measurement": case.measurement.model_copy(update={"log_amp": altered})}
    )
    second = AttenuationSIRTAlgorithm().run(changed, cfg)
    assert first.failure_reason is None, first.failure_reason
    np.testing.assert_array_equal(
        first.attenuation_np_per_m, second.attenuation_np_per_m
    )
    assert np.max(first.attenuation_np_per_m) <= 0.1
    assert first.metrics["stop_reason"] != "not_terminated"
    op = ForwardProjector.from_case(case)
    prediction = op.forward(first.attenuation_np_per_m).reshape(op.ray_shape)
    split = make_data_split(
        -case.measurement.log_amp,
        valid_mask=case.measurement.valid_mask,
        receiver_indices=[1],
        tx_positions=case.geometry.tx_pos_m,
        rx_positions=case.geometry.rx_pos_m,
    )
    expected = split.evaluate(prediction, -case.measurement.log_amp)["train"][
        "weighted_residual_norm"
    ]
    np.testing.assert_allclose(first.metrics["data_residual_norm"], expected)
