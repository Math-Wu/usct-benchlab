import numpy as np
import pytest

from usctbench.algorithms.ray import (
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
)
from usctbench.algorithms.bent_ray import BentRayGNAdapter
from usctbench.core.schema import AlgorithmConfig, GroundTruthSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators.eikonal import EikonalForward


@pytest.mark.parametrize(
    "name", ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"]
)
def test_model_prior_uses_no_gt_or_validation_for_updates_and_obeys_budget(name):
    case = make_sound_speed_case(
        shape=(8, 8), n_transducers=8, inclusion_radius_m=0.002, inclusion_mps=1490
    )
    if name == "bent_ray_gn":
        case.geometry.tx_pos_m *= 0.25
        case.geometry.rx_pos_m *= 0.25
        op = EikonalForward(case.grid, case.geometry)
        distance = np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
        )
        case.measurement.delta_tof_s = (
            op.forward(1 / case.ground_truth.sound_speed_mps).reshape(8, 8)
            - distance / 1500
        )
    case.grid.roi_mask = None
    case.ground_truth = GroundTruthSpec()
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 1] += 7e-6
    config = AlgorithmConfig(
        parameters={
            "model_grid_shape": [4, 4],
            "iterations": 2,
            **({"inner_iterations": 4} if name == "bent_ray_gn" else {}),
            **(
                {"regularization": "laplacian", "regularization_lambda": 0.0001}
                if name in {"straight_cgls", "bent_ray_gn"}
                else {}
            ),
            "evaluation": {"receiver_indices": [1]},
            "stopping": {
                "restore_best_validation": False,
                "validation_patience": None,
                "objective_rtol": None,
                "update_rtol": None,
            },
        }
    )
    algo = {
        "straight_cgls": StraightRayCGLSAlgorithm,
        "straight_sirt": StraightRaySIRTAlgorithm,
        "straight_sart": StraightRaySARTAlgorithm,
        "bent_ray_gn": BentRayGNAdapter,
    }[name]()
    a, b = algo.run(case, config), algo.run(altered, config)
    assert a.status == b.status == "success", (a.failure_reason, b.failure_reason)
    assert a.sound_speed_mps.shape == case.grid.shape
    assert "rmse" not in a.metrics
    np.testing.assert_array_equal(a.sound_speed_mps, b.sound_speed_mps)
    assert a.metrics["model_parameterization"]["propagation_shape"] == [8, 8]
    assert a.metrics["data_residual_reduction"] > 0
    config.parameters["stopping"]["max_forward_calls"] = 1
    result = algo.run(case, config)
    assert result.status == "success", result.failure_reason
    assert result.metrics["stopping"]["work"]["forward_calls"] <= 1
    assert result.metrics["stop_reason"] == "forward_calls_budget"
    assert np.isfinite(result.sound_speed_mps).all()
