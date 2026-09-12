"""End-to-end numerical inversion and hold-out isolation, not clinical validation."""

import numpy as np
import pytest

from usctbench.algorithms.bent_ray import BentRayGNAdapter
from usctbench.algorithms.rwave import RWaveAdapter
from usctbench.core.schema import AlgorithmConfig, GroundTruthSpec, MeasurementSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.ray_born import RayBornOperator


def small_case(physics):
    case = make_sound_speed_case(
        shape=(8, 8), n_transducers=8, inclusion_radius_m=0.002, inclusion_mps=1490
    )
    # Smaller exterior water grid keeps routine CI fast; no geometry inference from GT.
    case.geometry.tx_pos_m *= 0.25
    case.geometry.rx_pos_m *= 0.25
    c = case.ground_truth.sound_speed_mps
    if physics == "bent":
        op = EikonalForward(case.grid, case.geometry)
        distance = np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None, :], axis=-1
        )
        case.measurement.delta_tof_s = op.forward(1 / c).reshape(8, 8) - distance / 1500
    else:
        op = RayBornOperator(case.grid, case.geometry, [120e3, 160e3, 200e3])
        case.measurement = MeasurementSpec(
            domain="frequency",
            freq_data=op.predict(c),
            frequencies_hz=op.frequencies_hz,
            source_spectrum=op.source_spectrum,
            valid_mask=op.valid_pair_mask,
        )
    return case


@pytest.mark.parametrize("mode", ["fixed_background", "nonlinear"])
def test_born_zero_cache_preserves_reconstruction(mode):
    from usctbench.algorithms.configuration import validate_algorithm_config

    case = small_case("born")
    parameters = {
        "mode": mode,
        "green_backend": "volume_integral",
        "green_solver_rtol": 1e-7,
        "iterations": 2,
    }
    if mode == "fixed_background":
        parameters.pop("iterations")
        parameters.update(inner_iterations=2, stopping={"max_iterations": 2})
    config = AlgorithmConfig(parameters=parameters)
    cached = RWaveAdapter().run(case, config)
    uncached_config = AlgorithmConfig(parameters={**parameters, "max_cache_bytes": 0})
    uncached = RWaveAdapter().run(case, uncached_config)
    repeated = RWaveAdapter().run(
        case, validate_algorithm_config("rwave_adapter", uncached_config)
    )
    assert cached.status == uncached.status == repeated.status == "success"
    np.testing.assert_allclose(
        uncached.sound_speed_mps, cached.sound_speed_mps, rtol=1e-12
    )
    np.testing.assert_array_equal(uncached.sound_speed_mps, repeated.sound_speed_mps)


@pytest.mark.parametrize(
    "physics,algorithm", [("bent", BentRayGNAdapter), ("born", RWaveAdapter)]
)
def test_native_inversions_reduce_residual_and_need_no_truth(physics, algorithm):
    case = small_case(physics)
    case.ground_truth = GroundTruthSpec()
    config = AlgorithmConfig(
        parameters={
            **(
                {"mode": "fixed_background"}
                if physics == "born"
                else {"inner_iterations": 5}
            ),
            "iterations": 5,
            "regularization_lambda": 3e-5,
            "evaluation": {"receiver_indices": [1]},
            "stopping": {"update_rtol": None, "objective_rtol": None},
        }
    )
    result = algorithm().run(case, config)
    assert result.status == "success", result.failure_reason
    assert result.metrics["data_residual_reduction"] > 0.5
    assert result.metrics["evaluation"]["receiver"]["num_samples"] > 0
    assert result.metrics["stopping"]["ground_truth_used_for_stopping"] is False
    assert "rmse" not in result.metrics
    assert result.metrics["stopping"]["work"]["adjoint_calls"] > 0
    stop = result.metrics["stopping"]
    assert stop["update_variable"] == (
        "full_slowness" if physics == "bent" else "full_squared_slowness"
    )
    assert stop["update_units"] == ("s/m" if physics == "bent" else "s^2/m^2")
    assert stop["resolved_policy"]["max_iterations"] == 5


@pytest.mark.parametrize(
    "physics,algorithm", [("bent", BentRayGNAdapter), ("born", RWaveAdapter)]
)
def test_heldout_values_do_not_change_training_trajectory(physics, algorithm):
    case = small_case(physics)
    altered = case.model_copy(deep=True)
    data = (
        altered.measurement.delta_tof_s
        if physics == "bent"
        else altered.measurement.freq_data
    )
    data[..., 1] *= 7
    # Disable validation checkpoint selection to compare optimization itself.
    config = AlgorithmConfig(
        parameters={
            **(
                {"mode": "fixed_background"}
                if physics == "born"
                else {"inner_iterations": 3}
            ),
            "iterations": 2,
            "evaluation": {"receiver_indices": [1]},
            "stopping": {
                "restore_best_validation": False,
                "update_rtol": None,
                "objective_rtol": None,
            },
        }
    )
    a, b = algorithm().run(case, config), algorithm().run(altered, config)
    assert a.status == b.status == "success"
    np.testing.assert_array_equal(a.sound_speed_mps, b.sound_speed_mps)
    np.testing.assert_allclose(a.metrics["residual_curve"], b.metrics["residual_curve"])
    assert (
        a.metrics["evaluation_split"]["mask_sha256"]
        == b.metrics["evaluation_split"]["mask_sha256"]
    )


@pytest.mark.parametrize(
    "physics,algorithm", [("bent", BentRayGNAdapter), ("born", RWaveAdapter)]
)
def test_zero_budget_returns_initial_checkpoint_with_reason(physics, algorithm):
    case = small_case(physics)
    result = algorithm().run(
        case, AlgorithmConfig(parameters={"stopping": {"max_forward_calls": 0}})
    )
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] == "forward_calls_budget"
    assert result.metrics["stopping"]["work"]["forward_calls"] == 0
    np.testing.assert_allclose(result.sound_speed_mps, 1500)


def test_frequency_holdout_does_not_train_or_leak_reciprocal_receiver():
    case = small_case("born")
    result = RWaveAdapter().run(
        case,
        AlgorithmConfig(
            parameters={
                # Match the fixed-background Born generator in this split test.
                "mode": "fixed_background",
                "iterations": 4,
                "evaluation": {"frequency_indices": [2], "receiver_indices": [1]},
            }
        ),
    )
    assert result.status == "success", result.failure_reason
    assert result.metrics["evaluation"]["frequency"]["num_samples"] > 0
    assert result.metrics["evaluation"]["joint"]["num_samples"] > 0
    assert result.metrics["evaluation_split"]["reciprocal_tx_excluded"] == [1]


def test_approximate_wkb_failure_is_not_a_successful_inverse_solve():
    result = RWaveAdapter().run(
        small_case("born"),
        AlgorithmConfig(
            parameters={
                "mode": "nonlinear",
                "green_backend": "eikonal_wkb",
                "iterations": 4,
                "evaluation": {"frequency_indices": [2], "receiver_indices": [1]},
            }
        ),
    )
    assert (
        result.metrics["derivative_kind"]
        == "continuum_born_approximation_not_discrete_wkb_derivative"
    )
    assert result.metrics["stop_reason"] == "line_search_failed"
    assert result.status == "failed"
    assert result.metrics["recovered_checkpoint_only"]


def test_bent_cgls_initialization_does_not_use_heldout_data():
    case = small_case("bent")
    case.ground_truth = GroundTruthSpec()
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 1] *= 100
    config = AlgorithmConfig(
        parameters={
            "initialization": "cgls",
            "initialization_iterations": 10,
            "iterations": 2,
            "inner_iterations": 4,
            "smooth_sigma": 0.4,
            "evaluation": {"receiver_indices": [1]},
            "stopping": {
                "restore_best_validation": False,
                "objective_rtol": None,
                "update_rtol": None,
            },
        }
    )
    a, b = [BentRayGNAdapter().run(c, config) for c in (case, altered)]
    assert a.status == b.status == "success", a.failure_reason
    np.testing.assert_array_equal(a.sound_speed_mps, b.sound_speed_mps)
    assert a.metrics["initialization_training_only"] is True
    assert "rmse" not in a.metrics
