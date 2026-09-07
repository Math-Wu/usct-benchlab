from __future__ import annotations

from usctbench.algorithms.attenuation import AttenuationSIRTAlgorithm
from usctbench.algorithms.bent_ray import BentRayGNAdapter
from usctbench.algorithms.ray import (
    StraightRayCGLSAlgorithm,
    StraightRayProjector,
    StraightRaySARTAlgorithm,
    StraightRaySIRTAlgorithm,
)
from usctbench.algorithms.rwave import RWaveAdapter
from usctbench.core.schema import AlgorithmConfig, ResultStatus
from usctbench.data.synthetic import make_attenuation_case


def test_projector_adjoint_identity(synthetic_case):
    projector = StraightRayProjector.from_case(synthetic_case)
    image = synthetic_case.ground_truth.sound_speed_mps
    ray_values = projector.forward(image)

    lhs = float((projector.forward(image) * ray_values).sum())
    rhs = float((image * projector.adjoint(ray_values)).sum())

    assert abs(lhs - rhs) / max(abs(lhs), 1.0) < 1.0e-10


def test_ray_sound_speed_algorithms_run(synthetic_case):
    config = AlgorithmConfig(
        parameters={
            "iterations": 1,
            "subsets": 4,
            "inner_iterations": 4,
            "outer_iterations": 1,
        }
    )

    for algorithm in (
        StraightRayCGLSAlgorithm(),
        StraightRaySIRTAlgorithm(),
        StraightRaySARTAlgorithm(),
        BentRayGNAdapter(),
    ):
        result = algorithm.run(synthetic_case, config)
        assert result.status == ResultStatus.SUCCESS
        assert result.sound_speed_mps is not None
        assert "data_relative_residual" in result.metrics


def test_native_bent_ray_and_rwave_feature_rejection(synthetic_case):
    config = AlgorithmConfig(parameters={"outer_iterations": 1, "inner_iterations": 3})
    bent = BentRayGNAdapter().run(synthetic_case, config)
    assert bent.status == ResultStatus.SUCCESS
    assert bent.metrics["backend"] == "native_eikonal_fast_marching"
    assert bent.metrics["true_bent_ray"] is True
    assert bent.metrics["surrogate_travel_time_backend"] is False
    assert bent.metrics["stop_reason"] != "not_terminated"
    rwave = RWaveAdapter().run(synthetic_case, config)
    assert rwave.status == ResultStatus.FAILED
    assert "TOF-only" in rwave.failure_reason


def test_string_false_bool_parameters_do_not_enable_ray_options(synthetic_case):
    config = AlgorithmConfig(
        parameters={
            "iterations": 2,
            "subsets": 4,
            "outer_iterations": 1,
            "inner_iterations": 2,
            "roi_update_only": "false",
            "roi_laplacian": "false",
            "line_search": "false",
            "coverage_preconditioning": "false",
            "coverage_preconditioner_normalize": "false",
        }
    )

    cgls = StraightRayCGLSAlgorithm().run(synthetic_case, config)
    sirt = StraightRaySIRTAlgorithm().run(synthetic_case, config)
    sart = StraightRaySARTAlgorithm().run(synthetic_case, config)
    bent = BentRayGNAdapter().run(synthetic_case, config)

    assert cgls.metrics["roi_update_only"] is False
    assert cgls.metrics["roi_laplacian"] is False
    assert cgls.metrics["coverage_preconditioning"] is False
    assert sirt.metrics["roi_update_only"] is False
    assert sart.metrics["roi_update_only"] is False
    assert bent.metrics["roi_update_only"] is False
    assert bent.metrics["roi_laplacian"] is False
    assert bent.metrics["line_search"] is False


def test_attenuation_algorithm_runs_on_log_amplitude_case():
    case = make_attenuation_case(shape=(10, 10), n_transducers=8)
    result = AttenuationSIRTAlgorithm().run(
        case, AlgorithmConfig(parameters={"iterations": 3})
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.attenuation_np_per_m is not None
    assert result.metrics["attenuation_input_has_signal"] is True
