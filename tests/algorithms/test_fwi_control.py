import numpy as np
import pytest

from usctbench.algorithms.fwi.adapter import (
    KWaveFWIAdapterAlgorithm,
    _configured_iteration,
)
from usctbench.core.schema import AlgorithmConfig, GroundTruthSpec, MeasurementSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators.base import Linearization


def test_legacy_fwi_rejects_online_controls_it_cannot_enforce():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    result = KWaveFWIAdapterAlgorithm().run(
        case, AlgorithmConfig(parameters={"stopping": {"max_iterations": 1}})
    )
    assert result.status == "failed"
    assert "cannot enforce" in result.failure_reason
    with pytest.raises(ValueError, match="allow_ground_truth_selection"):
        _configured_iteration(
            AlgorithmConfig(parameters={"reconstruction_iteration": "best"}), {}, case
        )


def test_controlled_fwi_uses_same_online_monitor_without_gt(monkeypatch):
    # Tests the external callback/control wiring; the PDE derivative is tested
    # separately against actual MATLAB in tests/operators/test_fwi.py.
    import usctbench.algorithms.fwi.controlled as controlled

    class ExternalFixture:
        def __init__(self, grid, geometry, frequencies, **kw):
            self.grid = grid
            self.background_builds = self.eikonal_solves = 0
            self.data_shape = (1, 8, 8)

        def linearize(self, model):
            self.background_builds += 1
            return Linearization(model[None, :, :].astype(complex), self)

        def forward(self, delta):
            return delta[None, :, :].astype(complex)

        def adjoint(self, values):
            return values[0].real

    monkeypatch.setattr(controlled, "MatlabHelmholtzForward", ExternalFixture)
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
    case.measurement = MeasurementSpec(
        domain="frequency",
        frequencies_hz=np.array([100e3]),
        freq_data=np.full((1, 8, 8), 1 / 1510.0**2, complex),
    )
    case.ground_truth = GroundTruthSpec()
    result = KWaveFWIAdapterAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "controlled_operator": True,
                "functions_path": "unused_fixture",
                "pml_m": 0.001,
                "assume_unit_source": True,
                "roi_update_only": False,
                "inner_iterations": 1,
                "stopping": {"max_iterations": 3, "target_relative_residual": 1e-9},
            }
        ),
    )
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] in {"target_residual", "exact_data_fit"}
    assert result.metrics["online_stopping"] is True
    assert result.metrics["production_driver_trajectory_reproduced"] is False
    assert "rmse" not in result.metrics
    np.testing.assert_allclose(result.sound_speed_mps, 1510)
