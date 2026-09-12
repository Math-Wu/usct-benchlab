"""A recovered image is a diagnostic checkpoint, not a successful inversion."""

import numpy as np
import pytest

from usctbench.algorithms.bent_ray import BentRayGNAdapter
from usctbench.algorithms.ray import (
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
)
from usctbench.core.schema import AlgorithmConfig
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators.base import Linearization
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.straight_ray import StraightRayProjector


@pytest.mark.parametrize(
    "algorithm",
    [
        StraightRayCGLSAlgorithm,
        StraightRaySIRTAlgorithm,
        StraightRaySARTAlgorithm,
        BentRayGNAdapter,
    ],
)
@pytest.mark.parametrize("fault_at", [1, 2])
def test_nonfinite_prediction_fails_even_with_finite_recovery_checkpoint(
    monkeypatch, algorithm, fault_at
):
    case = make_sound_speed_case(
        shape=(8, 8), n_transducers=8, inclusion_radius_m=0.002, inclusion_mps=1490
    )
    case.geometry.tx_pos_m *= 0.25
    case.geometry.rx_pos_m *= 0.25
    is_bent = algorithm is BentRayGNAdapter
    cls = EikonalForward if is_bent else StraightRayProjector
    method = "linearize" if is_bent else "forward"
    original = getattr(cls, method)
    count = [0]

    def broken(self, image):
        output = original(self, image)
        count[0] += 1
        if count[0] >= fault_at:
            if is_bent:
                return Linearization(
                    np.full_like(output.value, np.nan), output.jacobian
                )
            return np.full_like(output, np.nan)
        return output

    monkeypatch.setattr(cls, method, broken)
    result = algorithm().run(
        case, AlgorithmConfig(parameters={"iterations": 3, "inner_iterations": 2})
    )
    assert result.status == "failed", result.metrics.get("stop_reason")
    assert result.failure_reason
    assert result.metrics["stop_reason"] in {
        "numerical_failure",
        "linear_solver_breakdown",
    }
    assert result.metrics["stopping"]["completed_iterations"] == 0
    assert np.isfinite(result.sound_speed_mps).all()
    assert result.metrics["recovered_checkpoint_only"] is True
