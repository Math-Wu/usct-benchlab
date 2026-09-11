"""A zero data residual is not stationarity for a regularized inverse problem."""

import numpy as np

from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.solvers.projected_cg import projected_cg


class Identity:
    def forward(self, x):
        return x.copy()

    def adjoint(self, y):
        return y.copy()


def test_exact_data_fit_does_not_bypass_regularizer():
    # min 1/2 ||x-1||^2 + 1/2 ||x||^2 has x*=1/2, not x=1.
    case = USCTCase(
        case_id="regularized_exact_fit",
        measurement=MeasurementSpec(domain="features", delta_tof_s=np.ones((2, 2))),
        grid=GridSpec(shape=(2, 2), spacing_m=(1, 1)),
        geometry=GeometrySpec(tx_pos_m=[[0, 0], [1, 0]], rx_pos_m=[[0, 1], [1, 1]]),
    )
    control = InversionControl(
        case,
        AlgorithmConfig(
            parameters={
                "stopping": {
                    "max_iterations": 4,
                    "objective_rtol": None,
                    "update_rtol": None,
                    "restore_best_validation": False,
                }
            }
        ),
        np.ones((2, 2)),
        default_iterations=4,
    )
    result, metrics = projected_cg(
        Identity(),
        control,
        initial=np.ones((2, 2)),
        reference=np.zeros((2, 2)),
        project=lambda x: x,
        to_image=lambda x: x,
        damping=1,
    )
    np.testing.assert_allclose(result, 0.5, atol=1e-15)
    assert metrics["iteration_history"][-1]["objective"] == 1
    assert metrics["stopping"]["reason"] != "exact_data_fit"
