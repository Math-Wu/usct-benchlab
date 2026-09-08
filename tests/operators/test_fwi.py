"""Complex Helmholtz adjoint plus opt-in tests against the production matrix."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu

from usctbench.core.schema import GridSpec, GeometrySpec
from usctbench.core.schema import USCTCase, MeasurementSpec, AlgorithmConfig
from usctbench.algorithms.fwi.controlled import run_controlled
from usctbench.operators import adjoint_error
from usctbench.operators.forward.fwi import HelmholtzJacobian, MatlabHelmholtzForward


def test_complex_helmholtz_jacobian_transpose_with_column_sampling():
    rng = np.random.default_rng(73)
    grid = GridSpec(shape=(3, 4), spacing_m=(0.001, 0.001))
    matrix = csc_matrix(
        rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)) + 20 * np.eye(12)
    )
    derivative = csc_matrix(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    fields = rng.normal(size=(12, 2)) + 1j * rng.normal(size=(12, 2))
    model = SimpleNamespace(
        grid=grid, data_shape=(1, 2, 3), rx=eye(12, format="csr")[[1, 5, 9]]
    )
    op = HelmholtzJacobian(model, [(splu(matrix), derivative, fields)])
    assert (
        adjoint_error(
            op,
            rng.normal(size=grid.shape),
            rng.normal(size=model.data_shape) + 1j * rng.normal(size=model.data_shape),
        )
        < 1e-12
    )


@pytest.mark.skipif(
    not os.environ.get("USCT_WUST_FUNCTIONS"), reason="external MATLAB/WUST is opt-in"
)
def test_external_wust_forward_derivative_and_adjoint():
    grid = GridSpec(
        shape=(23, 25), spacing_m=(0.0005, 0.0005), origin_m=(-0.00575, -0.00625)
    )
    geom = GeometrySpec(
        tx_pos_m=[[-0.003, -0.001], [0.0025, 0.003]],
        rx_pos_m=[[0.003, -0.002], [-0.002, 0.0025]],
    )
    forward = MatlabHelmholtzForward(
        grid,
        geom,
        [150e3],
        functions_path=os.environ["USCT_WUST_FUNCTIONS"],
        matlab=os.environ.get("USCT_MATLAB", "matlab"),
        pml_m=0.0015,
    )
    yy, xx = np.indices(grid.shape)
    speed = 1500 + 20 * np.exp(-((yy - 11) ** 2 + (xx - 12) ** 2) / 15)
    model = 1 / speed**2
    lin = forward.linearize(model)
    rng = np.random.default_rng(76)
    dm = rng.normal(size=grid.shape) * 1e-8
    values = rng.normal(size=forward.data_shape) + 1j * rng.normal(
        size=forward.data_shape
    )
    assert adjoint_error(lin.jacobian, dm, values) < 1e-10
    eps = 1e-3
    numerical = (
        forward.forward(model + eps * dm) - forward.forward(model - eps * dm)
    ) / (2 * eps)
    np.testing.assert_allclose(
        lin.jacobian.forward(dm), numerical, rtol=1e-4, atol=1e-16
    )


@pytest.mark.skipif(
    not os.environ.get("USCT_WUST_FUNCTIONS"), reason="external MATLAB/WUST is opt-in"
)
def test_external_controlled_fwi_one_truth_free_update():
    """Integration only: synthetic data from WUST itself is an inverse-crime test."""
    yy, xx = np.indices((23, 25))
    roi = (yy - 11) ** 2 + (xx - 12) ** 2 < 25
    grid = GridSpec(
        shape=(23, 25),
        spacing_m=(0.0005, 0.0005),
        origin_m=(-0.00575, -0.00625),
        roi_mask=roi,
    )
    geom = GeometrySpec(
        tx_pos_m=[[-0.003, -0.001], [0.0025, 0.003]],
        rx_pos_m=[[0.003, -0.002], [-0.002, 0.0025], [0.002, -0.003]],
    )
    options = dict(
        functions_path=os.environ["USCT_WUST_FUNCTIONS"],
        matlab=os.environ.get("USCT_MATLAB", "matlab"),
        pml_m=0.0015,
    )
    forward = MatlabHelmholtzForward(grid, geom, [150e3], **options)
    speed = np.where(roi, 1505.0, 1500.0)
    case = USCTCase(
        case_id="wust_online_integration",
        grid=grid,
        geometry=geom,
        measurement=MeasurementSpec(
            domain="frequency",
            freq_data=forward.forward(1 / speed**2),
            frequencies_hz=np.array([150e3]),
        ),
    )
    result = run_controlled(
        case,
        AlgorithmConfig(
            name="fwi_kwave_adapter",
            parameters={
                "functions_path": options["functions_path"],
                "matlab_bin": options["matlab"],
                "pml_m": options["pml_m"],
                "assume_unit_source": True,
                "inner_iterations": 1,
                "evaluation": {"receiver_indices": [2]},
                "stopping": {"max_iterations": 1, "restore_best_validation": False},
            },
        ),
    )
    assert result.metrics["online_stopping"] is True
    assert result.metrics["stop_reason"] == "max_iterations"
    assert result.metrics["iterations"] == 1
    assert result.metrics["data_residual_reduction"] > 0
    assert "rmse" not in result.metrics
