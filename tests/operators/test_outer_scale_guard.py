"""Outer norm guards for representable gradients below the squaring range."""

import math

import numpy as np
import pytest

import usctbench.algorithms.bent_ray as bent
import usctbench.solvers.nonlinear as nonlinear
from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.base import Linearization


class _DiagonalForward:
    def __init__(self, initial, diagonal):
        self.initial, self.diagonal = initial, diagonal
        self.n_rays = initial.size
        self.spatial_order = 1
        self.background_builds = self.eikonal_solves = 0

    def forward(self, image):
        return self.diagonal * image

    def adjoint(self, data):
        return self.diagonal * data.reshape(self.initial.shape)

    def linearize(self, model):
        self.background_builds += 1
        return Linearization(self.forward(model - self.initial), self)


@pytest.fixture(params=["nonlinear", "bent"])
def run_outer(request, monkeypatch):
    def run(*, blocked=False, direction="exact"):
        initial = np.full(
            (2, 2), 1 / 1500 ** (2 if request.param == "nonlinear" else 1)
        )
        diagonal = np.full(initial.shape, 1e-80)
        observed = np.arange(1.0, 5.0).reshape(initial.shape) * 1e-88
        if blocked:
            # The ordinary gradient remains large; only its feasible part is tiny.
            diagonal[0, 0], observed[0, 0] = -1.0, 1.0
        forward = _DiagonalForward(initial, diagonal)
        case = USCTCase(
            case_id="outer_scale_guard",
            grid=GridSpec(shape=initial.shape, spacing_m=(0.001, 0.001)),
            geometry=GeometrySpec(
                tx_pos_m=[[-0.01, 0], [-0.01, 0.001]],
                rx_pos_m=[[0.01, 0], [0.01, 0.001]],
            ),
            # Absolute data avoid adding/subtracting a macroscopic water offset.
            measurement=MeasurementSpec(domain="features", tof_s=observed),
        )
        options = {"rtol": 1e-9}
        config = AlgorithmConfig(
            parameters={
                "iterations": 1,
                "inner_solver": "lsqr",
                "inner_options": options,
                "damping": 0.0,
                "regularization": "identity",
                "sound_speed_bounds_mps": [1400, 1500],
                "stopping": {
                    "objective_rtol": None,
                    "update_rtol": None,
                    "restore_best_validation": False,
                },
            }
        )
        calls = []

        def inner_step(jacobian, residual, current, control, **settings):
            calls.append(settings)
            assert current.shape == initial.shape
            if direction == "tiny":
                return np.full_like(current, 1e-170)
            if direction == "zero":
                return np.zeros_like(current)
            return residual / diagonal

        if request.param == "nonlinear":
            monkeypatch.setattr(nonlinear, "normal_step", inner_step)
            control = InversionControl(case, config, observed, default_iterations=1)
            state, metrics = nonlinear.nonlinear_least_squares(
                forward,
                control,
                initial=initial,
                bounds=(1400, 1500),
                inner_solver="lsqr",
                inner_options=options,
                regularization="identity",
                max_update_mps=100,
            )
        else:
            monkeypatch.setattr(bent, "normal_step", inner_step)
            monkeypatch.setattr(bent, "EikonalForward", lambda *a, **kw: forward)
            result = bent.BentRayGNAdapter().run(case, config)
            expected_status = "failed" if direction == "tiny" else "success"
            assert result.status == expected_status, result.failure_reason
            state, metrics = 1 / result.sound_speed_mps, result.metrics
        assert state.shape == initial.shape
        return initial, state, metrics, calls, -diagonal * observed

    return run


@pytest.mark.parametrize("blocked", [False, True], ids=["ordinary", "projected"])
def test_tiny_representable_gradient_reaches_inner_solve(run_outer, blocked):
    initial, state, metrics, calls, gradient = run_outer(blocked=blocked)
    feasible = gradient.copy()
    if blocked:
        feasible[0, 0] = 0
    assert np.any(feasible != 0) and np.linalg.norm(feasible) == 0
    assert len(calls) == 1
    assert calls[0]["method"] == "lsqr" and calls[0]["rtol"] == 1e-9
    first = metrics["gradient_checks"][0]
    assert first["norm"] == pytest.approx(
        math.hypot(*gradient.ravel()), rel=1e-14, abs=0
    )
    assert first["relative_to_initial"] == 1.0
    assert metrics["iterations"] == 1
    assert np.all(state.ravel()[1:] > initial.ravel()[1:])
    if blocked:
        assert state[0, 0] == initial[0, 0]
    assert any(row["accepted"] for row in metrics["line_search_history"])


@pytest.mark.parametrize("direction", ["tiny", "zero"])
def test_only_exact_zero_direction_uses_fallback_scale(run_outer, direction):
    initial, state, metrics, calls, _ = run_outer(direction=direction)
    assert len(calls) == 1
    assert metrics["gradient_checks"][0]["norm"] > 0
    if direction == "tiny":
        np.testing.assert_array_equal(state, initial)
        assert metrics["stop_reason"] == "line_search_failed"
        assert not any(row["accepted"] for row in metrics["line_search_history"])
    else:
        assert np.all(state > initial)
        assert any(
            row["accepted"] and row["proposal"] == "projected_gradient"
            for row in metrics["line_search_history"]
        )
