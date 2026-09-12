"""Nonlinear model acceptance, not just a frozen-Jacobian residual test."""

import numpy as np
import pytest

from usctbench.algorithms.rwave import RWaveAdapter
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.core.schema import AlgorithmConfig, GroundTruthSpec, MeasurementSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators import adjoint_error
from usctbench.operators.ray_born import RayBornForward


def nonlinear_case():
    case = make_sound_speed_case(
        shape=(8, 8), n_transducers=8, inclusion_radius_m=0.002
    )
    case.geometry.tx_pos_m *= 0.25
    case.geometry.rx_pos_m *= 0.25
    yy, xx = np.indices(case.grid.shape)
    speed = 1500 + 8 * np.exp(-((yy - 3) ** 2 + (xx - 4) ** 2) / 5)
    forward = RayBornForward(case.grid, case.geometry, [250e3, 300e3, 350e3])
    linearization = forward.linearize(1 / speed**2)
    case.measurement = MeasurementSpec(
        domain="frequency",
        frequencies_hz=forward.frequencies_hz,
        freq_data=linearization.value,
        source_spectrum=np.ones((3, 8), complex),
        valid_mask=linearization.jacobian.valid_pair_mask,
    )
    case.ground_truth = GroundTruthSpec()
    return case, forward, linearization


def config(**extra):
    return AlgorithmConfig(
        parameters={
            "mode": "nonlinear",
            "outer_iterations": 3,
            "inner_iterations": 4,
            "smooth_sigma": 0.6,
            "evaluation": {"receiver_indices": [1]},
            "stopping": {
                "restore_best_validation": False,
                "update_rtol": None,
                "objective_rtol": None,
            },
            **extra,
        }
    )


def test_rebuilds_background_and_accepts_only_true_nonlinear_decrease():
    case, _, _ = nonlinear_case()
    result = RWaveAdapter().run(case, config())
    assert result.status == "success", result.failure_reason
    metrics = result.metrics
    assert metrics["nonlinear_background_updates"] >= 1, metrics
    curve = [row["objective"] for row in metrics["iteration_history"]]
    assert np.all(np.diff(curve) < 0), curve
    assert metrics["data_residual_reduction"] > 0.1
    assert metrics["stopping"]["work"]["background_builds"] >= len(curve)
    assert metrics["background_relinearization"] is True
    assert metrics["full_upstream_rwave_port"] is False
    assert "rmse" not in metrics


def test_nonlinear_holdout_is_not_used_for_updates_or_line_search():
    case, _, _ = nonlinear_case()
    changed = case.model_copy(deep=True)
    changed.measurement.freq_data[..., 1] *= 15j
    original = RWaveAdapter().run(case, config(outer_iterations=1))
    altered = RWaveAdapter().run(changed, config(outer_iterations=1))
    assert original.status == altered.status == "success"
    np.testing.assert_array_equal(original.sound_speed_mps, altered.sound_speed_mps)
    assert (
        original.metrics["line_search_history"]
        == altered.metrics["line_search_history"]
    )


def test_born_adjoint_is_exact_but_derivative_contract_is_explicit():
    case, _, linearization = nonlinear_case()
    rng = np.random.default_rng(109)
    op = linearization.jacobian
    assert (
        adjoint_error(
            op,
            rng.normal(size=case.grid.shape),
            rng.normal(size=op.data_shape) + 1j * rng.normal(size=op.data_shape),
        )
        < 1e-12
    )
    assert (
        linearization.derivative_kind
        == "continuum_born_approximation_not_discrete_wkb_derivative"
    )


def test_source_is_required_and_complex_source_roundtrips(tmp_path):
    case, _, _ = nonlinear_case()
    case.measurement.source_spectrum *= 2 + 3j
    path = write_case_hdf5(case, tmp_path / "pressure.h5")
    loaded = read_case_hdf5(path)
    np.testing.assert_array_equal(
        loaded.measurement.source_spectrum, case.measurement.source_spectrum
    )
    loaded.measurement.source_spectrum = None
    result = RWaveAdapter().run(loaded, config())
    assert result.status == "failed"
    assert "calibrated source_spectrum" in result.failure_reason


def test_relative_regularization_is_invariant_to_pressure_units():
    case, _, _ = nonlinear_case()
    other = case.model_copy(deep=True)
    other.measurement.freq_data *= 1e-6
    other.measurement.source_spectrum *= 1e-6
    cfg = config(
        outer_iterations=1,
        regularization_scaling="relative_jacobian_diagonal",
        regularization_lambda=0.2,
        regularization="laplacian",
    )
    a, b = (RWaveAdapter().run(c, cfg) for c in (case, other))
    assert a.status == b.status == "success"
    np.testing.assert_allclose(a.sound_speed_mps, b.sound_speed_mps, rtol=1e-10)
    np.testing.assert_allclose(
        b.metrics["effective_damping"],
        a.metrics["effective_damping"] * 1e-12,
        rtol=1e-12,
    )


def test_underresolved_complex_inverse_grid_is_rejected():
    case, _, _ = nonlinear_case()
    case.grid.spacing_m = (0.005, 0.005)
    result = RWaveAdapter().run(case, config())
    assert result.status == "failed"
    assert "underresolved" in result.failure_reason


def test_physical_laplacian_has_an_exact_transpose_on_anisotropic_grid():
    from usctbench.solvers.least_squares import regularizer, normal_regularizer

    rng = np.random.default_rng(24)
    x, y = rng.normal(size=(2, 9, 11))
    kind = (0.7, 2.3)
    np.testing.assert_allclose(
        np.vdot(regularizer(x, kind), y), np.vdot(x, regularizer(y, kind)), rtol=1e-13
    )
    np.testing.assert_allclose(
        np.vdot(x, normal_regularizer(x, kind)),
        np.linalg.norm(regularizer(x, kind)) ** 2,
        rtol=1e-13,
    )


def test_phase_seed_and_pressure_updates_exclude_heldout_frequency():
    from usctbench.operators.ray_born import RayBornOperator

    case, _, _ = nonlinear_case()
    f = np.array([100e3, 150e3, 200e3, 250e3])
    water = RayBornOperator(case.grid, case.geometry, f)
    speed = np.full(case.grid.shape, 1503.0)
    forward = RayBornForward(
        case.grid, case.geometry, f, green_backend="volume_integral"
    )
    case.measurement = MeasurementSpec(
        domain="frequency",
        frequencies_hz=f,
        freq_data=forward.forward(1 / speed**2),
        water_reference=water.background_data(),
        source_spectrum=water.source_spectrum,
        valid_mask=water.valid_pair_mask,
    )
    changed = case.model_copy(deep=True)
    changed.measurement.freq_data[-1] *= 30j
    cfg = config(
        outer_iterations=1,
        initialization="phase_cgls",
        initialization_iterations=8,
        initialization_smooth_mm=0.3,
        green_backend="volume_integral",
        evaluation={"frequency_indices": [3], "receiver_indices": [1]},
        stopping={
            "restore_best_validation": False,
            "update_rtol": None,
            "objective_rtol": None,
        },
    )
    a, b = [RWaveAdapter().run(c, cfg) for c in (case, changed)]
    assert a.status == b.status == "success", a.failure_reason
    np.testing.assert_array_equal(a.sound_speed_mps, b.sound_speed_mps)
    assert a.metrics["initialization_qc"]["frequencies_hz"] == [100e3, 150e3, 200e3]


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_nonlinear_budget_returns_a_complete_checkpoint(budget):
    case, _, _ = nonlinear_case()
    result = RWaveAdapter().run(case, config(stopping={"max_forward_calls": budget}))
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] == "forward_calls_budget"
    assert result.metrics["stopping"]["work"]["forward_calls"] == budget
    np.testing.assert_allclose(result.sound_speed_mps, 1500)


@pytest.mark.parametrize(
    "settings",
    [
        {"initialization_iterations": 0},
        {"initialization_iterations": 1.5},
        {"initialization_iterations": True},
        {"initialization_lambda": float("nan")},
        {"initialization_smooth_mm": -1},
        {"mode": "fixed_background", "initialization": "phase_cgls"},
    ],
)
def test_invalid_initialization_settings_fail_explicitly(settings):
    case, _, _ = nonlinear_case()
    candidate = config(**settings)
    if settings.get("mode") == "fixed_background":
        # Keep this an initialization rejection test, not an unrelated budget
        # rejection now that variant-specific unused fields fail closed.
        candidate = AlgorithmConfig(parameters=settings)
    result = RWaveAdapter().run(case, candidate)
    assert result.status == "failed"
    assert "initialization" in result.failure_reason
