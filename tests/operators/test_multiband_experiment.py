"""Experiment plumbing: portable configs and no held-out initialization leakage."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import AlgorithmConfig
from usctbench.operators.forward.band_delay import BandCorrelationDelay
from usctbench.operators.model_space import BilinearBasis


def load_experiment():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_finite_frequency_traveltime.py"
    )
    spec = importlib.util.spec_from_file_location("multiband_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_regularization_has_portable_yaml_types():
    module = load_experiment()
    regularization = module.physical_regularization(
        0.35, np.float64(800e3), (0.0003203125, 0.0003203125)
    )
    decoded = yaml.safe_load(yaml.safe_dump({"regularization": regularization}))
    np.testing.assert_allclose(decoded["regularization"], regularization)
    assert all(type(v) is float for v in regularization)


def test_direct_observed_correlation_chain_and_channel_locality():
    module = load_experiment()
    f = np.linspace(80e3, 350e3, 15)
    water = np.ones((len(f), 2, 3), complex)
    base = BandCorrelationDelay(f, water, np.ones((1, len(f))), -6e-6, 6e-6)
    measured = np.exp(2j * np.pi * f[:, None, None] * 2e-6) * water
    offset = base.linearize(measured).value
    observation = module.ObservedCorrelationDelay(base, measured, offset)
    np.testing.assert_allclose(
        observation.linearize(measured).value, offset, atol=1e-15
    )
    predicted = measured * np.exp(2j * np.pi * f[:, None, None] * 0.4e-6)
    lin = observation.linearize(predicted)
    np.testing.assert_allclose(lin.value - offset, 0.4e-6, atol=1e-15)
    rng = np.random.default_rng(21)
    dp = rng.normal(size=water.shape) + 1j * rng.normal(size=water.shape)
    h = 1e-5
    fd = (
        observation.linearize(predicted + h * dp).value
        - observation.linearize(predicted - h * dp).value
    ) / (2 * h)
    np.testing.assert_allclose(lin.forward(dp), fd, rtol=1e-7, atol=1e-15)
    y = rng.normal(size=offset.shape)
    np.testing.assert_allclose(
        np.vdot(lin.forward(dp), y).real, np.vdot(dp, lin.adjoint(y)).real, rtol=1e-12
    )
    measured[:, :, 2] *= np.exp(2j * np.pi * f[:, None] * -1e-6)
    changed = module.ObservedCorrelationDelay(base, measured, offset).linearize(
        predicted
    )
    np.testing.assert_array_equal(lin.value[:, :, :2], changed.value[:, :, :2])
    np.testing.assert_array_equal(
        lin.coefficient[:, :, :, :2], changed.coefficient[:, :, :, :2]
    )


def test_reduced_nonlinear_full_chain_and_checkpoint(tmp_path):
    from usctbench.core.schema import GridSpec, GeometrySpec, USCTCase, MeasurementSpec
    from usctbench.operators.forward.ray_born import RayBornOperator, RayBornForward
    from usctbench.operators.forward.band_delay import FiniteFrequencyTravelTimeForward

    module = load_experiment()
    grid = GridSpec(shape=(10, 10), spacing_m=(0.001, 0.001), origin_m=(-0.005, -0.005))
    geom = GeometrySpec(
        tx_pos_m=[[-0.008, 0], [0, -0.008]], rx_pos_m=[[0.008, 0], [0, 0.008]]
    )
    f = np.linspace(100e3, 300e3, 9)
    water = RayBornOperator(grid, geom, f).background_data()
    obs = BandCorrelationDelay(f, water, np.ones((1, len(f))), -3e-6, 3e-6)
    fine = FiniteFrequencyTravelTimeForward(
        RayBornForward(
            grid, geom, f, green_backend="volume_integral", green_solver_rtol=1e-12
        ),
        obs,
        water,
    )
    basis = BilinearBasis(grid, (4, 4))
    reduced = module.CoefficientForward(fine, basis)
    model = np.full(basis.shape, 1 / 1500**2)
    model[1:3, 1:3] = 1 / 1480**2
    lin = reduced.linearize(model)
    np.testing.assert_allclose(lin.value, fine.forward(basis.forward(model)))
    rng = np.random.default_rng(7)
    dm = rng.normal(size=basis.shape) * 1e-9
    h = 1e-2
    fd = (reduced.forward(model + h * dm) - reduced.forward(model - h * dm)) / (2 * h)
    np.testing.assert_allclose(lin.jacobian.forward(dm), fd, rtol=1e-4, atol=1e-15)
    y = rng.normal(size=lin.value.shape)
    np.testing.assert_allclose(
        np.vdot(lin.jacobian.forward(dm), y).real,
        np.vdot(dm, lin.jacobian.adjoint(y)).real,
        rtol=1e-12,
    )
    assert reduced.grid.shape == (4, 4)
    assert reduced.physics.grid.shape == (10, 10)
    case = USCTCase(
        case_id="coefficient_checkpoint",
        grid=grid,
        geometry=geom,
        measurement=MeasurementSpec(domain="features", delta_tof_s=lin.value[0]),
    )
    control = module.ProgressControl(
        case, AlgorithmConfig(), lin.value, default_iterations=2, out=tmp_path
    )
    control.basis = basis
    control.observe(0, model, lin.value)
    saved = np.load(tmp_path / "checkpoint.npz")
    np.testing.assert_array_equal(saved["squared_slowness"], basis.forward(model))
    np.testing.assert_array_equal(saved["coefficients"], model)


def test_phase_initialization_ignores_validation_and_ground_truth(synthetic_case):
    module = load_experiment()
    case = synthetic_case
    case.grid.roi_mask = None
    distance = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    f = np.linspace(80e3, 250e3, 15)
    delay = distance * (1 / 1490 - 1 / 1500)
    measured = np.exp(2j * np.pi * f[:, None, None] * delay)
    water = np.ones_like(measured)
    config = AlgorithmConfig(
        parameters={
            "evaluation": {
                "receiver_fraction": 0.25,
                "seed": 42,
                "exclude_reciprocal": True,
            }
        }
    )

    def control():
        return InversionControl(
            case,
            config,
            delay[None],
            default_iterations=2,
            valid_mask=(distance > 0)[None],
        )

    first_control = control()
    args = SimpleNamespace(
        initialization_iterations=8,
        initialization_lambda=0.02,
        initialization_smooth_mm=3,
    )
    first, qc = module.phase_initialization(
        case, measured, water, f, distance, first_control, args
    )
    excluded = ~np.all(first_control.split.train, axis=0)
    measured[:, excluded] = np.nan + 1j * np.nan
    water[:, excluded] = np.nan + 1j * np.nan
    case.ground_truth.sound_speed_mps[:] = 1700
    second, second_qc = module.phase_initialization(
        case, measured, water, f, distance, control(), args
    )
    np.testing.assert_array_equal(first, second)
    assert qc == second_qc
    assert qc["heldout_values_used"] is False
    assert np.isfinite(first).all()
