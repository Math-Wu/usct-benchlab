"""Experiment plumbing: portable configs and no held-out initialization leakage."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import AlgorithmConfig
from usctbench.core.schema import ReconstructionResult
from usctbench.core.io import write_case_hdf5, write_result_hdf5
from usctbench.metrics import compute_regional_image_metrics
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


def test_fixed_regularization_weight_does_not_follow_band_sensitivity():
    select = load_experiment().regularization_weight
    diagonal = np.array([0.0, 1.0, 2.0, 3.0])
    assert select(diagonal, 0.02) == pytest.approx(0.04)
    assert select(diagonal * 100, 0.02) == pytest.approx(4.0)
    assert select(diagonal, 0.02, 0.04) == 0.04
    assert select(diagonal * 100, 0.02, 0.04) == 0.04
    assert select(diagonal, 0.02, 0.0) == 0.0
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="nonnegative"):
            select(diagonal, 0.02, bad)
    with pytest.raises(ValueError, match="sensitivity"):
        select(np.zeros(3), 0.02)


def test_nested_acquisitions_share_validation_and_use_mean_training_precision():
    from usctbench.evaluation.data import make_data_split

    module = load_experiment()
    controls = []
    for stride in (2, 1):
        parent = np.arange(0, 128, stride)
        positions = np.column_stack(
            [np.sin(parent * np.pi / 64), np.cos(parent * np.pi / 64)]
        )
        distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        valid = np.broadcast_to(
            distance > 0.5 * distance.max(), (1, len(parent), len(parent))
        )
        valid, evaluation = module.shared_channel_validation(valid, parent, parent)
        split = make_data_split(
            np.ones(valid.shape),
            valid_mask=valid,
            tx_positions=positions,
            rx_positions=positions,
            **evaluation,
        )
        assert parent[split.metadata["receiver_indices"]].tolist() == [
            10,
            52,
            76,
            88,
            106,
            120,
            124,
            126,
        ]
        control = SimpleNamespace(
            split=split, precision=np.where(split.train, split.weights, 0)
        )
        before = control.precision.copy()
        divisor = module.normalize_training_weight(control)
        split = control.split
        assert divisor == split.train.sum()
        np.testing.assert_allclose(control.precision.sum(), 1)
        np.testing.assert_allclose(control.precision * divisor, before)
        np.testing.assert_array_equal(control.precision[~split.train], 0)
        assert np.all(split.weights[split.validation] == 1 / divisor)
        controls.append(control)
    low, full = [control.split for control in controls]
    np.testing.assert_array_equal(low.validation, full.validation[:, ::2, ::2])
    np.testing.assert_array_equal(low.train, full.train[:, ::2, ::2])
    assert low.validation.sum() == full.validation.sum()
    assert not full.validation[:, 1::2].any()
    assert not full.train[:, [10, 52, 76, 88, 106, 120, 124, 126]].any()
    assert full.train[:, 1::2, 1::2].any()
    with pytest.raises(ValueError, match="reference receivers"):
        module.shared_channel_validation(
            np.ones((1, 64, 64), bool), np.arange(1, 128, 2), np.arange(1, 128, 2)
        )


def test_ring_exclusion_uses_parent_indices_and_preserves_legacy_mask():
    mask = load_experiment().ring_pair_mask
    ids = np.arange(0, 128, 2)
    positions = np.column_stack([np.sin(ids * np.pi / 64), np.cos(ids * np.pi / 64)])
    distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    legacy = mask(distance, ids, ids)
    np.testing.assert_array_equal(legacy, distance > distance.max() * 0.5)
    quarter, half = [mask(distance, ids, ids, f) for f in (0.25, 0.5)]
    assert np.all((~quarter).sum(axis=0) == 17)
    assert np.all((~half).sum(axis=0) == 33)
    # MATLAB FWI uses a per-side fraction; this experiment uses the total arc.
    half_width = round((63 / 512) * 128)
    fwi_mask = np.array(
        [
            ~np.isin(ids, (source + np.arange(-half_width, half_width + 1)) % 128)
            for source in ids
        ]
    )
    np.testing.assert_array_equal(quarter, fwi_mask)
    assert np.all(~half | legacy) and np.all(~legacy | quarter)
    np.testing.assert_array_equal(quarter, quarter.T)
    np.testing.assert_array_equal(
        mask(distance, ids.astype(np.uint8), ids.astype(np.uint8), 0.25), quarter
    )
    np.testing.assert_array_equal(
        mask(distance[::2], ids[::2], ids, 0.25), quarter[::2]
    )
    np.testing.assert_array_equal(
        mask(distance, (ids + 19) % 128, (ids + 19) % 128, 0.25), quarter
    )
    for bad in (-0.01, 1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="fraction"):
            mask(distance, ids, ids, bad)


def test_near_channels_cannot_leak_into_finite_frequency_gradient():
    from usctbench.core.schema import GridSpec, GeometrySpec, USCTCase, MeasurementSpec
    from usctbench.operators.forward.ray_born import RayBornOperator, RayBornForward
    from usctbench.operators.forward.band_delay import FiniteFrequencyTravelTimeForward

    ids = np.arange(8)
    positions = 0.01 * np.column_stack(
        [np.sin(ids * np.pi / 4), np.cos(ids * np.pi / 4)]
    )
    distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    pair = load_experiment().ring_pair_mask(distance, ids, ids, 0.25, elements=8)
    grid = GridSpec(shape=(8, 8), spacing_m=(0.001, 0.001), origin_m=(-0.004, -0.004))
    geom = GeometrySpec(tx_pos_m=positions, rx_pos_m=positions)
    frequencies = np.linspace(100e3, 300e3, 9)
    water = RayBornOperator(grid, geom, frequencies).background_data()
    observation = BandCorrelationDelay(
        frequencies, np.where(pair[None], water, 0), np.ones((1, 9)), -3e-6, 3e-6
    )
    forward = FiniteFrequencyTravelTimeForward(
        RayBornForward(grid, geom, frequencies, green_backend="volume_integral"),
        observation,
        water,
    )
    model = np.full(grid.shape, 1 / 1500**2)
    model[3:5, 3:5] = 1 / 1480**2
    lin = forward.linearize(model)
    observed = lin.value + 1e-7
    case = USCTCase(
        case_id="masked_gradient",
        grid=grid,
        geometry=geom,
        measurement=MeasurementSpec(domain="features", delta_tof_s=observed[0]),
    )
    gradients = []
    for bad in (np.nan, 1e20):
        values = observed.copy()
        values[:, ~pair] = bad
        control = InversionControl(
            case, AlgorithmConfig(), values, default_iterations=1, valid_mask=pair[None]
        )
        sensitivity = control.weighted_residual(lin.value)
        np.testing.assert_array_equal(sensitivity[:, ~pair], 0)
        pressure_source = lin.jacobian.observation.adjoint(sensitivity)
        np.testing.assert_array_equal(pressure_source[:, ~pair], 0)
        gradients.append(lin.jacobian.adjoint(sensitivity))
    np.testing.assert_array_equal(gradients[0], gradients[1])
    assert np.isfinite(gradients[0]).all() and np.linalg.norm(gradients[0]) > 0


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
    assert int(saved["iteration"]) == 0
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


def audit_function():
    path = Path(__file__).resolve().parents[2] / "scripts/_inverse_solver_audit.py"
    spec = importlib.util.spec_from_file_location("inverse_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.channel_derivative_audit


def test_linear_model_channels_and_excluded_nan():
    audit = audit_function()
    value = np.array([[[1.0, 2.0, np.nan]]])
    tangent = np.array([[[0.2, -0.3, np.nan]]])
    weights = np.array([[[2.0, 0.5, 0.0]]])
    out = audit(
        value, value + tangent, value - tangent, tangent, value * 0, weights, 1.0
    )
    assert out["training_count"] == 2
    assert out["invalid_training_count"] == 0
    assert out["complete_training_derivative"]
    assert abs(out["finite_training_data_derivative_error"]) < 1e-15


def test_one_jump_is_localized_without_claiming_cause():
    audit = audit_function()
    value = np.ones((1, 2, 3)) * 1e-6
    tangent = np.ones_like(value) * 1e-8
    plus = value + tangent
    plus[0, 1, 2] += 3e-6
    out = audit(
        value, plus, value - tangent, tangent, value * 0, np.ones_like(value), 1.0
    )
    top = out["top_channels"][0]
    assert top["index"] == [0, 1, 2]
    assert top["plus_linearization_error_s"] == pytest.approx(3e-6)
    assert out["finite_training_data_derivative_error"] == pytest.approx(
        top["derivative_error"]
    )
    assert "not_by_itself_proof" in out["interpretation"]


def test_invalid_training_ray_is_not_silently_dropped():
    value = np.ones((1, 1, 2))
    plus = np.array([[[1.0, np.nan]]])
    out = audit_function()(value, plus, value, value * 0, value * 0, value, 0.5)
    assert out["invalid_training_count"] == 1
    assert not out["complete_training_derivative"]


@pytest.mark.parametrize("h", [0, -1, np.nan])
def test_invalid_step(h):
    x = np.ones((1, 1, 1))
    with pytest.raises(ValueError, match="difference step"):
        audit_function()(x, x, x, x, x, x, h)


def test_explicit_variants_and_unfinished_refusal(tmp_path, synthetic_case):
    repo = Path(__file__).resolve().parents[2]
    for relative in (
        "cases/high_band/D510022534/envelope_case.h5",
        "cases/low_band/breast_train_speed_class_1_000000/pressure_case.h5",
    ):
        write_case_hdf5(synthetic_case, tmp_path / relative)
    truth = synthetic_case.ground_truth.sound_speed_mps
    metrics = compute_regional_image_metrics(truth, truth)
    metrics["stop_reason"] = "max_iterations"
    for sample in ("high_d", "low_ob"):
        out = tmp_path / f"outer_{sample}_lsmr64"
        out.mkdir()
        write_result_hdf5(
            ReconstructionResult(
                algorithm="test",
                case_id=synthetic_case.case_id,
                sound_speed_mps=truth,
                metrics=metrics,
            ),
            out / "result.h5",
        )
        (out / "metrics.json").write_text(json.dumps(metrics))
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "tx_parent_indices": list(range(128)),
                    "rx_parent_indices": list(range(128)),
                }
            )
        )
        other = tmp_path / f"outer_{sample}_unfinished"
        other.mkdir()
        np.savez(other / "checkpoint.npz", squared_slowness=1 / truth**2)
        (other / "progress.json").write_text(json.dumps([{"iteration": 2}]))
    command = [
        sys.executable,
        str(repo / "scripts/render_multiband_traveltime.py"),
        "--runs",
        str(tmp_path),
        "--handoff",
        str(tmp_path),
        "--variant",
        "LSMR64",
        "outer_{sample}_lsmr64",
        "--out",
        str(tmp_path / "comparison.png"),
    ]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    records = json.loads((tmp_path / "comparison.json").read_text())
    assert len(records) == 2
    assert all(row["variant"] == "LSMR64" for row in records)
    assert all(row["final"] and row["n_tx"] == row["n_rx"] == 128 for row in records)
    assert all(len(row["artifact_sha256"]) == 64 for row in records)
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    unfinished = command + ["--variant", "Pending", "outer_{sample}_unfinished"]
    refused = subprocess.run(unfinished, env=env, capture_output=True, text=True)
    assert refused.returncode != 0
    assert "unfinished run" in refused.stderr
    allowed = subprocess.run(
        unfinished + ["--allow-checkpoint"], env=env, capture_output=True, text=True
    )
    assert allowed.returncode == 0, allowed.stderr
    records = json.loads((tmp_path / "comparison.json").read_text())
    assert records[1]["status"] == "Intermediate checkpoint, iteration unverified"
    assert records[1]["checkpoint_iteration"] is None
    for sample in ("high_d", "low_ob"):
        np.savez(
            tmp_path / f"outer_{sample}_unfinished/checkpoint.npz",
            squared_slowness=1 / truth**2,
            iteration=1,
        )
    allowed = subprocess.run(
        unfinished + ["--allow-checkpoint"], env=env, capture_output=True, text=True
    )
    assert allowed.returncode == 0, allowed.stderr
    records = json.loads((tmp_path / "comparison.json").read_text())
    assert records[1]["status"] == "Intermediate iteration 1, not final"
    bad_metrics = {**metrics, "rmse": 123.0}
    write_result_hdf5(
        ReconstructionResult(
            algorithm="test",
            case_id=synthetic_case.case_id,
            sound_speed_mps=truth,
            metrics=bad_metrics,
        ),
        tmp_path / "outer_high_d_lsmr64/result.h5",
    )
    refused = subprocess.run(command, env=env, capture_output=True, text=True)
    assert refused.returncode != 0
    assert "recorded rmse disagrees" in refused.stderr
