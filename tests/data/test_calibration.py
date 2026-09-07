import numpy as np
import pytest

from usctbench.data.calibration import fit_water_source
from usctbench.algorithms.rwave import RWaveAdapter
from usctbench.core.schema import AlgorithmConfig, MeasurementSpec, GroundTruthSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators.forward.ray_born import RayBornOperator


def test_complex_calibration_known_sources_and_missing_channels():
    rng = np.random.default_rng(127)
    unit = rng.normal(size=(3, 4, 6)) + 1j * rng.normal(size=(3, 4, 6))
    source = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    water = unit * source[..., None]
    water[..., 0] = np.nan
    fitted, record = fit_water_source(unit, water)
    np.testing.assert_allclose(fitted, source, rtol=1e-14, atol=1e-14)
    assert record["relative_residual"] < 1e-14
    assert record["specimen_data_used"] is False
    with pytest.raises(ValueError, match="nonzero"):
        fit_water_source(unit * 0, water)
    with pytest.raises(ValueError, match="matching"):
        fit_water_source(unit, water[0])


def test_water_scales_jacobian_not_just_background_and_needs_no_truth():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8, inclusion_mps=1490)
    frequencies = np.array([120e3, 160e3, 200e3])
    source = np.array([2 + 3j, -4 + 2j, 3 - 1j])
    op = RayBornOperator(case.grid, case.geometry, frequencies, source_spectrum=source)
    case.measurement = MeasurementSpec(
        domain="frequency",
        freq_data=op.predict(case.ground_truth.sound_speed_mps),
        frequencies_hz=frequencies,
        valid_mask=op.valid_pair_mask,
        water_reference=op.background_data(),
    )
    case.ground_truth = GroundTruthSpec()
    cfg = AlgorithmConfig(
        parameters={
            "iterations": 4,
            "evaluation": {"receiver_indices": [1], "frequency_indices": [2]},
            "stopping": {
                "restore_best_validation": False,
                "objective_rtol": None,
                "update_rtol": None,
            },
        }
    )
    fitted = RWaveAdapter().run(case, cfg)
    configured = RWaveAdapter().run(
        case, AlgorithmConfig(parameters={**cfg.parameters, "source_spectrum": source})
    )
    assert fitted.failure_reason is None, fitted.failure_reason
    assert configured.failure_reason is None, configured.failure_reason
    np.testing.assert_allclose(
        fitted.sound_speed_mps, configured.sound_speed_mps, atol=1e-8
    )
    assert fitted.metrics["source_spectrum_assumed_unit"] is False
    assert fitted.metrics["source_calibration"]["relative_residual"] < 1e-14
    changed = case.model_copy(deep=True)
    changed.measurement.freq_data[..., 1] *= 8
    changed.measurement.freq_data[2] *= 11
    isolated = RWaveAdapter().run(changed, cfg)
    np.testing.assert_array_equal(fitted.sound_speed_mps, isolated.sound_speed_mps)
