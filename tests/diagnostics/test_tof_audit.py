"""Numerical/provenance gates for matched-data experiments."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.core.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.forward.eikonal import EikonalForward

path = Path(__file__).resolve().parents[2] / "scripts/tof_audit_common.py"
spec = importlib.util.spec_from_file_location("tof_audit_common", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def case():
    return USCTCase(
        case_id="test",
        grid=GridSpec(shape=(9, 11), spacing_m=(0.001, 0.0012)),
        geometry=GeometrySpec(
            tx_pos_m=[[-0.001, 0.002], [0.004, -0.001]],
            rx_pos_m=[[0.011, 0.005], [0.009, 0.015]],
        ),
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=[[1e-7, 2e-7], [np.nan, 3e-7]],
            valid_mask=[[True, True], [False, True]],
            ray_weights=[[1, 0.8], [0, 0.5]],
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=np.full((9, 11), 1500.0)),
        metadata={"measurement_provenance": "self_simulated_kwave_wavefield"},
    )


@pytest.mark.parametrize("label", ["matched_straight", "matched_eikonal", "kwave"])
def test_control_roundtrip_and_provenance(tmp_path, label):
    c = case()
    before = c.measurement.delta_tof_s.copy()
    data = before if label == "kwave" else np.zeros((2, 2))
    result = audit.diagnostic_case(c, data, label)
    write_case_hdf5(result, tmp_path / "case.h5")
    restored = read_case_hdf5(tmp_path / "case.h5")
    assert restored.measurement.domain == "features"
    assert restored.measurement.freq_data is None
    np.testing.assert_array_equal(c.measurement.delta_tof_s, before)
    np.testing.assert_array_equal(
        result.measurement.ray_weights, c.measurement.ray_weights
    )
    assert restored.grid.roi_mask is None
    if label != "kwave":
        assert restored.metadata["measurement_provenance"] == "oracle_travel_time"
        assert restored.metadata["uses_kwave_wavefield"] is False


def test_cannot_label_modified_data_as_kwave():
    with pytest.raises(ValueError, match="cannot replace"):
        audit.diagnostic_case(case(), np.zeros((2, 2)), "kwave")


@pytest.mark.parametrize(
    "data", [np.zeros((4,)), np.zeros((2, 2), complex), np.full((2, 2), np.nan)]
)
def test_invalid_control_rejected(data):
    with pytest.raises(ValueError):
        audit.diagnostic_case(case(), data, "matched_straight")


def test_streaming_matches_the_actual_discrete_map():
    c = case()
    speed = 1450 + np.arange(99).reshape((9, 11))
    actual = audit.streamed_eikonal_delta(c.grid, c.geometry, speed)
    op = EikonalForward(c.grid, c.geometry)
    distance = np.linalg.norm(
        c.geometry.tx_pos_m[:, None] - c.geometry.rx_pos_m[None], axis=-1
    )
    expected = op.forward(1 / speed).reshape((2, 2)) - distance / 1500
    np.testing.assert_allclose(actual, expected, atol=2e-20, rtol=1e-12)
    np.testing.assert_array_equal(
        audit.streamed_eikonal_delta(c.grid, c.geometry, np.full(c.grid.shape, 1500.0)),
        np.zeros((2, 2)),
    )


def test_nonfinite_candidate_rejected():
    c = case()
    with pytest.raises(ValueError):
        audit.streamed_eikonal_delta(c.grid, c.geometry, np.full(c.grid.shape, np.nan))
