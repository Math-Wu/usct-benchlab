import h5py
from itertools import permutations
import numpy as np
import pytest

from usctbench.core.io import read_case_hdf5
from usctbench.data.waveforms import (
    _read_channels,
    convert_kwave_pressure_mat,
    pressure_spectrum,
)


@pytest.mark.parametrize("axes", list(permutations(("time", "tx", "rx"))))
def test_chunked_channel_slabs_preserve_axes_and_selection(tmp_path, axes):
    canonical = np.arange(17 * 5 * 6, dtype=np.float32).reshape(17, 5, 6)
    permutation = tuple(("time", "tx", "rx").index(a) for a in axes)
    raw = canonical.transpose(permutation)
    chunks = tuple(1 if a == "time" else raw.shape[i] for i, a in enumerate(axes))
    with h5py.File(tmp_path / "chunked.h5", "w") as handle:
        dataset = handle.create_dataset(
            "p", data=raw, chunks=chunks, compression="gzip"
        )
        actual = _read_channels(dataset, axes, [4, 0], [5, 2], 17, 300)
        expected = canonical[:, [4, 0]][:, :, [5, 2]]
        np.testing.assert_array_equal(actual, expected)


def test_dtft_sign_scale_nonzero_origin_and_nonfft_frequency():
    t = 2.3e-6 + np.arange(32) * 1e-7
    p = np.zeros((32, 2, 3))
    p[7] = 3.0
    f = np.array([123456.0, 310123.0])
    result = pressure_spectrum(p, t, f)
    expected = 3e-7 * np.exp(2j * np.pi * f * t[7])
    np.testing.assert_allclose(
        result, np.broadcast_to(expected[:, None, None], result.shape)
    )


@pytest.mark.parametrize(
    "time,freq",
    [([0, 1, 3], [0.1]), ([0, 0, 0], [1]), ([0, 1, 2], [0.6]), ([0, 1, 2], [0.1, 0.1])],
)
def test_reject_invalid_sampling(time, freq):
    with pytest.raises(ValueError):
        pressure_spectrum(np.ones((3, 2, 2)), time, freq)


def write_fixture(path, speed=1510, truth=True):
    # Asymmetric image and acquisition distinguish all axes, not just shapes.
    p = np.arange(3 * 3 * 16, dtype=np.float32).reshape(3, 3, 16)
    with h5py.File(path, "w") as f:
        f["full_dataset"] = p
        f["water"] = p * 0.5
        f["time"] = np.arange(16) * 1e-7 + 1e-6
        f["transducerPositionsXY"] = [[0.01, 0], [0, 0.02], [-0.03, 0]]
        f["xi_orig"] = [-0.001, 0, 0.001]
        f["yi_orig"] = [-0.002, 0.002]
        if truth:
            f["C"] = np.full((2, 3), speed) + np.arange(6).reshape(2, 3)
    return p


def test_preserve_channels_grid_reference_roundtrip_and_no_oracle(tmp_path):
    source = tmp_path / "channels.mat"
    p = write_fixture(source)
    kw = dict(
        frequencies_hz=[200e3, 333e3],
        tx_indices=[2, 0],
        rx_indices=[1, 2],
        water_dataset="water",
    )
    case = convert_kwave_pressure_mat(source, tmp_path / "case.h5", **kw)
    restored = read_case_hdf5(tmp_path / "case.h5")
    expected = p[[2, 0]][:, [1, 2]].transpose(2, 0, 1)
    np.testing.assert_array_equal(restored.measurement.time_data, expected)
    np.testing.assert_array_equal(
        restored.measurement.freq_data, case.measurement.freq_data
    )
    np.testing.assert_allclose(
        case.measurement.water_reference, 0.5 * case.measurement.freq_data
    )
    np.testing.assert_allclose(case.geometry.tx_pos_m, [[0, -0.03], [0, 0.01]])
    np.testing.assert_allclose(case.grid.origin_m, [-0.004, -0.0015])
    np.testing.assert_allclose(case.grid.spacing_m, [0.004, 0.001])
    np.testing.assert_array_equal(
        case.ground_truth.sound_speed_mps, 1510 + np.arange(6).reshape(2, 3)
    )
    assert case.measurement.delta_tof_s is None
    assert case.metadata["ground_truth_used_for_measurement"] is False
    write_fixture(source, speed=1690)
    changed = convert_kwave_pressure_mat(source, tmp_path / "changed.h5", **kw)
    np.testing.assert_array_equal(
        case.measurement.freq_data, changed.measurement.freq_data
    )
    assert (
        case.metadata["reference_sound_speed_mps"]
        == changed.metadata["reference_sound_speed_mps"]
    )


def test_no_truth_and_byte_guard(tmp_path):
    source = tmp_path / "channels.mat"
    write_fixture(source, truth=False)
    case = convert_kwave_pressure_mat(source, tmp_path / "case.h5")
    assert case.ground_truth.sound_speed_mps is None
    assert case.measurement.time_data.shape == (16, 3, 3)
    with pytest.raises(ValueError, match="max_output_bytes"):
        convert_kwave_pressure_mat(source, tmp_path / "no.h5", max_output_bytes=16)
    with pytest.raises(ValueError, match="indices"):
        convert_kwave_pressure_mat(source, tmp_path / "no.h5", tx_indices=[1, 1])
