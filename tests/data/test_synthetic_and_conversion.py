from __future__ import annotations

import zipfile

import h5py
import numpy as np
import pytest

from usctbench.core.io import read_case_hdf5
from usctbench.data.conversion import _grid_from_coordinates
from usctbench.data.conversion import convert_nbp_slice2d_mat, convert_kwave_channel_mat
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.data.nbpslice2d import (
    inspect_nbp_slice2d_zip,
    make_nbp_slice2d_smoke_subset,
)
from usctbench.data.synthetic import (
    make_sound_speed_case,
    make_synthetic_smoke_subset,
)


def test_synthetic_cases_have_algorithm_ready_features():
    sos_case = make_sound_speed_case(shape=(10, 10), n_transducers=8)

    assert sos_case.measurement.delta_tof_s.shape == (8, 8)
    assert sos_case.ground_truth.sound_speed_mps.shape == (10, 10)


def test_make_synthetic_smoke_subset_writes_hdf5_cases(tmp_path):
    records = make_synthetic_smoke_subset(
        tmp_path / "synthetic", shape=(8, 8), n_transducers=8
    )

    assert len(records) == 2
    case = read_case_hdf5(
        tmp_path / "synthetic" / "cases" / "synthetic_circular_sos.h5"
    )
    assert case.grid.shape == (8, 8)


def test_grid_from_coordinates_uses_point_spacing():
    xi = np.linspace(-0.01, 0.01, 5)
    yi = np.linspace(-0.02, 0.02, 9)

    grid = _grid_from_coordinates((9, 5), xi=xi, yi=yi)

    assert np.isclose(grid.spacing_m[0], 0.005)
    assert np.isclose(grid.spacing_m[1], 0.005)


@pytest.mark.parametrize("dataset", ["nbp", "kwave"])
def test_raw_extra_material_maps_are_not_required_or_used(tmp_path, dataset):
    source = tmp_path / "A_case.mat"
    speed = np.arange(64, dtype=float).reshape(8, 8) + 1450
    with h5py.File(source, "w") as handle:
        if dataset == "nbp":
            handle["sos"] = speed / 1000
            handle["den"] = np.ones((8, 8))
            handle["label"] = np.ones((8, 8), dtype=np.uint8)
            handle["type"] = [ord("A")]
        else:
            handle["C"] = speed.T
            angle = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            handle["transducerPositionsXY"] = 0.02 * np.column_stack(
                [np.cos(angle), np.sin(angle)]
            )
            handle["time"] = np.arange(5) * 1e-7
            handle["full_dataset"] = np.zeros((8, 8, 5))
    converter = (
        convert_nbp_slice2d_mat if dataset == "nbp" else convert_kwave_channel_mat
    )

    def convert(folder):
        record = converter(
            source, tmp_path / folder, output_shape=(8, 8), n_transducers=8
        )[0]
        return read_case_hdf5(record["path"])

    without = convert("without")
    with h5py.File(source, "a") as handle:
        # Malformed, unit-unknown raw maps are deliberately irrelevant.
        handle["att" if dataset == "nbp" else "atten"] = np.full((2, 3), np.nan)
        handle["y"] = [-999.0]
    with_extra = convert("with_extra")
    np.testing.assert_array_equal(
        with_extra.ground_truth.sound_speed_mps, without.ground_truth.sound_speed_mps
    )
    np.testing.assert_array_equal(
        with_extra.measurement.delta_tof_s, without.measurement.delta_tof_s
    )
    np.testing.assert_allclose(
        with_extra.ground_truth.sound_speed_mps, speed, rtol=1e-15
    )
    op = StraightRayProjector.from_case(with_extra)
    expected = op.forward(
        1 / with_extra.ground_truth.sound_speed_mps
        - 1 / with_extra.metadata["reference_sound_speed_mps"]
    )
    np.testing.assert_array_equal(with_extra.measurement.delta_tof_s.ravel(), expected)
    assert not any("atten" in key for key in with_extra.metadata)
    if dataset == "nbp":
        assert with_extra.measurement.frequencies_hz is None


def test_nbpslice2d_zip_inspection_and_smoke_conversion(tmp_path):
    zip_path = tmp_path / "NBPslices2D.zip"
    mat_path = tmp_path / "A_sample.mat"
    with h5py.File(mat_path, "w") as handle:
        handle.create_dataset("sos", data=np.full((4, 4), 1.5))
        handle.create_dataset("den", data=np.ones((4, 4)))
        handle.create_dataset("att", data=np.zeros((4, 4)))
        handle.create_dataset("label", data=np.ones((4, 4), dtype=np.uint8))
        handle.create_dataset("type", data=np.asarray([ord("A")], dtype=np.uint8))
        handle.create_dataset("y", data=np.asarray([1.0]))
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(mat_path, "A/A_sample.mat")

    index = inspect_nbp_slice2d_zip(zip_path, tmp_path / "index.json")
    manifest = make_nbp_slice2d_smoke_subset(
        zip_path, tmp_path / "nbp", converted_shape=(4, 4), n_transducers=8
    )

    assert index["summary"]["num_cases"] == 1
    assert manifest["converted_cases"]
    assert (tmp_path / "nbp" / "cases").exists()
