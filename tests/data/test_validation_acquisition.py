import hashlib
import json

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from usctbench.core.schema import GridSpec
from usctbench.data.validation_acquisition import (
    read_validation_acquisition,
    sample_speed,
)


def fixture(directory):
    # Non-symmetric image and element order expose [x,y] / [y,x] mistakes.
    speed = 1500 + np.arange(64).reshape(8, 8)
    dx = 0.001
    centres = (np.arange(8) - 4) * dx
    elements = np.array([[2, 1], [4, 5], [6, 3]])
    positions = (elements - 4) * dx
    a = dict(
        sim_speed_yx=speed,
        image_speed_yx=speed,
        dx=dx,
        image_y=centres,
        image_x=centres,
        elements_yx=elements,
        positions_yx=positions,
    )
    savemat(directory / "simulation_input.mat", a)
    manifest = {
        "input_sha256": hashlib.sha256(
            (directory / "simulation_input.mat").read_bytes()
        ).hexdigest(),
        "simulation_shape": [8, 8],
        "spacing_m": dx,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with h5py.File(directory / "pressure.mat", "w") as f:
        f["C"] = speed
        f["transducerPositionsXY"] = positions[:, ::-1]
        f["xi_orig"], f["yi_orig"] = centres, centres
        f["simulation_shape"], f["simulation_dx"] = [8, 8], dx
        f["full_dataset"], f["water_dataset"] = np.zeros((3, 3, 12)), np.zeros(
            (3, 3, 12)
        )
    return speed


def test_exact_input_and_export_identity(tmp_path):
    speed = fixture(tmp_path)
    result = read_validation_acquisition(tmp_path)
    np.testing.assert_array_equal(result.speed_mps, speed)
    np.testing.assert_allclose(result.grid.origin_m, [-0.0045, -0.0045])
    assert result.identity["reconstructed_input_max_error_mps"] < 1e-9


@pytest.mark.parametrize("mutation", ["sha", "transpose", "half_pixel", "order"])
def test_reject_changed_or_misregistered_export(tmp_path, mutation):
    fixture(tmp_path)
    if mutation == "sha":
        with (tmp_path / "simulation_input.mat").open("ab") as f:
            f.write(b"changed")
    else:
        with h5py.File(tmp_path / "pressure.mat", "r+") as f:
            if mutation == "transpose":
                f["C"][...] = f["C"][...].T
            elif mutation == "half_pixel":
                f["transducerPositionsXY"][...] += 0.0005
            else:
                f["transducerPositionsXY"][...] = f["transducerPositionsXY"][...][::-1]
    with pytest.raises(ValueError):
        read_validation_acquisition(tmp_path)


def test_refinement_preserves_original_centres_and_nodes():
    grid = GridSpec(shape=(8, 8), spacing_m=(0.001, 0.001), origin_m=(-0.0045, -0.0045))
    refined = GridSpec(
        shape=(15, 15), spacing_m=(0.0005, 0.0005), origin_m=(-0.00425, -0.00425)
    )
    speed = np.full((8, 8), 1500.0)
    speed[2:6, 2:6] += np.arange(16).reshape(4, 4)
    result = sample_speed(speed, grid, refined)
    np.testing.assert_allclose(result[::2, ::2], speed, atol=1e-10, rtol=0)
