"""Read the exact medium used by scripts/kwave_validation_pair.m.

This is an evaluation-only reader, not an observation generator or a source of
inversion priors. Refuse changed files and inconsistent coordinate conventions.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

from usctbench.core.schema import GridSpec


@dataclass(frozen=True)
class ValidationAcquisition:
    grid: GridSpec
    speed_mps: np.ndarray
    positions_yx_m: np.ndarray
    image_grid: GridSpec
    image_speed_mps: np.ndarray
    manifest: dict
    identity: dict


def _centred_grid(y, x):
    axes = [np.asarray(a, dtype=float).ravel() for a in (y, x)]
    for a in axes:
        if a.size < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
            raise ValueError("image axes must be finite increasing cell centres")
        if not np.allclose(np.diff(a), np.diff(a)[0], rtol=1e-10, atol=1e-14):
            raise ValueError("image axes must be uniform")
    h = tuple(float(np.diff(a)[0]) for a in axes)
    return GridSpec(
        shape=tuple(len(a) for a in axes),
        spacing_m=h,
        origin_m=tuple(float(a[0] - d / 2) for a, d in zip(axes, h)),
    )


def sample_speed(speed, source_grid, target_grid, *, background=1500.0):
    """Reproduce prepare's centre-sampled speed interpolation, not slowness."""
    points = np.indices(target_grid.shape, dtype=float)
    coordinates = [
        (
            target_grid.origin_m[a]
            + (points[a] + 0.5) * target_grid.spacing_m[a]
            - source_grid.origin_m[a]
        )
        / source_grid.spacing_m[a]
        - 0.5
        for a in range(2)
    ]
    return map_coordinates(
        speed, coordinates, order=1, mode="constant", cval=background
    )


def read_validation_acquisition(directory):
    """Verify exact input SHA, physical element centres and pressure export.

    Applies only to this repository's explicit lossless 2-D validation exporter.
    It does not guess the meaning of arbitrary external MATLAB datasets.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    source = directory / "simulation_input.mat"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest["input_sha256"]:
        raise ValueError("simulation_input.mat SHA does not match manifest")
    a = loadmat(source)
    speed = np.asarray(a["sim_speed_yx"], dtype=float)
    image = np.asarray(a["image_speed_yx"], dtype=float)
    if (
        speed.ndim != 2
        or speed.shape[0] != speed.shape[1]
        or not np.isfinite(speed).all()
        or np.min(speed) <= 0
        or not np.isfinite(image).all()
        or np.min(image) <= 0
    ):
        raise ValueError("validation speed maps must be finite, positive and 2-D")
    dx = float(a["dx"].item())
    n = speed.shape[0]
    grid = GridSpec(
        shape=speed.shape, spacing_m=(dx, dx), origin_m=(-n // 2 * dx - dx / 2,) * 2
    )
    if list(speed.shape) != manifest["simulation_shape"] or dx != manifest["spacing_m"]:
        raise ValueError("simulation grid differs from manifest")
    image_grid = _centred_grid(a["image_y"], a["image_x"])
    if image.shape != image_grid.shape:
        raise ValueError("image map shape differs from cell-centre axes")
    positions = np.asarray(a["positions_yx"], dtype=float)
    elements = np.asarray(a["elements_yx"])
    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or elements.shape != positions.shape
        or not np.isfinite(positions).all()
        or elements.dtype.kind not in "iu"
        or np.any(elements < 0)
        or np.any(elements >= n)
        or len(np.unique(elements, axis=0)) != len(elements)
    ):
        raise ValueError("invalid or duplicate simulation elements")
    centres = (elements - n // 2) * dx
    if not np.allclose(positions, centres, rtol=0, atol=1e-14):
        raise ValueError(
            "element positions are not the recorded simulation cell centres"
        )
    with h5py.File(directory / "pressure.mat") as f:
        export_positions = np.asarray(f["transducerPositionsXY"])[:, ::-1]
        if not np.array_equal(export_positions, positions):
            raise ValueError(
                "pressure export element order/coordinates differ from simulation input"
            )
        if not np.array_equal(np.asarray(f["C"]), image):
            raise ValueError("pressure export image differs from input [y,x] map")
        for name, expected in (("xi_orig", a["image_x"]), ("yi_orig", a["image_y"])):
            if not np.array_equal(np.asarray(f[name]).ravel(), expected.ravel()):
                raise ValueError("pressure export image coordinates differ from input")
        if tuple(np.asarray(f["simulation_shape"]).ravel().astype(int)) != grid.shape:
            raise ValueError("pressure export simulation shape differs from input")
        if float(np.asarray(f["simulation_dx"]).item()) != dx:
            raise ValueError("pressure export simulation spacing differs from input")
        channel_shape = f["full_dataset"].shape
        if (
            channel_shape != f["water_dataset"].shape
            or channel_shape[:2] != (len(positions),) * 2
        ):
            raise ValueError("object/water channel shapes differ from acquisition")
    reconstructed = sample_speed(image, image_grid, grid)
    error = reconstructed - speed
    identity = {
        "input_sha256_verified": digest,
        "pressure_image_identity": True,
        "pressure_geometry_identity": True,
        "element_cell_centres_verified": True,
        "reconstructed_input_max_error_mps": float(np.max(np.abs(error))),
        "reconstructed_input_rms_error_mps": float(np.sqrt(np.mean(error**2))),
        "scope": "input/export correspondence, not physical continuum convergence",
        "ground_truth_role": "posthoc model evaluation only",
    }
    return ValidationAcquisition(
        grid, speed, positions, image_grid, image, manifest, identity
    )
