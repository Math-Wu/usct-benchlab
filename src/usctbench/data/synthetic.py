"""Small synthetic fixtures for local tests and smoke runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.core.io import write_case_hdf5
from usctbench.core.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.core.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    USCTCase,
)


def make_grid(
    shape: tuple[int, int] = (32, 32),
    spacing_m: tuple[float, float] = (1.0e-3, 1.0e-3),
) -> GridSpec:
    """Create a centered Cartesian image grid."""

    ny, nx = shape
    dy, dx = spacing_m
    origin_m = (-(ny * dy) / 2.0, -(nx * dx) / 2.0)
    return GridSpec(
        shape=shape,
        spacing_m=spacing_m,
        origin_m=origin_m,
        roi_mask=np.ones(shape, dtype=bool),
    )


def make_ring_geometry(n_transducers: int = 32, radius_m: float = 0.03) -> GeometrySpec:
    """Create a ring geometry using every transducer as both source and receiver."""

    if n_transducers < 4:
        raise ValueError("n_transducers must be at least 4")
    angles = np.linspace(0.0, 2.0 * np.pi, n_transducers, endpoint=False)
    positions = np.column_stack((radius_m * np.sin(angles), radius_m * np.cos(angles)))
    return GeometrySpec(
        type="ring", tx_pos_m=positions, rx_pos_m=positions.copy(), radius_m=radius_m
    )


def circular_sound_speed(
    grid: GridSpec,
    *,
    background_mps: float = 1500.0,
    inclusion_mps: float = 1450.0,
    radius_m: float = 0.006,
    center_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a circular sound-speed phantom."""

    y0, x0 = grid.origin_m
    dy, dx = grid.spacing_m
    ny, nx = grid.shape
    y = y0 + (np.arange(ny) + 0.5) * dy
    x = x0 + (np.arange(nx) + 0.5) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    mask = (yy - center_m[0]) ** 2 + (xx - center_m[1]) ** 2 <= radius_m**2
    image = np.full(grid.shape, background_mps, dtype=float)
    image[mask] = inclusion_mps
    return image


def circular_attenuation(
    grid: GridSpec,
    *,
    background_np_per_m: float = 0.0,
    inclusion_np_per_m: float = 8.0,
    radius_m: float = 0.006,
    center_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a circular attenuation phantom in Np/m."""

    y0, x0 = grid.origin_m
    dy, dx = grid.spacing_m
    ny, nx = grid.shape
    y = y0 + (np.arange(ny) + 0.5) * dy
    x = x0 + (np.arange(nx) + 0.5) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    mask = (yy - center_m[0]) ** 2 + (xx - center_m[1]) ** 2 <= radius_m**2
    image = np.full(grid.shape, background_np_per_m, dtype=float)
    image[mask] = inclusion_np_per_m
    return image


def make_sound_speed_case(
    *,
    case_id: str = "synthetic_circular_sos",
    shape: tuple[int, int] = (32, 32),
    n_transducers: int = 32,
    background_mps: float = 1500.0,
    inclusion_mps: float = 1450.0,
    inclusion_radius_m: float = 0.006,
    inclusion_center_m: tuple[float, float] = (0.0, 0.0),
) -> USCTCase:
    """Create a feature-domain case with straight-ray travel-time differences."""

    grid = make_grid(shape=shape)
    geometry = make_ring_geometry(n_transducers=n_transducers)
    sound_speed = circular_sound_speed(
        grid,
        background_mps=background_mps,
        inclusion_mps=inclusion_mps,
        radius_m=inclusion_radius_m,
        center_m=inclusion_center_m,
    )
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)
    delta_slowness = (1.0 / sound_speed) - (1.0 / background_mps)
    delta_tof_s = projector.forward(delta_slowness).reshape(
        (n_transducers, n_transducers)
    )
    valid_mask = ~np.eye(n_transducers, dtype=bool)
    return USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features", delta_tof_s=delta_tof_s, valid_mask=valid_mask
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=sound_speed),
        metadata=stamp_measurement_metadata(
            {
                "case_type": "synthetic_oracle",
                "synthetic": True,
                "feature_provenance": "oracle_straight_ray_forward_from_ground_truth_sound_speed",
                "reference_sound_speed_mps": background_mps,
                "synthetic_inclusion_radius_m": float(inclusion_radius_m),
                "synthetic_inclusion_center_m": [
                    float(value) for value in inclusion_center_m
                ],
            },
            measurement_provenance=MeasurementProvenance.ORACLE_TRAVEL_TIME,
            benchmark_type="oracle_travel_time",
            forward_model="straight_ray_oracle_from_ground_truth",
            feature_source="oracle_straight_ray_delta_tof",
            extra={
                "measurement_limitations": [
                    "fast debug / solver sanity benchmark",
                    "travel-time data are generated directly from ground truth",
                    "not a formal wavefield inversion benchmark",
                ],
            },
        ),
    )


def make_attenuation_case(
    *,
    case_id: str = "synthetic_circular_attenuation",
    shape: tuple[int, int] = (32, 32),
    n_transducers: int = 32,
) -> USCTCase:
    """Create a feature-domain case with straight-ray log-amplitude ratios."""

    grid = make_grid(shape=shape)
    geometry = make_ring_geometry(n_transducers=n_transducers)
    attenuation = circular_attenuation(grid)
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)
    line_integral = projector.forward(attenuation).reshape(
        (n_transducers, n_transducers)
    )
    valid_mask = ~np.eye(n_transducers, dtype=bool)
    return USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features", log_amp=-line_integral, valid_mask=valid_mask
        ),
        ground_truth=GroundTruthSpec(attenuation_np_per_m=attenuation),
        metadata=stamp_measurement_metadata(
            {
                "case_type": "synthetic_oracle",
                "synthetic": True,
                "feature_provenance": "oracle_straight_ray_forward_from_ground_truth_attenuation",
                "log_amp_convention": "log(case/reference) = -integral(alpha ds)",
            },
            measurement_provenance=MeasurementProvenance.ORACLE_TRAVEL_TIME,
            benchmark_type="oracle_travel_time",
            forward_model="straight_ray_oracle_from_ground_truth",
            feature_source="oracle_straight_ray_log_amp",
            extra={
                "measurement_limitations": [
                    "fast debug / solver sanity benchmark",
                    "attenuation line integral is generated directly from ground truth",
                    "not a formal wavefield inversion benchmark",
                ],
            },
        ),
    )


def make_synthetic_smoke_subset(
    out_root: str | Path,
    *,
    shape: tuple[int, int] = (24, 24),
    n_transducers: int = 24,
) -> list[dict[str, object]]:
    """Write small deterministic sound-speed cases for local smoke benchmarks."""

    root = Path(out_root)
    case_dir = root / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        make_sound_speed_case(
            case_id="synthetic_homogeneous_sos",
            shape=shape,
            n_transducers=n_transducers,
            inclusion_mps=1500.0,
        ),
        make_sound_speed_case(
            case_id="synthetic_circular_sos",
            shape=shape,
            n_transducers=n_transducers,
            inclusion_mps=1450.0,
        ),
    ]
    records: list[dict[str, object]] = []
    for case in cases:
        path = case_dir / f"{case.case_id}.h5"
        write_case_hdf5(case, path)
        records.append(
            {
                "case_id": case.case_id,
                "path": str(path),
                "case_type": case.metadata.get("case_type", "synthetic_oracle"),
                "benchmark_type": case.metadata.get(
                    "benchmark_type", "synthetic_oracle"
                ),
                "shape": list(case.grid.shape),
                "n_transducers": int(case.geometry.tx_pos_m.shape[0]),
                "feature_provenance": case.metadata.get("feature_provenance", ""),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"cases": records}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return records
