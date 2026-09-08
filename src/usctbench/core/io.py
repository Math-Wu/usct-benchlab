"""HDF5 serialization for shared USCT schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.core.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    ReconstructionResult,
    USCTCase,
)

SCHEMA_VERSION = "0.1"


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "h5py is required for USCT HDF5 I/O; install usct-benchlab or requirements.txt"
        ) from exc
    return h5py


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _read_str_attr(attrs: Any, key: str) -> str:
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _write_dataset(group: Any, name: str, value: np.ndarray | None) -> None:
    if value is not None:
        group.create_dataset(name, data=np.asarray(value))


def _read_dataset(group: Any, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return group[name][()]


def _prepare_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_case_hdf5(case: USCTCase, path: str | Path) -> Path:
    """Write a USCT case to the standard HDF5 layout."""

    h5py = _h5py()
    out = _prepare_path(path)
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["record_type"] = "USCTCase"
        handle.attrs["case_id"] = case.case_id
        handle.attrs["metadata_json"] = _json_dumps(case.metadata)

        grid = handle.create_group("grid")
        grid.attrs["shape"] = case.grid.shape
        grid.attrs["spacing_m"] = case.grid.spacing_m
        grid.attrs["origin_m"] = case.grid.origin_m
        _write_dataset(grid, "roi_mask", case.grid.roi_mask)

        geometry = handle.create_group("geometry")
        geometry.attrs["type"] = str(case.geometry.type)
        if case.geometry.radius_m is not None:
            geometry.attrs["radius_m"] = case.geometry.radius_m
        _write_dataset(geometry, "tx_pos_m", case.geometry.tx_pos_m)
        _write_dataset(geometry, "rx_pos_m", case.geometry.rx_pos_m)

        measurement = handle.create_group("measurement")
        measurement.attrs["domain"] = str(case.measurement.domain)
        for name in (
            "frequencies_hz",
            "freq_data",
            "time_data",
            "water_reference",
            "water_reference_time",
            "source_spectrum",
            "source_wavelet",
            "time_axis_s",
            "tof_s",
            "delta_tof_s",
            "tof_first_arrival_s",
            "tof_xcorr_s",
            "phase_slope_delay_s",
            "log_amp",
            "valid_mask",
            "feature_quality",
            "ray_weights",
        ):
            _write_dataset(measurement, name, getattr(case.measurement, name))

        ground_truth = handle.create_group("ground_truth")
        _write_dataset(
            ground_truth, "sound_speed_mps", case.ground_truth.sound_speed_mps
        )
        _write_dataset(
            ground_truth, "attenuation_np_per_m", case.ground_truth.attenuation_np_per_m
        )
        _write_dataset(
            ground_truth, "density_kg_per_m3", case.ground_truth.density_kg_per_m3
        )
    return out


def read_case_hdf5(path: str | Path) -> USCTCase:
    """Read a USCT case from the standard HDF5 layout."""

    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        case_id = _read_str_attr(handle.attrs, "case_id")
        metadata = _json_loads(handle.attrs.get("metadata_json", ""))

        grid_group = handle["grid"]
        grid = GridSpec(
            shape=tuple(int(value) for value in grid_group.attrs["shape"]),
            spacing_m=tuple(float(value) for value in grid_group.attrs["spacing_m"]),
            origin_m=tuple(float(value) for value in grid_group.attrs["origin_m"]),
            roi_mask=_read_dataset(grid_group, "roi_mask"),
        )

        geometry_group = handle["geometry"]
        geometry = GeometrySpec(
            type=_read_str_attr(geometry_group.attrs, "type"),
            tx_pos_m=_read_dataset(geometry_group, "tx_pos_m"),
            rx_pos_m=_read_dataset(geometry_group, "rx_pos_m"),
            radius_m=geometry_group.attrs.get("radius_m"),
        )

        measurement_group = handle["measurement"]
        measurement = MeasurementSpec(
            domain=_read_str_attr(measurement_group.attrs, "domain"),
            frequencies_hz=_read_dataset(measurement_group, "frequencies_hz"),
            freq_data=_read_dataset(measurement_group, "freq_data"),
            time_data=_read_dataset(measurement_group, "time_data"),
            water_reference=_read_dataset(measurement_group, "water_reference"),
            water_reference_time=_read_dataset(
                measurement_group, "water_reference_time"
            ),
            source_spectrum=_read_dataset(measurement_group, "source_spectrum"),
            source_wavelet=_read_dataset(measurement_group, "source_wavelet"),
            time_axis_s=_read_dataset(measurement_group, "time_axis_s"),
            tof_s=_read_dataset(measurement_group, "tof_s"),
            delta_tof_s=_read_dataset(measurement_group, "delta_tof_s"),
            tof_first_arrival_s=_read_dataset(measurement_group, "tof_first_arrival_s"),
            tof_xcorr_s=_read_dataset(measurement_group, "tof_xcorr_s"),
            phase_slope_delay_s=_read_dataset(measurement_group, "phase_slope_delay_s"),
            log_amp=_read_dataset(measurement_group, "log_amp"),
            valid_mask=_read_dataset(measurement_group, "valid_mask"),
            feature_quality=_read_dataset(measurement_group, "feature_quality"),
            ray_weights=_read_dataset(measurement_group, "ray_weights"),
        )

        ground_truth_group = handle["ground_truth"]
        ground_truth = GroundTruthSpec(
            sound_speed_mps=_read_dataset(ground_truth_group, "sound_speed_mps"),
            attenuation_np_per_m=_read_dataset(
                ground_truth_group, "attenuation_np_per_m"
            ),
            density_kg_per_m3=_read_dataset(ground_truth_group, "density_kg_per_m3"),
        )
    return USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=measurement,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def write_result_hdf5(result: ReconstructionResult, path: str | Path) -> Path:
    """Write a reconstruction result to HDF5."""

    h5py = _h5py()
    out = _prepare_path(path)
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["record_type"] = "ReconstructionResult"
        handle.attrs["algorithm"] = result.algorithm
        handle.attrs["case_id"] = result.case_id
        handle.attrs["runtime_s"] = result.runtime_s
        handle.attrs["status"] = str(result.status)
        if result.failure_reason is not None:
            handle.attrs["failure_reason"] = result.failure_reason
        handle.attrs["metrics_json"] = _json_dumps(result.metrics)
        handle.attrs["artifacts_json"] = _json_dumps(result.artifacts)

        for name in (
            "sound_speed_mps",
            "attenuation_np_per_m",
            "reflectivity",
            "uncertainty",
        ):
            _write_dataset(handle, name, getattr(result, name))
    return out


def read_result_hdf5(path: str | Path) -> ReconstructionResult:
    """Read a reconstruction result from HDF5."""

    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        return ReconstructionResult(
            algorithm=_read_str_attr(handle.attrs, "algorithm"),
            case_id=_read_str_attr(handle.attrs, "case_id"),
            sound_speed_mps=_read_dataset(handle, "sound_speed_mps"),
            attenuation_np_per_m=_read_dataset(handle, "attenuation_np_per_m"),
            reflectivity=_read_dataset(handle, "reflectivity"),
            uncertainty=_read_dataset(handle, "uncertainty"),
            metrics=_json_loads(handle.attrs.get("metrics_json", "")),
            runtime_s=float(handle.attrs["runtime_s"]),
            status=_read_str_attr(handle.attrs, "status"),
            failure_reason=handle.attrs.get("failure_reason"),
            artifacts=_json_loads(handle.attrs.get("artifacts_json", "")),
        )
