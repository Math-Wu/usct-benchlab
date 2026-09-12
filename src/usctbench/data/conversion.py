"""Dataset-specific conversion helpers for standard USCTCase files."""

from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.data.synthetic import make_grid, make_ring_geometry
from usctbench.core.io import write_case_hdf5
from usctbench.core.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.core.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    USCTCase,
)

NBP_PIXEL_SPACING_M = 1.0e-4
NBP_DEFAULT_ATTENUATION_FREQUENCY_MHZ = 1.0
NBP_ROI_FOV_FRACTION = 0.72
NBP_DENSITY_CLASSES = {
    "A": "almost_entirely_fatty",
    "B": "scattered_fibroglandular",
    "C": "heterogeneously_dense",
    "D": "extremely_dense",
}


def convert_speed_mat_volume(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    indices: list[int] | None = None,
    index_metadata: dict[int, dict[str, Any]] | None = None,
    dataset_name: str | None = None,
    case_id_prefix: str | None = None,
    output_shape: tuple[int, int] = (64, 64),
    spacing_m: tuple[float, float] = (1.0e-3, 1.0e-3),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
) -> list[dict[str, Any]]:
    """Convert selected slices of a MATLAB v7.3 speed volume to USCTCase HDF5.

    This converter is intended for speed-only OpenBreastUS mirrors such as a
    `breast_train_speed.mat` file containing `[ny, nx, n_cases]` sound-speed
    maps. Since those files do not contain measured wavefields, the converter
    generates straight-ray surrogate travel-time features from the speed map and
    a zero log-amplitude field so the classical benchmark harness can be smoked
    end-to-end. Metadata records these assumptions explicitly.
    """

    source = Path(mat_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with _open_speed_volume(source, dataset_name=dataset_name) as speed_volume:
        name = speed_volume["name"]
        dataset = speed_volume["dataset"]
        shape = tuple(int(value) for value in dataset.shape)
        if len(shape) != 3:
            raise ValueError(
                f"speed dataset must be 3-D [ny, nx, n_cases] or [n_cases,ny,nx], got {shape}"
            )
        sample_axis = _infer_speed_volume_sample_axis(shape)
        case_count = shape[sample_axis]
        selected_indices = indices if indices is not None else [0]
        for index in selected_indices:
            if index < 0 or index >= case_count:
                raise IndexError(f"case index {index} outside dataset shape {shape}")
            extra_metadata = dict((index_metadata or {}).get(index, {}))
            sound_speed = np.asarray(
                _read_speed_volume_slice(dataset, index, sample_axis), dtype=float
            )
            sound_speed = _downsample_mean(sound_speed, output_shape)
            case_id_suffix = extra_metadata.pop("case_id_suffix", None)
            prefix = case_id_prefix or source.stem
            case_id = (
                f"{prefix}_{case_id_suffix}"
                if case_id_suffix
                else f"{prefix}_{index:06d}"
            )
            case = _speed_array_to_case(
                sound_speed,
                case_id=case_id,
                source=source,
                dataset_name=name,
                source_index=index,
                source_shape=shape,
                spacing_m=spacing_m,
                n_transducers=n_transducers,
                reference_sound_speed_mps=reference_sound_speed_mps,
            )
            if extra_metadata:
                case = case.model_copy(
                    update={"metadata": {**case.metadata, **extra_metadata}}
                )
            case_path = out_path / f"{case_id}.h5"
            write_case_hdf5(case, case_path)
            record = {
                "case_id": case_id,
                "path": str(case_path),
                "source_path": str(source),
                "source_dataset": name,
                "source_index": index,
                "shape": list(sound_speed.shape),
                "conversion": case.metadata["conversion"],
                "case_type": case.metadata["case_type"],
                "benchmark_type": case.metadata["benchmark_type"],
                "feature_provenance": case.metadata["feature_provenance"],
                "measurement_limitations": case.metadata["measurement_limitations"],
                "has_measured_attenuation": False,
                "attenuation_evidence": "surrogate_zero_log_amp",
            }
            if extra_metadata:
                record.update(extra_metadata)
            records.append(record)
    return records


def speed_mat_metadata(mat_path: str | Path) -> dict[str, Any] | None:
    """Return lightweight metadata for a MATLAB v7.3 speed volume if possible."""

    path = Path(mat_path)
    try:
        with _open_speed_volume(path) as speed_volume:
            dataset = speed_volume["dataset"]
            shape = tuple(int(value) for value in dataset.shape)
            return {
                "dataset": str(speed_volume["name"]),
                "shape": list(shape),
                "dtype": str(dataset.dtype),
                "sample_axis": (
                    int(_infer_speed_volume_sample_axis(shape))
                    if len(shape) == 3
                    else None
                ),
                "mat_format": str(speed_volume["format"]),
            }
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def convert_kwave_channel_mat(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    case_id_prefix: str | None = None,
    output_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
) -> list[dict[str, Any]]:
    """Convert a compact k-Wave channel MAT file to a standard USCTCase.

    Supported files contain `C`, `atten`, `full_dataset`, `time`, and
    `transducerPositionsXY`. The standard case stores image-domain sound speed
    and attenuation ground truth, surrogate straight-ray delay features from
    `C`, and attenuation line-integral features from the simulated attenuation
    map. The source channel tensor is not copied into the benchmark case.
    """

    h5py = _h5py()
    source = Path(mat_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        metadata = _kwave_channel_metadata_from_handle(handle)
        if metadata is None:
            raise ValueError(f"not a supported k-Wave channel MAT file: {source}")
        sound_speed_raw = np.asarray(handle["C"][()], dtype=float)
        attenuation_raw = np.asarray(handle["atten"][()], dtype=float)
        sound_speed = _external_xy_image_to_internal_yx(sound_speed_raw)
        attenuation = _external_xy_image_to_internal_yx(attenuation_raw)
        positions_xy = np.asarray(handle["transducerPositionsXY"][()], dtype=float)
        time_s = np.asarray(handle["time"][()], dtype=float).reshape(-1)
        full_shape = tuple(int(v) for v in handle["full_dataset"].shape)
        source_label = (
            _decode_matlab_chars(handle.get("sim_metadata/sim_label")) or source.stem
        )
        source_npy_path = _decode_matlab_chars(
            handle.get("sim_metadata/source_npy_path")
        )
        frequency_hz = _read_scalar(handle.get("sim_metadata/f_tx"))
        xi = (
            np.asarray(handle["xi_orig"][()], dtype=float).reshape(-1)
            if "xi_orig" in handle
            else None
        )
        yi = (
            np.asarray(handle["yi_orig"][()], dtype=float).reshape(-1)
            if "yi_orig" in handle
            else None
        )

    sound_speed_small = _downsample_mean(sound_speed, output_shape)
    attenuation_small = _downsample_mean(attenuation, output_shape)
    case_id = case_id_prefix or _safe_case_id(source_label)
    case = _kwave_arrays_to_case(
        sound_speed_small,
        attenuation_small,
        positions_xy=positions_xy,
        time_s=time_s,
        source=source,
        source_label=source_label,
        source_shape=tuple(int(v) for v in sound_speed_raw.shape),
        full_dataset_shape=full_shape,
        source_npy_path=source_npy_path,
        frequency_hz=frequency_hz,
        xi=xi,
        yi=yi,
        n_transducers=n_transducers,
    )
    case = case.model_copy(update={"case_id": case_id})
    case_path = out_path / f"{case_id}.h5"
    write_case_hdf5(case, case_path)
    return [
        {
            "case_id": case_id,
            "path": str(case_path),
            "source_path": str(source),
            "source_dataset": "kWave_channel_mat",
            "shape": list(sound_speed_small.shape),
            "source_shape": list(sound_speed_raw.shape),
            "array_axis_convention_raw": "[x,y]",
            "array_axis_convention_internal": "[row=y,col=x]",
            "array_axis_conversion": "transpose_external_xy_to_internal_yx",
            "conversion": case.metadata["conversion"],
            "case_type": case.metadata["case_type"],
            "benchmark_type": case.metadata["benchmark_type"],
            "feature_provenance": case.metadata["feature_provenance"],
            "measurement_limitations": case.metadata["measurement_limitations"],
            "has_measured_attenuation": False,
            "has_simulated_attenuation": True,
            "attenuation_evidence": "simulated_ground_truth_line_integral",
        }
    ]


def kwave_channel_mat_metadata(mat_path: str | Path) -> dict[str, Any] | None:
    """Return metadata for a supported k-Wave channel MAT file if present."""

    h5py = _h5py()
    try:
        with h5py.File(mat_path, "r") as handle:
            return _kwave_channel_metadata_from_handle(handle)
    except Exception:
        return None


def convert_nbp_slice2d_mat(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    case_id_prefix: str | None = None,
    output_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
    attenuation_frequency_mhz: float = NBP_DEFAULT_ATTENUATION_FREQUENCY_MHZ,
) -> list[dict[str, Any]]:
    """Convert one NBPslices2D MAT file to a standard feature-domain case."""

    h5py = _h5py()
    source = Path(mat_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        case, record = _nbp_handle_to_case_record(
            handle,
            case_id=case_id_prefix or _safe_case_id(source.stem),
            source_path=str(source),
            source_member=None,
            output_shape=output_shape,
            n_transducers=n_transducers,
            reference_sound_speed_mps=reference_sound_speed_mps,
            attenuation_frequency_mhz=attenuation_frequency_mhz,
        )
    case_path = out_path / f"{case.case_id}.h5"
    write_case_hdf5(case, case_path)
    record["path"] = str(case_path)
    return [record]


def convert_nbp_slice2d_zip(
    zip_path: str | Path,
    out_dir: str | Path,
    *,
    cases_per_type: int = 1,
    output_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
    attenuation_frequency_mhz: float = NBP_DEFAULT_ATTENUATION_FREQUENCY_MHZ,
) -> list[dict[str, Any]]:
    """Convert selected NBPslices2D MAT members from a ZIP archive."""

    if cases_per_type <= 0:
        raise ValueError("cases_per_type must be positive")

    h5py = _h5py()
    source = Path(zip_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with zipfile.ZipFile(source) as archive:
        selected = _select_nbp_zip_members(
            archive.namelist(), cases_per_type=cases_per_type
        )
        for member in selected:
            data = archive.read(member)
            with h5py.File(io.BytesIO(data), "r") as handle:
                case_id = _safe_case_id(Path(member).stem)
                case, record = _nbp_handle_to_case_record(
                    handle,
                    case_id=case_id,
                    source_path=str(source),
                    source_member=member,
                    output_shape=output_shape,
                    n_transducers=n_transducers,
                    reference_sound_speed_mps=reference_sound_speed_mps,
                    attenuation_frequency_mhz=attenuation_frequency_mhz,
                )
            case_path = out_path / f"{case.case_id}.h5"
            write_case_hdf5(case, case_path)
            record["path"] = str(case_path)
            records.append(record)
    return records


def nbp_slice2d_mat_metadata(mat_path: str | Path) -> dict[str, Any] | None:
    """Return metadata for a supported NBPslices2D MAT file if present."""

    h5py = _h5py()
    try:
        with h5py.File(mat_path, "r") as handle:
            return _nbp_metadata_from_handle(handle)
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def _speed_array_to_case(
    sound_speed_mps: np.ndarray,
    *,
    case_id: str,
    source: Path,
    dataset_name: str,
    source_index: int,
    source_shape: tuple[int, int, int],
    spacing_m: tuple[float, float],
    n_transducers: int,
    reference_sound_speed_mps: float,
) -> USCTCase:
    grid = make_grid(
        shape=tuple(int(v) for v in sound_speed_mps.shape), spacing_m=spacing_m
    )
    radius_m = 0.6 * max(
        grid.shape[0] * grid.spacing_m[0], grid.shape[1] * grid.spacing_m[1]
    )
    geometry = make_ring_geometry(n_transducers=n_transducers, radius_m=radius_m)
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)
    delta_slowness = (1.0 / sound_speed_mps) - (1.0 / reference_sound_speed_mps)
    delta_tof_s = projector.forward(delta_slowness).reshape(projector.ray_shape)
    valid_mask = ~np.eye(projector.ray_shape[0], projector.ray_shape[1], dtype=bool)
    log_amp = np.zeros(projector.ray_shape, dtype=float)
    return USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=delta_tof_s,
            log_amp=log_amp,
            valid_mask=valid_mask,
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=sound_speed_mps),
        metadata=stamp_measurement_metadata(
            {
                "source_dataset": "OpenBreastUS",
                "case_type": "openbreastus_speedmap_surrogate",
                "source_path": str(source),
                "source_mat_dataset": dataset_name,
                "source_index": source_index,
                "source_shape": list(source_shape),
                "conversion": "speed_map_to_straight_ray_surrogate",
                "feature_provenance": "surrogate_delta_tof_from_ground_truth_sound_speed",
                "measurement_domain": "features",
                "measurement_limitations": [
                    "source file contains sound-speed maps only",
                    "delta_tof_s was generated with a straight-ray projector",
                    "log_amp is a zero surrogate and must not be interpreted as measured attenuation",
                    "synthetic ring geometry was generated because measured transducer geometry was unavailable",
                ],
                "reference_sound_speed_mps": reference_sound_speed_mps,
                "spacing_m_assumption": list(spacing_m),
                "attenuation_note": "log_amp is a zero surrogate because this source file contains speed maps only.",
            },
            measurement_provenance=MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE,
            benchmark_type="speedmap_travel_time_surrogate",
            forward_model="straight_ray_speedmap_surrogate",
            feature_source="surrogate_delta_tof_from_ground_truth_sound_speed",
        ),
    )


def _kwave_arrays_to_case(
    sound_speed_mps: np.ndarray,
    attenuation_np_per_m: np.ndarray,
    *,
    positions_xy: np.ndarray,
    time_s: np.ndarray,
    source: Path,
    source_label: str,
    source_shape: tuple[int, int],
    full_dataset_shape: tuple[int, ...],
    source_npy_path: str | None,
    frequency_hz: float | None,
    xi: np.ndarray | None,
    yi: np.ndarray | None,
    n_transducers: int,
) -> USCTCase:
    grid = _grid_from_coordinates(sound_speed_mps.shape, xi=xi, yi=yi)
    geometry = _geometry_from_xy_positions(positions_xy, n_transducers=n_transducers)
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)
    c0 = float(np.nanmedian(sound_speed_mps))
    delta_slowness = (1.0 / sound_speed_mps) - (1.0 / c0)
    delta_tof_s = projector.forward(delta_slowness).reshape(projector.ray_shape)
    attenuation_integral = projector.forward(attenuation_np_per_m).reshape(
        projector.ray_shape
    )
    valid_mask = ~np.eye(projector.ray_shape[0], projector.ray_shape[1], dtype=bool)
    metadata = stamp_measurement_metadata(
        {
            "source_dataset": "kWave_USCT_simulation",
            "case_type": "openbreastus_wavefield",
            "source_path": str(source),
            "source_label": source_label,
            "source_npy_path": source_npy_path,
            "source_shape": list(source_shape),
            "internal_shape": list(sound_speed_mps.shape),
            "array_axis_convention_raw": "[x,y]",
            "array_axis_convention_internal": "[row=y,col=x]",
            "array_axis_conversion": "transpose_external_xy_to_internal_yx",
            "full_dataset_shape": list(full_dataset_shape),
            "conversion": "kwave_channel_mat_to_feature_case",
            "feature_provenance": "surrogate_delta_tof_from_sound_speed_and_attenuation_line_integral_from_simulated_ground_truth",
            "measurement_domain": "features",
            "measurement_limitations": [
                "source file is a k-Wave simulation MAT, not raw OpenBreastUS measured RF data",
                "delta_tof_s was generated with a straight-ray projector from the simulated sound-speed map",
                "log_amp was generated as a straight-ray line integral from the simulated attenuation map",
                "source channel waveforms are present but are not copied into the standard smoke HDF5 case",
            ],
            "reference_sound_speed_mps": c0,
            "attenuation_evidence": "simulated_ground_truth_line_integral",
            "has_simulated_attenuation": True,
            "time_range_s": (
                [float(np.nanmin(time_s)), float(np.nanmax(time_s))]
                if time_s.size
                else None
            ),
            "frequency_hz": frequency_hz,
        },
        measurement_provenance=MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE,
        benchmark_type="speedmap_travel_time_surrogate",
        forward_model="straight_ray_feature_surrogate_from_kwave_property_maps",
        feature_source="surrogate_delta_tof_and_log_amp_from_ground_truth_property_maps",
    )
    return USCTCase(
        case_id=_safe_case_id(source_label),
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features",
            frequencies_hz=(
                np.asarray([frequency_hz], dtype=float)
                if frequency_hz and frequency_hz > 0
                else None
            ),
            delta_tof_s=delta_tof_s,
            log_amp=-attenuation_integral,
            valid_mask=valid_mask,
        ),
        ground_truth=GroundTruthSpec(
            sound_speed_mps=sound_speed_mps, attenuation_np_per_m=attenuation_np_per_m
        ),
        metadata=metadata,
    )


def _nbp_handle_to_case_record(
    handle: Any,
    *,
    case_id: str,
    source_path: str,
    source_member: str | None,
    output_shape: tuple[int, int],
    n_transducers: int,
    reference_sound_speed_mps: float,
    attenuation_frequency_mhz: float,
) -> tuple[USCTCase, dict[str, Any]]:
    metadata = _nbp_metadata_from_handle(handle)
    if metadata is None:
        raise ValueError("not a supported NBPslices2D MAT file")

    sos_mps_raw = np.asarray(handle["sos"][()], dtype=float) * 1000.0
    y_power = float(np.asarray(handle["y"][()]).reshape(-1)[0])
    att_raw = np.asarray(handle["att"][()], dtype=float)
    attenuation_np_per_m_raw = _nbp_attenuation_to_np_per_m(
        att_raw,
        power_law_exponent=y_power,
        frequency_mhz=attenuation_frequency_mhz,
    )
    label_raw = np.asarray(handle["label"][()], dtype=np.uint8)
    density_raw = np.asarray(handle["den"][()], dtype=float)
    type_code = int(np.asarray(handle["type"][()]).reshape(-1)[0])
    density_label = chr(type_code) if 0 <= type_code <= 255 else str(type_code)
    density_class = NBP_DENSITY_CLASSES.get(density_label, "unknown")

    sos_mps_crop, attenuation_np_per_m_crop, label_crop, crop_info = (
        _fit_nbp_field_of_view(
            sos_mps_raw,
            attenuation_np_per_m_raw,
            label_raw,
        )
    )

    sound_speed_mps = _downsample_mean(sos_mps_crop, output_shape)
    attenuation_np_per_m = _downsample_mean(attenuation_np_per_m_crop, output_shape)
    label_small = _downsample_label(label_crop, output_shape)
    roi_mask = label_small > 0
    if not np.any(roi_mask):
        roi_mask = np.ones(output_shape, dtype=bool)

    spacing_m = (
        NBP_PIXEL_SPACING_M * sos_mps_crop.shape[0] / float(output_shape[0]),
        NBP_PIXEL_SPACING_M * sos_mps_crop.shape[1] / float(output_shape[1]),
    )
    grid = make_grid(shape=output_shape, spacing_m=spacing_m)
    grid = grid.model_copy(update={"roi_mask": roi_mask})
    radius_m = 0.6 * max(
        grid.shape[0] * grid.spacing_m[0], grid.shape[1] * grid.spacing_m[1]
    )
    geometry = make_ring_geometry(n_transducers=n_transducers, radius_m=radius_m)
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)

    delta_slowness = (1.0 / sound_speed_mps) - (1.0 / reference_sound_speed_mps)
    delta_tof_s = projector.forward(delta_slowness).reshape(projector.ray_shape)
    attenuation_integral = projector.forward(attenuation_np_per_m).reshape(
        projector.ray_shape
    )
    valid_mask = ~np.eye(projector.ray_shape[0], projector.ray_shape[1], dtype=bool)

    source_ref = f"{source_path}!{source_member}" if source_member else source_path
    case = USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features",
            frequencies_hz=np.asarray([attenuation_frequency_mhz * 1.0e6], dtype=float),
            delta_tof_s=delta_tof_s,
            log_amp=-attenuation_integral,
            valid_mask=valid_mask,
        ),
        ground_truth=GroundTruthSpec(
            sound_speed_mps=sound_speed_mps, attenuation_np_per_m=attenuation_np_per_m
        ),
        metadata=stamp_measurement_metadata(
            {
                "source_dataset": "NBPslices2D",
                "case_type": "nbpslice2d_property_map_surrogate",
                "source_path": source_path,
                "source_member": source_member,
                "source_ref": source_ref,
                "source_shape": list(sos_mps_raw.shape),
                "fitted_source_shape": list(sos_mps_crop.shape),
                "roi_fit": crop_info,
                "conversion": "nbpslice2d_to_feature_case",
                "feature_provenance": "surrogate_delta_tof_from_nbp_sound_speed_and_attenuation_line_integral_from_nbp_ground_truth",
                "measurement_domain": "features",
                "measurement_limitations": [
                    "NBPslices2D contains acoustic property maps, not measured RF or pressure wavefields",
                    "delta_tof_s was generated with a straight-ray projector from the sound-speed map",
                    "log_amp was generated as a straight-ray line integral from the attenuation map",
                    "synthetic ring geometry was generated because acquisition geometry is not included in the slice files",
                ],
                "reference_sound_speed_mps": reference_sound_speed_mps,
                "density_class": density_class,
                "density_label": density_label,
                "nbp_type_code": type_code,
                "attenuation_frequency_mhz": attenuation_frequency_mhz,
                "attenuation_power_law_exponent": y_power,
                "attenuation_source_units": "dB/(MHz^y mm)",
                "attenuation_conversion": "Np/m = att_dB_per_MHz_y_mm * frequency_mhz**y * ln(10)/20 * 1000",
                "density_source_units": "g/mm^3",
                "density_kg_per_m3_min": float(np.nanmin(density_raw) * 1.0e9),
                "density_kg_per_m3_max": float(np.nanmax(density_raw) * 1.0e9),
                "label_values": [
                    int(value) for value in sorted(np.unique(label_raw).tolist())
                ],
                "pixel_spacing_m_assumption": NBP_PIXEL_SPACING_M,
                "effective_spacing_m": list(spacing_m),
                "has_simulated_attenuation": True,
                "attenuation_evidence": "nbp_numerical_phantom_ground_truth_line_integral",
            },
            measurement_provenance=MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE,
            benchmark_type="speedmap_travel_time_surrogate",
            forward_model="straight_ray_speedmap_surrogate",
            feature_source="surrogate_delta_tof_and_log_amp_from_nbp_property_maps",
        ),
    )
    record = {
        "case_id": case_id,
        "source_path": source_path,
        "source_member": source_member,
        "source_dataset": "NBPslices2D",
        "case_type": case.metadata["case_type"],
        "benchmark_type": case.metadata["benchmark_type"],
        "shape": list(sound_speed_mps.shape),
        "source_shape": list(sos_mps_raw.shape),
        "fitted_source_shape": list(sos_mps_crop.shape),
        "roi_fit": crop_info,
        "conversion": case.metadata["conversion"],
        "feature_provenance": case.metadata["feature_provenance"],
        "measurement_limitations": case.metadata["measurement_limitations"],
        "density_label": density_label,
        "density_class": density_class,
        "attenuation_frequency_mhz": attenuation_frequency_mhz,
        "attenuation_power_law_exponent": y_power,
        "has_measured_attenuation": False,
        "has_simulated_attenuation": True,
        "attenuation_evidence": case.metadata["attenuation_evidence"],
    }
    return case, record


def _downsample_mean(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize a property map to the benchmark grid with linear interpolation.

    The historical name is kept for API stability. For quality comparisons this
    intentionally rescales the full fitted field of view instead of center
    cropping, so 480x480 maps converted to 256x256 keep the whole phantom.
    """

    array = np.asarray(image, dtype=float)
    ny, nx = array.shape
    out_y, out_x = output_shape
    if out_y <= 0 or out_x <= 0:
        raise ValueError("output_shape must be positive")
    if (ny, nx) == (out_y, out_x):
        return array.astype(float, copy=False)
    y_src = np.linspace(0.0, ny - 1.0, out_y)
    x_src = np.linspace(0.0, nx - 1.0, out_x)
    y0 = np.floor(y_src).astype(int)
    x0 = np.floor(x_src).astype(int)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    wy = (y_src - y0)[:, None]
    wx = (x_src - x0)[None, :]
    top = (1.0 - wx) * array[np.ix_(y0, x0)] + wx * array[np.ix_(y0, x1)]
    bottom = (1.0 - wx) * array[np.ix_(y1, x0)] + wx * array[np.ix_(y1, x1)]
    return ((1.0 - wy) * top + wy * bottom).astype(float, copy=False)


def _external_xy_image_to_internal_yx(image: np.ndarray) -> np.ndarray:
    """Convert external k-Wave/MATLAB image arrays from [x,y] to [row=y,col=x]."""

    array = np.asarray(image, dtype=float).squeeze()
    if array.ndim != 2:
        raise ValueError(
            f"external k-Wave image arrays must be 2-D [x,y], got {array.shape}"
        )
    return np.ascontiguousarray(array.T)


def _fit_nbp_field_of_view(
    sound_speed_mps: np.ndarray,
    attenuation_np_per_m: np.ndarray,
    label: np.ndarray,
    *,
    roi_fov_fraction: float = NBP_ROI_FOV_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Crop an NBPslice2D map so the breast ROI occupies a useful FOV."""

    label = np.asarray(label)
    roi = label > 0
    if not np.any(roi):
        info = {
            "enabled": False,
            "reason": "no_positive_label",
            "source_bbox_pixels": [0, 0, int(label.shape[0]), int(label.shape[1])],
            "crop_bbox_pixels": [0, 0, int(label.shape[0]), int(label.shape[1])],
            "target_roi_fov_fraction": float(roi_fov_fraction),
        }
        return sound_speed_mps, attenuation_np_per_m, label, info

    rows, cols = np.where(roi)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    roi_h = y1 - y0
    roi_w = x1 - x0
    if not 0.0 < roi_fov_fraction <= 1.0:
        raise ValueError("roi_fov_fraction must be in (0, 1]")
    side = int(np.ceil(max(roi_h, roi_w) / roi_fov_fraction))
    side = max(side, roi_h, roi_w, 1)
    side = min(side, int(label.shape[0]), int(label.shape[1]))
    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    crop_y0 = int(round(center_y - 0.5 * side))
    crop_x0 = int(round(center_x - 0.5 * side))
    crop_y0 = min(max(crop_y0, 0), int(label.shape[0]) - side)
    crop_x0 = min(max(crop_x0, 0), int(label.shape[1]) - side)
    crop_y1 = crop_y0 + side
    crop_x1 = crop_x0 + side
    info = {
        "enabled": True,
        "source_bbox_pixels": [y0, x0, y1, x1],
        "crop_bbox_pixels": [crop_y0, crop_x0, crop_y1, crop_x1],
        "source_roi_shape_pixels": [roi_h, roi_w],
        "crop_shape_pixels": [side, side],
        "target_roi_fov_fraction": float(roi_fov_fraction),
        "actual_roi_fov_fraction": float(max(roi_h, roi_w) / side),
    }
    crop = np.s_[crop_y0:crop_y1, crop_x0:crop_x1]
    return sound_speed_mps[crop], attenuation_np_per_m[crop], label[crop], info


def _downsample_label(label: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    label = np.asarray(label)
    ny, nx = label.shape
    out_y, out_x = output_shape
    if out_y <= 0 or out_x <= 0:
        raise ValueError("output_shape must be positive")
    if out_y > ny or out_x > nx:
        y_idx = np.linspace(0, ny - 1, out_y).round().astype(int)
        x_idx = np.linspace(0, nx - 1, out_x).round().astype(int)
        return label[np.ix_(y_idx, x_idx)].astype(label.dtype, copy=False)
    block_y = max(1, ny // out_y)
    block_x = max(1, nx // out_x)
    crop_y = out_y * block_y
    crop_x = out_x * block_x
    start_y = (ny - crop_y) // 2
    start_x = (nx - crop_x) // 2
    cropped = label[start_y : start_y + crop_y, start_x : start_x + crop_x]
    return cropped.reshape(out_y, block_y, out_x, block_x).max(axis=(1, 3))


def _nbp_attenuation_to_np_per_m(
    att_dB_per_mhz_y_mm: np.ndarray, *, power_law_exponent: float, frequency_mhz: float
) -> np.ndarray:
    if frequency_mhz <= 0:
        raise ValueError("attenuation_frequency_mhz must be positive")
    dB_per_mm = np.asarray(att_dB_per_mhz_y_mm, dtype=float) * (
        float(frequency_mhz) ** float(power_law_exponent)
    )
    return dB_per_mm * (np.log(10.0) / 20.0) * 1000.0


def _grid_from_coordinates(
    shape: tuple[int, int], *, xi: np.ndarray | None, yi: np.ndarray | None
) -> GridSpec:
    if xi is not None and yi is not None and xi.size > 1 and yi.size > 1:
        x_min, x_max = float(np.nanmin(xi)), float(np.nanmax(xi))
        y_min, y_max = float(np.nanmin(yi)), float(np.nanmax(yi))
        dy = (y_max - y_min) / float(max(shape[0] - 1, 1))
        dx = (x_max - x_min) / float(max(shape[1] - 1, 1))
        if dy > 0.0 and dx > 0.0:
            return GridSpec(
                shape=shape,
                spacing_m=(dy, dx),
                origin_m=(y_min, x_min),
                roi_mask=np.ones(shape, dtype=bool),
            )
    return make_grid(shape=shape, spacing_m=(1.0e-3, 1.0e-3))


def _geometry_from_xy_positions(
    positions_xy: np.ndarray, *, n_transducers: int
) -> GeometrySpec:
    positions_xy = np.asarray(positions_xy, dtype=float)
    if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
        raise ValueError("transducerPositionsXY must have shape (n, 2)")
    if n_transducers <= 0 or n_transducers > positions_xy.shape[0]:
        n_transducers = positions_xy.shape[0]
    indices = (
        np.linspace(0, positions_xy.shape[0] - 1, n_transducers).round().astype(int)
    )
    positions_yx = positions_xy[indices][:, [1, 0]]
    radius_m = float(np.median(np.linalg.norm(positions_yx, axis=1)))
    return GeometrySpec(
        type="ring",
        tx_pos_m=positions_yx,
        rx_pos_m=positions_yx.copy(),
        radius_m=radius_m,
    )


def _kwave_channel_metadata_from_handle(handle: Any) -> dict[str, Any] | None:
    required = ("C", "atten", "full_dataset", "transducerPositionsXY")
    if not all(name in handle for name in required):
        return None
    c_shape = tuple(int(v) for v in handle["C"].shape)
    atten_shape = tuple(int(v) for v in handle["atten"].shape)
    full_shape = tuple(int(v) for v in handle["full_dataset"].shape)
    pos_shape = tuple(int(v) for v in handle["transducerPositionsXY"].shape)
    if len(c_shape) != 2 or atten_shape != c_shape:
        return None
    if len(full_shape) != 3 or len(pos_shape) != 2 or pos_shape[1] != 2:
        return None
    return {
        "format": "kwave-channel-mat",
        "sound_speed_dataset": "C",
        "attenuation_dataset": "atten",
        "channel_dataset": "full_dataset",
        "geometry_dataset": "transducerPositionsXY",
        "sound_speed_shape": list(c_shape),
        "attenuation_shape": list(atten_shape),
        "channel_shape": list(full_shape),
        "geometry_shape": list(pos_shape),
    }


def _nbp_metadata_from_handle(handle: Any) -> dict[str, Any] | None:
    required = ("sos", "att", "den", "label", "type", "y")
    if not all(name in handle for name in required):
        return None
    sos_shape = tuple(int(v) for v in handle["sos"].shape)
    att_shape = tuple(int(v) for v in handle["att"].shape)
    den_shape = tuple(int(v) for v in handle["den"].shape)
    label_shape = tuple(int(v) for v in handle["label"].shape)
    if (
        len(sos_shape) != 2
        or att_shape != sos_shape
        or den_shape != sos_shape
        or label_shape != sos_shape
    ):
        return None
    type_code = _read_scalar(handle.get("type"))
    density_label = (
        chr(int(type_code))
        if type_code is not None and 0 <= int(type_code) <= 255
        else None
    )
    return {
        "format": "nbpslice2d-mat",
        "sound_speed_dataset": "sos",
        "attenuation_dataset": "att",
        "density_dataset": "den",
        "label_dataset": "label",
        "sound_speed_shape": list(sos_shape),
        "attenuation_shape": list(att_shape),
        "density_shape": list(den_shape),
        "label_shape": list(label_shape),
        "type_code": int(type_code) if type_code is not None else None,
        "density_label": density_label,
        "density_class": NBP_DENSITY_CLASSES.get(density_label or "", "unknown"),
        "attenuation_power_law_exponent": _read_scalar(handle.get("y")),
        "sound_speed_units": "mm/us",
        "attenuation_units": "dB/(MHz^y mm)",
        "density_units": "g/mm^3",
    }


def _select_nbp_zip_members(names: list[str], *, cases_per_type: int) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        path = Path(name)
        if path.suffix.lower() != ".mat":
            continue
        if any(part.startswith("__MACOSX") for part in path.parts):
            continue
        label = path.stem[:1].upper() or "unknown"
        grouped.setdefault(label, []).append(name)
    selected: list[str] = []
    for label in sorted(grouped):
        selected.extend(sorted(grouped[label])[:cases_per_type])
    return selected


def _decode_matlab_chars(dataset: Any) -> str | None:
    if dataset is None:
        return None
    try:
        values = np.asarray(dataset[()]).reshape(-1)
        return "".join(chr(int(value)) for value in values if int(value) != 0)
    except Exception:
        return None


def _read_scalar(dataset: Any) -> float | None:
    if dataset is None:
        return None
    try:
        return float(np.asarray(dataset[()]).reshape(-1)[0])
    except Exception:
        return None


def _safe_case_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "kwave_case"


def _largest_3d_dataset_name(handle: Any) -> str:
    candidates: list[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and len(obj.shape) == 3:
            size = int(np.prod(obj.shape))
            candidates.append((size, name))

    handle.visititems(visitor)
    if not candidates:
        raise ValueError("no 3-D dataset found in MAT/HDF5 file")
    return sorted(candidates, reverse=True)[0][1]


@contextmanager
def _open_speed_volume(path: Path, *, dataset_name: str | None = None):
    h5py = _h5py()
    try:
        with h5py.File(path, "r") as handle:
            name = dataset_name or _largest_3d_dataset_name(handle)
            yield {"name": name, "dataset": handle[name], "format": "matlab-v7.3-hdf5"}
            return
    except OSError:
        pass

    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{path} is not HDF5; scipy is required to read MATLAB v5 MAT speed volumes"
        ) from exc

    data = loadmat(path)
    name = dataset_name or _largest_3d_array_name(data)
    if name not in data:
        keys = [key for key in data if not key.startswith("__")]
        raise KeyError(f"{name!r} not found in {path}. Available keys: {keys}")
    yield {"name": name, "dataset": np.asarray(data[name]), "format": "matlab-v5"}


def _largest_3d_array_name(data: dict[str, Any]) -> str:
    candidates = [
        (int(np.prod(value.shape)), name)
        for name, value in data.items()
        if not name.startswith("__")
        and hasattr(value, "shape")
        and len(value.shape) == 3
    ]
    if not candidates:
        keys = [key for key in data if not key.startswith("__")]
        raise ValueError(f"no 3-D array found in MAT file; available keys: {keys}")
    return sorted(candidates, reverse=True)[0][1]


def _infer_speed_volume_sample_axis(shape: tuple[int, ...]) -> int:
    if len(shape) != 3:
        raise ValueError(f"speed dataset must be 3-D, got {shape}")
    if shape[1] == shape[2] and shape[0] != shape[1]:
        return 0
    if shape[0] == shape[1] and shape[2] != shape[0]:
        return 2
    return int(np.argmax(shape))


def _read_speed_volume_slice(dataset: Any, index: int, sample_axis: int) -> np.ndarray:
    if sample_axis == 0:
        return np.asarray(dataset[index, :, :])
    if sample_axis == 1:
        return np.asarray(dataset[:, index, :])
    if sample_axis == 2:
        return np.asarray(dataset[:, :, index])
    raise ValueError(f"invalid sample axis {sample_axis}")


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for MAT/HDF5 conversion") from exc
    return h5py
