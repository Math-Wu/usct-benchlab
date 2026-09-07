"""Shared schemas for USCT benchmark cases and reconstruction results."""

from __future__ import annotations

from typing import Any

from usctbench.core.compat import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GeometryType(StrEnum):
    """Supported acquisition geometry families."""

    RING = "ring"
    LINEAR = "linear"
    CUSTOM = "custom"


class MeasurementDomain(StrEnum):
    """Supported measurement representations."""

    FREQUENCY = "frequency"
    TIME = "time"
    FEATURES = "features"


class ResultStatus(StrEnum):
    """Outcome state for an algorithm run."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)


def _required_array(value: Any) -> np.ndarray:
    return np.asarray(value)


class _ArrayModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GridSpec(_ArrayModel):
    """Spatial grid for image-domain quantities."""

    shape: tuple[int, int]
    spacing_m: tuple[float, float]
    origin_m: tuple[float, float] = (0.0, 0.0)
    roi_mask: np.ndarray | None = None

    @field_validator("roi_mask", mode="before")
    @classmethod
    def _coerce_roi_mask(cls, value: Any) -> np.ndarray | None:
        array = _optional_array(value)
        if array is not None:
            array = array.astype(bool, copy=False)
        return array

    @model_validator(mode="after")
    def _validate_grid(self) -> "GridSpec":
        if len(self.shape) != 2 or any(size <= 0 for size in self.shape):
            raise ValueError("grid.shape must be two positive integers")
        if len(self.spacing_m) != 2 or any(spacing <= 0 for spacing in self.spacing_m):
            raise ValueError("grid.spacing_m must contain positive meter spacing")
        if len(self.origin_m) != 2:
            raise ValueError("grid.origin_m must contain two coordinates")
        if self.roi_mask is not None and self.roi_mask.shape != self.shape:
            raise ValueError("grid.roi_mask must match grid.shape")
        return self


class GeometrySpec(_ArrayModel):
    """Transducer positions and geometry metadata."""

    type: GeometryType = GeometryType.RING
    tx_pos_m: np.ndarray
    rx_pos_m: np.ndarray
    radius_m: float | None = None

    @field_validator("tx_pos_m", "rx_pos_m", mode="before")
    @classmethod
    def _coerce_position_array(cls, value: Any) -> np.ndarray:
        return _required_array(value).astype(float, copy=False)

    @model_validator(mode="after")
    def _validate_positions(self) -> "GeometrySpec":
        for name, value in (("tx_pos_m", self.tx_pos_m), ("rx_pos_m", self.rx_pos_m)):
            if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0:
                raise ValueError(f"geometry.{name} must have shape (n, 2)")
        if self.radius_m is not None and self.radius_m <= 0:
            raise ValueError("geometry.radius_m must be positive when provided")
        return self


class MeasurementSpec(_ArrayModel):
    """Measured data or extracted algorithm-ready features."""

    domain: MeasurementDomain
    frequencies_hz: np.ndarray | None = None
    freq_data: np.ndarray | None = None
    time_data: np.ndarray | None = None
    water_reference: np.ndarray | None = None
    source_wavelet: np.ndarray | None = None
    time_axis_s: np.ndarray | None = None
    tof_s: np.ndarray | None = None
    delta_tof_s: np.ndarray | None = None
    tof_first_arrival_s: np.ndarray | None = None
    tof_xcorr_s: np.ndarray | None = None
    phase_slope_delay_s: np.ndarray | None = None
    log_amp: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    feature_quality: np.ndarray | None = None
    ray_weights: np.ndarray | None = None

    @field_validator(
        "frequencies_hz",
        "freq_data",
        "time_data",
        "water_reference",
        "source_wavelet",
        "time_axis_s",
        "tof_s",
        "delta_tof_s",
        "tof_first_arrival_s",
        "tof_xcorr_s",
        "phase_slope_delay_s",
        "log_amp",
        "feature_quality",
        "ray_weights",
        mode="before",
    )
    @classmethod
    def _coerce_optional_arrays(cls, value: Any) -> np.ndarray | None:
        return _optional_array(value)

    @field_validator("valid_mask", mode="before")
    @classmethod
    def _coerce_valid_mask(cls, value: Any) -> np.ndarray | None:
        array = _optional_array(value)
        if array is not None:
            array = array.astype(bool, copy=False)
        return array

    @model_validator(mode="after")
    def _validate_measurement(self) -> "MeasurementSpec":
        if self.frequencies_hz is not None:
            if self.frequencies_hz.ndim != 1 or self.frequencies_hz.size == 0:
                raise ValueError(
                    "measurement.frequencies_hz must be a non-empty 1-D array"
                )
            if np.any(self.frequencies_hz <= 0):
                raise ValueError(
                    "measurement.frequencies_hz must be positive Hz values"
                )
        if self.domain == MeasurementDomain.FREQUENCY and self.freq_data is None:
            raise ValueError("frequency-domain measurements require freq_data")
        if self.domain == MeasurementDomain.TIME and self.time_data is None:
            raise ValueError("time-domain measurements require time_data")
        if self.domain == MeasurementDomain.FEATURES:
            has_feature = any(
                value is not None
                for value in (
                    self.tof_s,
                    self.delta_tof_s,
                    self.tof_first_arrival_s,
                    self.tof_xcorr_s,
                    self.phase_slope_delay_s,
                    self.log_amp,
                    self.ray_weights,
                )
            )
            if not has_feature:
                raise ValueError(
                    "feature-domain measurements require tof_s, delta_tof_s, or log_amp"
                )
        return self


class GroundTruthSpec(_ArrayModel):
    """Optional image-domain ground truth for synthetic or labeled cases."""

    sound_speed_mps: np.ndarray | None = None
    attenuation_np_per_m: np.ndarray | None = None
    density_kg_per_m3: np.ndarray | None = None

    @field_validator(
        "sound_speed_mps", "attenuation_np_per_m", "density_kg_per_m3", mode="before"
    )
    @classmethod
    def _coerce_ground_truth_arrays(cls, value: Any) -> np.ndarray | None:
        return _optional_array(value)


class USCTCase(_ArrayModel):
    """Complete input case consumed by benchmark algorithms."""

    case_id: str
    grid: GridSpec
    geometry: GeometrySpec
    measurement: MeasurementSpec
    ground_truth: GroundTruthSpec = Field(default_factory=GroundTruthSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_image_shapes(self) -> "USCTCase":
        expected_shape = self.grid.shape
        arrays = {
            "ground_truth.sound_speed_mps": self.ground_truth.sound_speed_mps,
            "ground_truth.attenuation_np_per_m": self.ground_truth.attenuation_np_per_m,
            "ground_truth.density_kg_per_m3": self.ground_truth.density_kg_per_m3,
        }
        for name, value in arrays.items():
            if value is not None and value.shape != expected_shape:
                raise ValueError(f"{name} must match grid.shape")
        self._validate_feature_measurement_shapes()
        return self

    def _validate_feature_measurement_shapes(self) -> None:
        expected_shape = (
            int(self.geometry.tx_pos_m.shape[0]),
            int(self.geometry.rx_pos_m.shape[0]),
        )
        arrays = {
            "measurement.tof_s": self.measurement.tof_s,
            "measurement.delta_tof_s": self.measurement.delta_tof_s,
            "measurement.tof_first_arrival_s": self.measurement.tof_first_arrival_s,
            "measurement.tof_xcorr_s": self.measurement.tof_xcorr_s,
            "measurement.phase_slope_delay_s": self.measurement.phase_slope_delay_s,
            "measurement.log_amp": self.measurement.log_amp,
            "measurement.valid_mask": self.measurement.valid_mask,
            "measurement.feature_quality": self.measurement.feature_quality,
            "measurement.ray_weights": self.measurement.ray_weights,
        }
        for name, value in arrays.items():
            if value is not None and value.shape != expected_shape:
                raise ValueError(f"{name} must match (n_tx, n_rx)={expected_shape}")


class AlgorithmConfig(_ArrayModel):
    """Algorithm configuration passed to registry entries."""

    name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReconstructionResult(_ArrayModel):
    """Standard output written by every algorithm run."""

    algorithm: str
    case_id: str
    sound_speed_mps: np.ndarray | None = None
    attenuation_np_per_m: np.ndarray | None = None
    reflectivity: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    runtime_s: float = 0.0
    status: ResultStatus = ResultStatus.SUCCESS
    failure_reason: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "sound_speed_mps",
        "attenuation_np_per_m",
        "reflectivity",
        "uncertainty",
        mode="before",
    )
    @classmethod
    def _coerce_result_arrays(cls, value: Any) -> np.ndarray | None:
        return _optional_array(value)

    @model_validator(mode="after")
    def _validate_result(self) -> "ReconstructionResult":
        if self.runtime_s < 0:
            raise ValueError("runtime_s must be non-negative")
        if self.status != ResultStatus.SUCCESS and not self.failure_reason:
            raise ValueError("failed or skipped results must include failure_reason")
        return self
