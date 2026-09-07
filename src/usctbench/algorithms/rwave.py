"""Finite-frequency Ray-Born inversion of complex pressure measurements."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms._control import InversionControl, add_image_metrics
from usctbench.algorithms.ray import (
    reference_sound_speed,
    run_with_failure_capture,
    speed_bounds,
)
from usctbench.core.config import coerce_bool
from usctbench.data.calibration import fit_water_source
from usctbench.core.registry import register_algorithm
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase
from usctbench.operators.forward.ray_born import RayBornOperator
from usctbench.solvers.least_squares import linear_cgls


class RWaveAdapter:
    """Fixed-background distorted-wave Born inversion, not a full r-Wave port.

    Complex pressure is required. Travel-time features cannot be substituted for
    finite-frequency measurements. Heterogeneous backgrounds use Eikonal/WKB
    Green functions; a uniform background uses the exact outgoing 2-D Hankel G.
    """

    name = "rwave_adapter"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return run_with_failure_capture(
            self.name, case, lambda: self._run(case, config)
        )

    def _run(self, case, config):
        p = config.parameters
        if (
            case.measurement.freq_data is None
            or case.measurement.frequencies_hz is None
        ):
            raise ValueError(
                "rwave_adapter requires complex freq_data and frequencies_hz; TOF-only cases are not Ray-Born data"
            )
        c0 = reference_sound_speed(case, config)
        background = np.broadcast_to(
            np.asarray(p.get("background_sound_speed_mps", c0), dtype=float),
            case.grid.shape,
        )
        source = p.get("source_spectrum", None)
        operator = RayBornOperator(
            case.grid,
            case.geometry,
            case.measurement.frequencies_hz,
            background_sound_speed_mps=background,
            exterior_speed_mps=c0,
            source_spectrum=source,
            max_cache_bytes=int(p.get("max_cache_bytes", 128 * 1024**2)),
        )
        observed = np.asarray(case.measurement.freq_data)
        if observed.shape != operator.data_shape:
            raise ValueError(
                f"freq_data must have canonical (frequency, tx, rx) shape {operator.data_shape}"
            )
        sign = case.metadata.get("frequency_convention", "exp(-i omega t)")
        if sign != "exp(-i omega t)":
            raise ValueError(
                "convert complex pressure to exp(-i omega t) convention before Ray-Born inversion"
            )
        valid = operator.valid_pair_mask.copy()
        if case.measurement.valid_mask is not None:
            valid &= case.measurement.valid_mask
        weights = case.measurement.ray_weights
        if weights is None:
            weights = case.measurement.feature_quality
        control = InversionControl(
            case,
            config,
            observed,
            default_iterations=int(p.get("inner_iterations", 30)),
            weights=weights,
            valid_mask=valid,
            iteration_unit="Ray-Born CGLS step",
        )
        control.work.counts["setup_eikonal_source_solves"] = operator.eikonal_solves
        bounds = speed_bounds(config)
        if np.any(background < bounds[0]) or np.any(background > bounds[1]):
            raise ValueError(
                "background sound speed lies outside sound_speed_bounds_mps"
            )
        roi_only = coerce_bool(p.get("roi_update_only", False))
        roi = case.grid.roi_mask if roi_only else None

        def project(delta):
            value = (
                np.clip(
                    delta + operator.background_squared_slowness,
                    1 / bounds[1] ** 2,
                    1 / bounds[0] ** 2,
                )
                - operator.background_squared_slowness
            )
            return np.where(roi, value, 0) if roi is not None else value

        def to_speed(delta):
            return 1 / np.sqrt(operator.background_squared_slowness + delta)

        calibration = {
            "method": "configured_source" if source is not None else "unit_source",
            "specimen_data_used": False,
        }
        # An independently acquired water trace identifies source amplitude and
        # phase. Scale the Jacobian too; replacing only the additive background
        # is incorrect whenever the actual source differs from unity.
        if case.measurement.water_reference is not None and source is None:
            water_operator = RayBornOperator(
                case.grid,
                case.geometry,
                case.measurement.frequencies_hz,
                background_sound_speed_mps=c0,
                exterior_speed_mps=c0,
                max_cache_bytes=0,
            )
            source, calibration = fit_water_source(
                water_operator.background_data(),
                case.measurement.water_reference,
                valid_mask=operator.valid_pair_mask,
            )
            operator.source_spectrum = source.copy()
        offset = operator.background_data()
        damping = float(
            p.get("damping", float(p.get("regularization_lambda", 0.0)) ** 2)
        )
        if not np.isfinite(damping) or damping < 0:
            raise ValueError("damping must be finite and nonnegative")
        delta, metrics = linear_cgls(
            operator,
            control,
            initial=np.zeros(case.grid.shape),
            offset=offset,
            to_speed=to_speed,
            project=project,
            reference=operator.background_squared_slowness,
            damping=damping,
            regularization=str(p.get("regularization", "identity")),
            roi=roi,
        )
        sound_speed = to_speed(delta)
        metrics.update(
            {
                "backend": "native_fixed_background_ray_born",
                "method_family": "finite_frequency_single_scattering",
                "ray_born_linearization": True,
                "full_ray_born_solver": False,
                "full_upstream_rwave_port": False,
                "surrogate_travel_time_backend": False,
                "green_method": operator.green_method,
                "model_parameter": "squared_slowness_s2_per_m2",
                "frequency_convention": "exp(-i omega t)",
                "source_spectrum_assumed_unit": source is None,
                "source_calibration": calibration,
                "roi_update_only": roi_only,
                "ground_truth_used_for_initialization": False,
            }
        )
        add_image_metrics(metrics, sound_speed, case, c0)
        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            sound_speed_mps=sound_speed,
            metrics=metrics,
        )


def register_rwave_algorithm(*, replace: bool = False) -> None:
    register_algorithm(
        "rwave_adapter",
        RWaveAdapter,
        description="Native fixed-background Ray-Born complex-pressure inversion.",
        tags=("ray-born", "frequency", "scattering"),
        replace=replace,
    )


__all__ = ["RWaveAdapter", "register_rwave_algorithm"]
