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
from usctbench.core.stopping import BudgetExhausted
from usctbench.operators.ray_born import RayBornForward, RayBornOperator
from usctbench.solvers.least_squares import linear_cgls
from usctbench.solvers.nonlinear import nonlinear_least_squares


class RWaveAdapter:
    """Relinearized distorted-wave Born inversion, not a full r-Wave port.

    Complex pressure is required. Travel-time features cannot substitute for
    finite-frequency measurements. Background propagation uses either full
    volume-integral Green fields or the explicitly approximate Eikonal/WKB path.
    """

    name = "rwave_adapter"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return run_with_failure_capture(
            self.name, case, lambda: self._run(case, config)
        )

    def _run(self, case, config):
        p = config.parameters
        green_settings = {
            "green_backend": str(p.get("green_backend", "eikonal_wkb")),
            "green_solver_rtol": float(p.get("green_solver_rtol", 1e-7)),
            "green_solver_maxiter": p.get("green_solver_maxiter", 20),
        }
        mode = str(p.get("mode", "nonlinear"))
        if mode not in {"nonlinear", "fixed_background"}:
            raise ValueError("rwave mode must be nonlinear or fixed_background")
        initialization = str(p.get("initialization", "configured"))
        if initialization not in {"configured", "phase_cgls"}:
            raise ValueError("initialization must be configured or phase_cgls")
        if mode == "fixed_background" and initialization != "configured":
            raise ValueError("phase_cgls initialization requires nonlinear mode")
        initialization_iterations = p.get("initialization_iterations", 80)
        if (
            isinstance(initialization_iterations, (bool, np.bool_))
            or not isinstance(initialization_iterations, (int, np.integer))
            or initialization_iterations <= 0
        ):
            raise ValueError("initialization_iterations must be a positive integer")
        initialization_lambda = float(p.get("initialization_lambda", 0.02))
        initialization_smooth_mm = float(p.get("initialization_smooth_mm", 3.0))
        if (
            not np.isfinite(initialization_lambda)
            or initialization_lambda < 0
            or not np.isfinite(initialization_smooth_mm)
            or initialization_smooth_mm < 0
        ):
            raise ValueError(
                "initialization lambda/smoothing must be finite and nonnegative"
            )
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
        inverse_ppw = float(
            np.min(background)
            / (np.max(case.measurement.frequencies_hz) * max(case.grid.spacing_m))
        )
        if inverse_ppw < 4 and not coerce_bool(p.get("allow_underresolved", False)):
            raise ValueError(
                f"Ray-Born inverse grid underresolved ({inverse_ppw:.2f} pixels/wavelength < 4); refine the image grid, not just the k-Wave simulation grid"
            )
        source = p.get("source_spectrum", case.measurement.source_spectrum)
        if (
            source is None
            and case.measurement.water_reference is None
            and not coerce_bool(p.get("assume_unit_source", False))
        ):
            raise ValueError(
                "Ray-Born requires a calibrated source_spectrum or independent water_reference; assume_unit_source is for explicitly unit-source synthetic data only"
            )
        observed = np.asarray(case.measurement.freq_data)
        data_shape = (
            len(case.measurement.frequencies_hz),
            len(case.geometry.tx_pos_m),
            len(case.geometry.rx_pos_m),
        )
        if observed.shape != data_shape:
            raise ValueError(
                f"freq_data must have canonical (frequency, tx, rx) shape {data_shape}"
            )
        sign = case.metadata.get("frequency_convention", "exp(-i omega t)")
        if sign != "exp(-i omega t)":
            raise ValueError(
                "convert complex pressure to exp(-i omega t) convention before Ray-Born inversion"
            )
        pair_mask = (
            np.linalg.norm(
                case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
            )
            > 0
        )
        valid = np.broadcast_to(pair_mask, observed.shape).copy()
        if case.measurement.valid_mask is not None:
            valid &= case.measurement.valid_mask
        # ToF confidence uses the broadband trace and can leak a held-out
        # frequency into pressure inversion. Pressure weights need a separate
        # acquisition-noise definition; the default is uniform precision.
        weights = None
        if coerce_bool(p.get("use_feature_weights", False)):
            weights = case.measurement.ray_weights
            if weights is None:
                weights = case.measurement.feature_quality
            if config.parameters.get("evaluation", {}).get(
                "frequency_indices"
            ) or config.parameters.get("evaluation", {}).get("frequency_fraction"):
                raise ValueError(
                    "broadband ToF feature weights cannot be used with a pressure frequency holdout"
                )
        control = InversionControl(
            case,
            config,
            observed,
            default_iterations=int(
                p.get("outer_iterations", 4)
                if mode == "nonlinear"
                else p.get("inner_iterations", 30)
            ),
            weights=weights,
            valid_mask=valid,
            iteration_unit=(
                "Ray-Born outer step" if mode == "nonlinear" else "Ray-Born CGLS step"
            ),
        )
        bounds = speed_bounds(config)
        control.declare_update(
            (
                "delta_squared_slowness"
                if mode == "fixed_background"
                else "squared_slowness"
            ),
            "full_squared_slowness",
            "s^2/m^2",
            normalization=(
                None
                if mode == "fixed_background"
                else "norm(q_new-q_old)/norm(q_old); positive bounded squared slowness"
            ),
        )
        green_settings["budget_check"] = control.work.check_time
        if (
            not np.all(np.isfinite(background))
            or np.any(background < bounds[0])
            or np.any(background > bounds[1])
        ):
            raise ValueError(
                "background sound speed lies outside sound_speed_bounds_mps"
            )
        roi_only = coerce_bool(p.get("roi_update_only", False))
        roi = case.grid.roi_mask if roi_only else None

        def exhausted_result(exc):
            control.monitor.finish(exc.reason)
            _, metrics = control.output(1 / background**2)
            metrics.update(
                backend="native_ray_born",
                online_stopping=True,
                calibration_completed=False,
                surrogate_travel_time_backend=False,
            )
            add_image_metrics(metrics, background, case, c0)
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                sound_speed_mps=background.copy(),
                metrics=metrics,
            )

        try:
            operator = control.call(
                "setup_background",
                RayBornOperator,
                case.grid,
                case.geometry,
                case.measurement.frequencies_hz,
                background_sound_speed_mps=background,
                exterior_speed_mps=c0,
                source_spectrum=source,
                max_cache_bytes=int(p.get("max_cache_bytes", 128 * 1024**2)),
                **green_settings,
            )
        except BudgetExhausted as exc:
            return exhausted_result(exc)
        control.work.counts["setup_eikonal_source_solves"] = operator.eikonal_solves

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
            try:
                water_prediction = control.call(
                    "source_calibration", water_operator.background_data
                )
            except BudgetExhausted as exc:
                return exhausted_result(exc)
            source, calibration = fit_water_source(
                water_prediction,
                case.measurement.water_reference,
                valid_mask=operator.valid_pair_mask,
            )
            operator.source_spectrum = source.copy()
        try:
            offset = control.call("background_prediction", operator.background_data)
        except BudgetExhausted as exc:
            return exhausted_result(exc)
        damping = float(
            p.get("damping", float(p.get("regularization_lambda", 0.0)) ** 2)
        )
        if not np.isfinite(damping) or damping < 0:
            raise ValueError("damping must be finite and nonnegative")
        scaling = str(p.get("regularization_scaling", "absolute"))
        penalty_scale = 1.0
        if scaling == "relative_jacobian_diagonal":
            try:
                diagonal = control.call(
                    "normal_diagonal", operator.normal_diagonal, control.precision
                )
            except BudgetExhausted as exc:
                return exhausted_result(exc)
            active_diagonal = diagonal[roi] if roi is not None else diagonal.ravel()
            positive = active_diagonal[active_diagonal > 0]
            if not positive.size:
                raise ValueError("no illuminated training pixels")
            penalty_scale = float(np.median(positive))
            damping *= penalty_scale
        elif scaling != "absolute":
            raise ValueError(
                "regularization_scaling must be absolute or relative_jacobian_diagonal"
            )
        regularization = str(
            p.get("regularization", "laplacian" if mode == "nonlinear" else "identity")
        )
        length_wavelengths = float(p.get("regularization_length_wavelengths", 0.0))
        if not np.isfinite(length_wavelengths) or length_wavelengths < 0:
            raise ValueError(
                "regularization_length_wavelengths must be finite and nonnegative"
            )
        physical_length = None
        if length_wavelengths:
            if regularization not in {"laplacian", "roughness"}:
                raise ValueError(
                    "physical regularization length requires a Laplacian penalty"
                )
            train_frequency = np.any(control.split.train, axis=(1, 2))
            physical_length = (
                length_wavelengths
                * c0
                / np.max(case.measurement.frequencies_hz[train_frequency])
            )
            regularization = tuple(
                (physical_length / h) ** 2 for h in case.grid.spacing_m
            )
        if mode == "fixed_background":
            delta, metrics = linear_cgls(
                operator,
                control,
                initial=np.zeros(case.grid.shape),
                offset=offset,
                to_speed=to_speed,
                project=project,
                reference=operator.background_squared_slowness,
                damping=damping,
                regularization=regularization,
                roi=roi,
            )
            sound_speed = to_speed(delta)
        else:
            initial_model = operator.background_squared_slowness
            initialization_qc = None
            if initialization == "phase_cgls":
                from types import SimpleNamespace
                from scipy.ndimage import gaussian_filter
                from usctbench.data.phase_delay import phase_slope_delays
                from usctbench.operators.straight_ray import (
                    StraightRayProjector,
                )
                from usctbench.solvers.least_squares import normal_step

                water_data = case.measurement.water_reference
                if water_data is None:
                    raise ValueError(
                        "phase_cgls initialization requires independent water_reference"
                    )
                delays, precision, initialization_qc = phase_slope_delays(
                    observed,
                    water_data,
                    case.measurement.frequencies_hz,
                    operator.direct_distance,
                    control.split.train,
                    c0=c0,
                    bounds=bounds,
                    max_phase_rms=float(p.get("initialization_max_phase_rms", 0.2)),
                    min_amplitude_ratio=float(
                        p.get("initialization_min_amplitude_ratio", 0.05)
                    ),
                )
                if not np.any(precision):
                    raise ValueError(
                        "no valid training phase-delay channels for initialization"
                    )
                training = SimpleNamespace(
                    observed=delays,
                    precision=precision,
                    call=control.call,
                    work=control.work,
                )
                try:
                    straight = control.call(
                        "initialization_setup", StraightRayProjector.from_case, case
                    )
                    delta = normal_step(
                        straight,
                        np.where(precision > 0, np.nan_to_num(delays) * precision, 0),
                        np.zeros(case.grid.shape),
                        training,
                        iterations=initialization_iterations,
                        damping=initialization_lambda**2,
                        regularization="laplacian",
                        roi=roi,
                    )
                    sigma = (
                        initialization_smooth_mm * 1e-3 / np.array(case.grid.spacing_m)
                    )
                    delta = gaussian_filter(delta, sigma, mode="nearest")
                    slowness = np.clip(1 / c0 + delta, 1 / bounds[1], 1 / bounds[0])
                    if roi is not None:
                        slowness = np.where(roi, slowness, 1 / c0)
                    initial_model = slowness**2
                except BudgetExhausted as exc:
                    return exhausted_result(exc)
            forward = RayBornForward(
                case.grid,
                case.geometry,
                case.measurement.frequencies_hz,
                exterior_speed_mps=c0,
                source_spectrum=operator.source_spectrum,
                max_cache_bytes=int(p.get("max_cache_bytes", 128 * 1024**2)),
                **green_settings,
            )
            state, metrics = nonlinear_least_squares(
                forward,
                control,
                initial=initial_model,
                bounds=bounds,
                inner_iterations=p.get("inner_iterations", 12),
                inner_solver=str(p.get("inner_solver", "lsmr")),
                inner_options=p.get("inner_options", {}),
                damping=damping,
                regularization=regularization,
                roi=roi,
                step_length=float(p.get("step_length", 1)),
                smooth_sigma=float(p.get("smooth_sigma", 0)),
                max_update_mps=float(p.get("max_update_mps", 12)),
                max_backtracks=p.get("max_backtracks", 10),
                gradient_rtol=float(p.get("gradient_rtol", 1e-8)),
            )
            sound_speed = 1 / np.sqrt(state)
            metrics["initialization"] = initialization
            metrics["initialization_qc"] = initialization_qc
        if control.last_prediction is not None:
            selected_prediction = (
                control.best_prediction
                if control.policy.restore_best_validation
                and control.best_prediction is not None
                else control.last_prediction
            )
            # The incident field can dominate full-pressure relative residuals.
            # A second denominator measures the information-bearing contrast.
            metrics["contrast_evaluation"] = control.split.evaluate(
                selected_prediction - offset, observed - offset
            )
        metrics.update(
            {
                "backend": (
                    "native_relinearized_ray_born"
                    if mode == "nonlinear"
                    else "native_fixed_background_ray_born"
                ),
                "method_family": (
                    "distorted_born_volume_integral"
                    if mode == "nonlinear"
                    and green_settings["green_backend"] == "volume_integral"
                    else "finite_frequency_ray_born"
                ),
                "ray_born_linearization": True,
                "full_ray_born_solver": False,
                "full_upstream_rwave_port": False,
                "background_relinearization": mode == "nonlinear",
                "nonlinear_prediction_model": (
                    green_settings["green_backend"]
                    if mode == "nonlinear"
                    else "fixed_background_born"
                ),
                "green_backend": green_settings["green_backend"],
                "uses_geometrical_ray_green_approximation": green_settings[
                    "green_backend"
                ]
                == "eikonal_wkb",
                "surrogate_travel_time_backend": False,
                "green_method": operator.green_method,
                "green_method_scope": "initial_background",
                "model_parameter": "squared_slowness_s2_per_m2",
                "frequency_convention": "exp(-i omega t)",
                "source_spectrum_assumed_unit": source is None,
                "source_calibration": calibration,
                "pressure_uses_feature_weights": weights is not None,
                "inverse_grid_points_per_wavelength": inverse_ppw,
                "inverse_grid_resolved": inverse_ppw >= 4,
                "regularization_scaling": scaling,
                "regularization_length_m": physical_length,
                "regularization_length_uses_training_band_only": physical_length
                is not None,
                "penalty_scale": penalty_scale,
                "effective_damping": damping,
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
        description="Native relinearized Ray-Born complex-pressure inversion.",
        tags=("ray-born", "frequency", "scattering"),
        replace=replace,
    )


__all__ = ["RWaveAdapter", "register_rwave_algorithm"]
