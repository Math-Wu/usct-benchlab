"""Nonlinear first-arrival Eikonal inversion; no straight-ray fallback."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms._control import InversionControl, add_image_metrics
from usctbench.algorithms.ray import (
    _gaussian_smooth,
    configured_ray_weights,
    reference_sound_speed,
    run_with_failure_capture,
    speed_bounds,
)
from usctbench.core.config import coerce_bool
from usctbench.core.registry import register_algorithm
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase
from usctbench.core.stopping import BudgetExhausted
from usctbench.operators.forward.eikonal import EikonalForward
from usctbench.operators.model_space import (
    BilinearBasis,
    FineGridRegularizer,
    ReducedLinearOperator,
)
from usctbench.solvers.least_squares import normal_step, regularizer


class BentRayGNAdapter:
    """Bounded, regularized Gauss-Newton inversion of absolute or differential TOF."""

    name = "bent_ray_gn"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return run_with_failure_capture(
            self.name, case, lambda: self._run(case, config)
        )

    def _run(self, case, config):
        p = config.parameters
        c0 = reference_sound_speed(case, config)
        bounds = speed_bounds(config)
        forward = EikonalForward(
            case.grid,
            case.geometry,
            background_speed_mps=c0,
            spatial_order=p.get("eikonal_order", 1),
        )
        distance = np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None, :], axis=-1
        )
        differential = case.measurement.delta_tof_s is not None
        observed = (
            case.measurement.delta_tof_s if differential else case.measurement.tof_s
        )
        if observed is None:
            raise ValueError("bent_ray_gn requires measured delta_tof_s or tof_s")
        observed = np.asarray(observed, dtype=float)
        valid = distance > 0
        if case.measurement.valid_mask is not None:
            valid &= case.measurement.valid_mask
        weights = configured_ray_weights(case, forward, valid.ravel(), config).reshape(
            observed.shape
        )
        control = InversionControl(
            case,
            config,
            observed,
            default_iterations=int(p.get("outer_iterations", 4)),
            weights=weights,
            valid_mask=valid,
            iteration_unit="Gauss-Newton outer step",
        )
        inner = int(p.get("inner_iterations", 16))
        if inner < 1:
            raise ValueError("inner_iterations must be positive")
        kind = str(p.get("regularization", "laplacian"))
        damping = float(
            p.get("damping", float(p.get("regularization_lambda", 2e-2)) ** 2)
        )
        step = float(p.get("step_length", 1.0))
        sigma = float(p.get("smooth_sigma", 0.0))
        if (
            not np.isfinite([damping, step, sigma]).all()
            or damping < 0
            or step <= 0
            or sigma < 0
        ):
            raise ValueError(
                "damping/smoothing must be nonnegative; step_length positive"
            )
        if str(p.get("robust_loss", "none")).lower() != "none":
            raise ValueError(
                "native Eikonal GN currently supports robust_loss=none; no surrogate fallback"
            )
        line_search = coerce_bool(p.get("line_search", True))
        roi_only = coerce_bool(p.get("roi_update_only", False))
        roi = case.grid.roi_mask if roi_only else None
        s0 = np.full(case.grid.shape, 1 / c0)
        initial_speed = np.broadcast_to(
            np.asarray(p.get("initial_sound_speed_mps", c0), dtype=float),
            case.grid.shape,
        )
        if not np.isfinite(initial_speed).all() or np.any(initial_speed <= 0):
            raise ValueError("initial_sound_speed_mps must be finite and positive")
        s = np.clip(1 / initial_speed, 1 / bounds[1], 1 / bounds[0])
        if roi is not None:
            s = np.where(roi, s, s0)
        initial = s.copy()
        offset = distance / c0 if differential else np.zeros_like(distance)
        initialization = str(p.get("initialization", "configured"))
        if initialization not in {"configured", "cgls"}:
            raise ValueError("initialization must be configured or cgls")
        initialization_iterations = p.get("initialization_iterations", 80)
        if (
            isinstance(initialization_iterations, bool)
            or not isinstance(initialization_iterations, (int, np.integer))
            or initialization_iterations <= 0
        ):
            raise ValueError("initialization_iterations must be a positive integer")

        basis = None
        coefficients = None
        solver_kind = kind
        if p.get("model_grid_shape") is not None:
            if roi is not None:
                raise ValueError("reduced model currently requires no inversion ROI")
            if initialization != "configured" or not np.all(initial_speed == c0):
                raise ValueError(
                    "reduced Bent currently requires configured water initialization"
                )
            if not bounds[0] <= c0 <= bounds[1]:
                raise ValueError("reference speed must lie within sound-speed bounds")
            basis = BilinearBasis(case.grid, p["model_grid_shape"])
            coefficients = np.zeros(basis.shape)
            solver_kind = FineGridRegularizer(basis, kind)

        def objective(state, prediction):
            active = control.split.train | control.split.validation
            if not np.isfinite(prediction[active]).all():
                raise FloatingPointError("non-finite Eikonal prediction on active data")
            residual = np.where(
                control.split.train, control.safe_observed - prediction, 0
            )
            reg = regularizer(state - s0, kind)
            return float(
                0.5 * np.sum(control.precision * residual**2)
                + 0.5 * damping * np.vdot(reg, reg).real
            )

        try:
            if initialization == "cgls":
                from usctbench.operators.forward.straight_ray import (
                    StraightRayProjector,
                )

                initializer = control.call(
                    "initialization_setup", StraightRayProjector.from_case, case
                )
                water_prediction = (
                    np.zeros_like(observed) if differential else distance / c0
                )
                seed_update = normal_step(
                    initializer,
                    control.weighted_residual(water_prediction),
                    np.zeros_like(s),
                    control,
                    iterations=initialization_iterations,
                    damping=damping,
                    regularization=kind,
                    roi=roi,
                )
                s = np.clip(s0 + seed_update, 1 / bounds[1], 1 / bounds[0])
                if roi is not None:
                    s = np.where(roi, s, s0)
            lin = control.call("forward", forward.linearize, s)
            prediction = lin.value.reshape(observed.shape) - offset
            cost = objective(s, prediction)
            control.observe(0, s, prediction, objective=cost, sound_speed=1 / s)
            for iteration in range(1, control.policy.max_iterations + 1):
                if control.monitor.reason:
                    break
                jacobian = (
                    lin.jacobian
                    if basis is None
                    else ReducedLinearOperator(lin.jacobian, basis)
                )
                update = normal_step(
                    jacobian,
                    control.weighted_residual(prediction),
                    s - s0 if basis is None else coefficients,
                    control,
                    iterations=inner,
                    damping=damping,
                    regularization=solver_kind,
                    roi=roi,
                )
                if sigma > 0:
                    if basis is None:
                        update = _gaussian_smooth(update, sigma)
                    else:
                        from scipy.ndimage import gaussian_filter

                        update = gaussian_filter(
                            update,
                            tuple(
                                sigma * k / n
                                for k, n in zip(basis.shape, case.grid.shape)
                            ),
                            mode="nearest",
                        )
                if roi is not None:
                    update = np.where(roi, update, 0)
                accepted = False
                for trial in range(10 if line_search else 1):
                    if basis is None:
                        candidate = s + step * 0.5**trial * update
                        candidate = np.clip(candidate, 1 / bounds[1], 1 / bounds[0])
                    else:
                        candidate_coefficients = np.clip(
                            coefficients + step * 0.5**trial * update,
                            1 / bounds[1] - 1 / c0,
                            1 / bounds[0] - 1 / c0,
                        )
                        candidate = s0 + basis.forward(candidate_coefficients)
                    if roi is not None:
                        candidate = np.where(roi, candidate, s0)
                    new_lin = control.call("line_search", forward.linearize, candidate)
                    new_prediction = new_lin.value.reshape(observed.shape) - offset
                    new_cost = objective(candidate, new_prediction)
                    if not line_search or new_cost <= cost:
                        accepted = True
                        break
                if not accepted:
                    control.monitor.finish("line_search_failed")
                    break
                relative = float(np.linalg.norm(candidate - s) / np.linalg.norm(s))
                s, lin, prediction, cost = candidate, new_lin, new_prediction, new_cost
                if basis is not None:
                    coefficients = candidate_coefficients
                control.observe(
                    iteration,
                    s,
                    prediction,
                    objective=cost,
                    update_relative=relative,
                    sound_speed=1 / s,
                )
            if not control.monitor.reason:
                control.monitor.finish("max_iterations")
        except BudgetExhausted as exc:
            control.monitor.finish(exc.reason)
        except FloatingPointError:
            control.monitor.finish("numerical_failure")
        s, metrics = control.output(initial)
        sound_speed = 1 / s
        metrics.update(
            {
                "backend": "native_eikonal_fast_marching",
                "eikonal_order": forward.spatial_order,
                "model_parameterization": (
                    {"kind": "pixels"} if basis is None else basis.metadata()
                ),
                "true_bent_ray": True,
                "uses_true_bent_rays": True,
                "surrogate_travel_time_backend": False,
                "full_external_eikonal_solver": False,
                "method_family": "first_arrival_eikonal",
                "linearization_parameter": "slowness_s_per_m",
                "inner_iterations": inner,
                "initialization": initialization,
                "initialization_training_only": initialization == "cgls",
                "initialization_iterations": (
                    initialization_iterations if initialization == "cgls" else 0
                ),
                "smoothing_applied_to": "update_direction_not_accumulated_image",
                "regularization": kind,
                "regularization_lambda_squared": damping,
                "line_search": line_search,
                "roi_update_only": roi_only,
                "roi_laplacian": coerce_bool(p.get("roi_laplacian", False)),
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


def register_bent_ray_algorithm(*, replace: bool = False) -> None:
    register_algorithm(
        "bent_ray_gn",
        BentRayGNAdapter,
        description="Native first-arrival Eikonal Gauss-Newton inversion.",
        tags=("travel-time", "refraction", "eikonal"),
        replace=replace,
    )


__all__ = ["BentRayGNAdapter", "register_bent_ray_algorithm"]
