"""Opt-in online FWI control using the upstream exported Helmholtz matrix.

The production MATLAB driver/result import remains the default. This mode uses
the same PDE matrix but a Python regularized GN driver, frozen stencil bounds,
independent source calibration, and portable CPU sparse solves. It is not claimed
to reproduce the production filtered Polak-Ribiere/FR trajectory or GPU runtime.
"""

import numpy as np

from usctbench.algorithms._control import InversionControl, add_image_metrics
from usctbench.algorithms.ray import reference_sound_speed, speed_bounds
from usctbench.core.config import coerce_bool
from usctbench.core.schema import ReconstructionResult
from usctbench.core.stopping import BudgetExhausted
from usctbench.data.calibration import fit_water_source
from usctbench.operators.forward.fwi import MatlabHelmholtzForward
from usctbench.solvers.nonlinear import nonlinear_least_squares


def run_controlled(case, config):
    p = config.parameters
    data = case.measurement.freq_data
    if data is None or case.measurement.frequencies_hz is None:
        raise ValueError("controlled FWI requires complex freq_data and frequencies_hz")
    if (
        case.metadata.get("frequency_convention", "exp(-i omega t)")
        != "exp(-i omega t)"
    ):
        raise ValueError("controlled FWI requires exp(-i omega t) pressure convention")
    c0, bounds = reference_sound_speed(case, config), speed_bounds(config)
    control = InversionControl(
        case,
        config,
        data,
        default_iterations=int(p.get("outer_iterations", 5)),
        weights=case.measurement.ray_weights,
        valid_mask=case.measurement.valid_mask,
        iteration_unit="full-wave GN outer step",
    )
    source = p.get("discrete_source_spectrum")
    control.declare_update(
        "squared_slowness",
        "full_squared_slowness",
        "s^2/m^2",
        normalization="norm(q_new-q_old)/norm(q_old); positive bounded squared slowness",
    )
    if (
        source is None
        and case.measurement.water_reference is None
        and not coerce_bool(p.get("assume_unit_source", False))
    ):
        raise ValueError(
            "controlled FWI requires independent water_reference or discrete_source_spectrum; analytic Ray-Born source factors are not interchangeable"
        )
    forward = MatlabHelmholtzForward(
        case.grid,
        case.geometry,
        case.measurement.frequencies_hz,
        functions_path=p["functions_path"],
        matlab=p.get("matlab_bin", "matlab"),
        pml_m=float(p.get("pml_m", 0.009)),
        pml_strength=float(p.get("pml_strength", 10)),
        stencil_speed_bounds=bounds,
        source_spectrum=source,
        timeout_s=float(p.get("operator_timeout_s", 180)),
    )
    initial_speed = np.broadcast_to(
        np.asarray(p.get("initial_sound_speed_mps", c0), float), case.grid.shape
    )
    if (
        np.any(initial_speed < bounds[0])
        or np.any(initial_speed > bounds[1])
        or not np.all(np.isfinite(initial_speed))
    ):
        raise ValueError("initial speed must be finite and inside physical bounds")
    initial = 1 / initial_speed**2
    calibration = {
        "method": "configured_discrete_source" if source is not None else "unit_source"
    }
    try:
        if source is None and case.measurement.water_reference is not None:
            water = control.call(
                "source_calibration",
                forward.forward,
                np.full(case.grid.shape, 1 / c0**2),
            )
            forward.source_spectrum, calibration = fit_water_source(
                water,
                case.measurement.water_reference,
                valid_mask=case.measurement.valid_mask,
            )
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    if control.monitor.reason is None:
        state, metrics = nonlinear_least_squares(
            forward,
            control,
            initial=initial,
            bounds=bounds,
            inner_iterations=p.get("inner_iterations", 4),
            damping=float(p.get("regularization_lambda", 0)) ** 2,
            regularization=p.get("regularization", "laplacian"),
            roi=(
                case.grid.roi_mask
                if coerce_bool(p.get("roi_update_only", True))
                else None
            ),
            max_update_mps=float(p.get("max_update_mps", 12)),
            step_length=float(p.get("step_length", 1)),
            max_backtracks=p.get("max_backtracks", 8),
        )
    else:
        state, metrics = control.output(initial)
    speed = 1 / np.sqrt(state)
    metrics.update(
        backend="external_wust_matrix_controlled_gn",
        online_stopping=True,
        production_driver_trajectory_reproduced=False,
        source_calibration=calibration,
        frequency_convention="exp(-i omega t)",
        model_parameter="squared_slowness_s2_per_m2",
        derivative_kind="exact_discrete_fixed_stencil",
        pde_linear_solver="scipy_superlu_cpu",
    )
    add_image_metrics(metrics, speed, case, c0)
    return ReconstructionResult(
        algorithm="fwi_kwave_adapter",
        case_id=case.case_id,
        sound_speed_mps=speed,
        metrics=metrics,
    )
