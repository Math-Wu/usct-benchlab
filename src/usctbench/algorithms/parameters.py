"""Authoritative native parameter models; constraints are numerical, not presets.

Null defaults denote case-dependent values and are omitted at the solver boundary.
Budgets and evaluation policies are validated separately in ``configuration``.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

Positive = Annotated[float, Field(strict=True, gt=0)]
Nonnegative = Annotated[float, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Bounds = tuple[Positive, Positive]


def parameter(default, description, units="1", exposure="advanced", **kwargs):
    return Field(
        default=default,
        description=description,
        json_schema_extra={"units": units, "exposure": exposure},
        **kwargs,
    )


class Parameters(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        values = dict(data)
        aliases = {
            "lambda": "regularization_lambda",
            "ray_weight_min": "min_ray_weight",
            "ray_weight_threshold": "min_ray_weight",
            "roi_aware_laplacian": "roi_laplacian",
        }
        if "damping" in values:
            damping = values.pop("damping")
            if (
                isinstance(damping, bool)
                or not isinstance(damping, (int, float))
                or not math.isfinite(damping)
                or damping < 0
            ):
                raise ValueError(
                    "damping must be a finite nonnegative squared regularization coefficient"
                )
            root = math.sqrt(damping)
            for alias in ("regularization_lambda", "lambda"):
                if alias in values and not math.isclose(
                    values[alias], root, rel_tol=1e-12, abs_tol=0
                ):
                    raise ValueError(
                        "conflicting damping and regularization_lambda/lambda"
                    )
            values["regularization_lambda"] = root
        for old, new in aliases.items():
            if old in values:
                value = values.pop(old)
                if new in values and values[new] != value:
                    raise ValueError(f"conflicting aliases: {old} and {new}")
                values[new] = value
        if "regularization" in values:
            values["regularization"] = {"l2": "identity", "roughness": "laplacian"}.get(
                values["regularization"], values["regularization"]
            )
        if "robust_loss" in values:
            values["robust_loss"] = {"irls": "huber", "huber_irls": "huber"}.get(
                values["robust_loss"], values["robust_loss"]
            )
        return values

    @model_validator(mode="after")
    def ordered_bounds(self):
        bounds = getattr(self, "sound_speed_bounds_mps", None)
        if bounds is not None and bounds[0] >= bounds[1]:
            raise ValueError("sound_speed_bounds_mps requires positive low < high")
        return self

    def backend_parameters(self):
        return self.model_dump(exclude_none=True)


class SoundSpeedParameters(Parameters):
    reference_sound_speed_mps: Positive | None = parameter(
        None,
        "Measurement reference; case metadata, otherwise 1500 m/s.",
        "m/s",
        "internal",
    )
    sound_speed_bounds_mps: Bounds = parameter(
        (1300.0, 1700.0),
        "Ordered numerical projection bounds, not a calibrated physiological range.",
        "m/s",
        "agent",
    )
    roi_update_only: bool = parameter(
        False, "Restrict updates to the supplied ROI, if present."
    )


class RayParameters(SoundSpeedParameters):
    model_grid_shape: tuple[PositiveInt, PositiveInt] | None = parameter(
        None,
        "Optional coefficient grid [row=y, col=x]; null uses the case grid.",
        "pixels",
    )
    min_ray_weight: Annotated[float, Field(strict=True, ge=0, le=1)] = parameter(
        0.0, "Reject ray confidence below this threshold."
    )
    ray_weight_power: Positive = parameter(
        1.0, "Exponent applied to confidence weights."
    )


class CGLSParameters(RayParameters):
    projector_backend: Literal["csr", "reference"] = parameter(
        "csr", "Equivalent straight-ray implementation.", exposure="internal"
    )
    regularization: Literal["identity", "laplacian"] = parameter(
        "identity", "Penalty operator L in lambda^2 ||L x||^2.", exposure="agent"
    )
    regularization_lambda: Nonnegative = parameter(
        0.0,
        "Penalty amplitude; legacy damping equals its square.",
        "objective/operator dependent",
    )
    robust_loss: Literal["none", "huber"] = parameter(
        "none", "Quadratic residual or Huber IRLS.", exposure="agent"
    )
    huber_delta_s: Positive = parameter(
        5e-7, "Huber transition on travel-time residual.", "s"
    )
    irls_iterations: PositiveInt = parameter(
        3, "Maximum robust reweighting stages, only used with Huber.", "stages"
    )
    gradient_rtol: Nonnegative = parameter(
        1e-10, "Relative projected optimality tolerance."
    )
    roi_laplacian: bool = parameter(False, "Use ROI-aware Laplacian regularization.")
    coverage_preconditioning: bool = parameter(
        False, "Enable diagonal coverage preconditioning."
    )
    coverage_preconditioner_eps: Positive = parameter(
        1e-12, "Coverage denominator floor.", "coverage units"
    )
    coverage_preconditioner_max_scale: Positive = parameter(
        10.0, "Maximum diagonal preconditioner scale."
    )
    coverage_preconditioner_normalize: bool = parameter(
        True, "Normalize positive ROI coverage scales by their median."
    )
    boundary_band_pixels: NonnegativeInt = parameter(
        4, "Post-hoc boundary diagnostic width; never used for stopping.", "pixels"
    )


class SIRTParameters(RayParameters):
    projector_backend: Literal["csr", "reference"] = parameter(
        "csr", "Equivalent straight-ray implementation.", exposure="internal"
    )
    relaxation: Annotated[float, Field(strict=True, gt=0, lt=2)] = parameter(
        0.3,
        "Algebraic update relaxation; stability also depends on smoothing/projection.",
        exposure="agent",
    )
    smooth_sigma: Nonnegative = parameter(
        0.0, "Gaussian smoothing standard deviation.", "pixels"
    )
    smooth_every: NonnegativeInt = parameter(
        0, "Smoothing interval; zero disables smoothing.", "complete iterations"
    )
    boundary_band_pixels: NonnegativeInt = parameter(
        4, "Post-hoc boundary diagnostic width.", "pixels"
    )


class SARTParameters(SIRTParameters):
    relaxation: Annotated[float, Field(strict=True, gt=0, lt=2)] = parameter(
        0.2, "Subset update relaxation.", exposure="agent"
    )
    subsets: PositiveInt = parameter(
        8, "Number of ordered subsets in one complete sweep.", "subsets"
    )


class InnerOptions(Parameters):
    rtol: Nonnegative = parameter(1e-7, "Normal residual tolerance.")
    atol: Nonnegative = parameter(
        1e-10, "Augmented least-squares absolute solver tolerance."
    )
    btol: Nonnegative = parameter(
        1e-10, "Augmented least-squares relative solver tolerance."
    )
    conlim: Nonnegative = parameter(1e8, "LSMR/LSQR condition limit, zero disables.")
    preconditioner: Literal["none", "column_rms"] = parameter(
        "none", "Inner column preconditioner."
    )
    preconditioner_probes: PositiveInt = parameter(
        8, "Diagonal estimation probes.", "probes"
    )
    preconditioner_seed: NonnegativeInt = parameter(0, "Deterministic probe seed.")


class GaussNewtonParameters(SoundSpeedParameters):
    inner_iterations: PositiveInt = parameter(
        16,
        "Truncated inner linear solve budget, not a nonlinear iteration count.",
        "inner iterations",
    )
    inner_solver: Literal["lsmr", "lsqr", "normal_cg"] = parameter(
        "lsmr", "Inner least-squares method."
    )
    inner_options: InnerOptions = parameter(
        InnerOptions(), "Inner numerical controls; never exposed to default Agent."
    )
    step_length: Positive = parameter(1.0, "Initial line-search step length.")
    smooth_sigma: Nonnegative = parameter(
        0.0, "Update smoothing standard deviation.", "pixels"
    )
    gradient_rtol: Nonnegative = parameter(
        1e-8, "Relative training gradient tolerance."
    )


class BentParameters(GaussNewtonParameters, RayParameters):
    initialization: Literal["configured", "cgls"] = parameter(
        "configured",
        "Scalar initialization or counted CGLS initialization.",
        exposure="agent",
    )
    initial_sound_speed_mps: Positive | list[list[Positive]] | None = parameter(
        None, "Null uses reference speed; expert calls may supply a [y,x] image.", "m/s"
    )
    initialization_iterations: PositiveInt = parameter(
        80, "CGLS setup iterations charged to work budgets.", "iterations"
    )
    regularization: Literal["identity", "laplacian"] = parameter(
        "laplacian", "Penalty on slowness perturbation.", exposure="agent"
    )
    regularization_lambda: Nonnegative = parameter(
        0.02,
        "Penalty amplitude; damping is its square.",
        "objective/operator dependent",
    )
    line_search: bool = parameter(True, "Enable objective backtracking.")
    eikonal_order: Literal[1, 2] = parameter(1, "Fast-marching spatial order.")


class BornParameters(GaussNewtonParameters):
    mode: Literal["nonlinear", "fixed_background"] = parameter(
        "nonlinear",
        "Relinearized inversion or fixed-background linear Born.",
        exposure="agent",
    )
    green_backend: Literal["eikonal_wkb", "volume_integral"] = parameter(
        "eikonal_wkb", "Physical Green model, not a compute device.", exposure="agent"
    )
    background_sound_speed_mps: Positive | list[list[Positive]] | None = parameter(
        None,
        "Case reference by default; expert calls may supply a [y,x] background.",
        "m/s",
    )
    initialization: Literal["configured", "phase_cgls"] = parameter(
        "configured", "Background initialization method.", exposure="agent"
    )
    initialization_iterations: PositiveInt = parameter(
        80, "Counted phase-CGLS setup budget.", "iterations"
    )
    initialization_lambda: Nonnegative = parameter(
        0.02, "Initialization regularization amplitude.", "objective/operator dependent"
    )
    initialization_smooth_mm: Nonnegative = parameter(
        3.0, "Initialization smoothing scale.", "mm"
    )
    initialization_max_phase_rms: Positive = parameter(
        0.2, "Phase-fit acceptance RMS.", "rad"
    )
    initialization_min_amplitude_ratio: Nonnegative = parameter(
        0.05, "Minimum usable phase amplitude ratio."
    )
    inner_iterations: PositiveInt = parameter(
        12,
        "Nonlinear inner solve budget; fixed-background legacy value migrates to run budget.",
        "inner iterations",
    )
    regularization: Literal["identity", "laplacian"] | None = parameter(
        None,
        "Null resolves to laplacian for nonlinear, identity for fixed background.",
        exposure="agent",
    )
    regularization_lambda: Nonnegative = parameter(
        0.0,
        "Penalty amplitude before optional Jacobian scaling.",
        "objective/operator dependent",
    )
    regularization_scaling: Literal["absolute", "relative_jacobian_diagonal"] = (
        parameter("absolute", "Penalty scaling convention.")
    )
    regularization_length_wavelengths: Nonnegative = parameter(
        0.0,
        "Laplacian length in training-band wavelengths; zero uses legacy pixel penalty.",
        "wavelengths",
        "agent",
    )
    max_update_mps: Positive = parameter(
        12.0, "Bound on each nonlinear sound-speed update.", "m/s", "agent"
    )
    max_backtracks: NonnegativeInt = parameter(
        10, "Maximum line-search backtracks.", "trials"
    )
    green_solver_rtol: Positive = parameter(
        1e-7, "Volume-integral Green solver tolerance."
    )
    green_solver_maxiter: PositiveInt = parameter(
        20, "Volume-integral Green solver iteration cap.", "iterations"
    )
    allow_underresolved: bool = parameter(
        False, "Expert override of the inverse-grid PPW guard."
    )
    assume_unit_source: bool = parameter(
        False, "Only for deliberately unit-source synthetic data.", exposure="internal"
    )
    use_feature_weights: bool = parameter(
        False, "Use feature-derived pressure weights, subject to leakage checks."
    )
    max_cache_bytes: PositiveInt = parameter(
        128 * 1024**2, "Green cache capacity.", "bytes", "internal"
    )

    @model_validator(mode="after")
    def resolve_regularizer(self):
        if self.regularization is None:
            self.regularization = (
                "laplacian" if self.mode == "nonlinear" else "identity"
            )
        return self


class AttenuationParameters(Parameters):
    relaxation: Annotated[float, Field(strict=True, gt=0, lt=2)] = parameter(
        0.8, "Algebraic attenuation relaxation.", exposure="agent"
    )
    attenuation_upper_np_per_m: Positive = parameter(
        80.0,
        "Legacy projection bound; reference frequency is not established by this parameter.",
        "Np/m (legacy frequency provenance)",
    )
    min_ray_weight: Annotated[float, Field(strict=True, ge=0, le=1)] = parameter(
        0.0, "Reject confidence below threshold."
    )
    ray_weight_power: Positive = parameter(1.0, "Confidence exponent.")


# Fixed-background CGLS has no nonlinear step, smoothing or inner solver controls.
# Reuse the identical physical field definitions without advertising unused knobs.
FixedBornParameters = create_model(
    "FixedBornParameters",
    __base__=SoundSpeedParameters,
    **{
        name: (
            BornParameters.model_fields[name].annotation,
            BornParameters.model_fields[name],
        )
        for name in (
            "green_backend",
            "background_sound_speed_mps",
            "regularization_lambda",
            "regularization_scaling",
            "regularization_length_wavelengths",
            "green_solver_rtol",
            "green_solver_maxiter",
            "allow_underresolved",
            "assume_unit_source",
            "use_feature_weights",
            "max_cache_bytes",
        )
    },
    mode=(
        Literal["fixed_background"],
        parameter(
            "fixed_background",
            "Fixed-background linear Born inversion.",
            exposure="agent",
        ),
    ),
    regularization=(
        Literal["identity", "laplacian"],
        parameter("identity", "Squared-slowness penalty operator.", exposure="agent"),
    ),
)


class TinyFWIParameters(Parameters):
    sound_speed_bounds_mps: Bounds = parameter(
        (1300.0, 1700.0), "Sanity-model projection bounds.", "m/s", "agent"
    )
    frequencies_hz: list[Positive] = parameter(
        [1e5, 1.5e5, 2e5],
        "Sanity-model frequencies, not a production schedule.",
        "Hz",
        min_length=1,
    )
    spacing_m: Positive | None = parameter(None, "Null uses case column spacing.", "m")
    initial_sound_speed_mps: Positive | None = parameter(
        None,
        "Sanity-model initialization; null uses case reference, otherwise 1500 m/s.",
        "m/s",
        "agent",
    )
    learning_rate: Positive = parameter(
        1e6, "Sanity-model update scale; no production calibration.", "model dependent"
    )
