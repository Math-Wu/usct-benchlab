"""Small WUST parameter contract; deployment is hidden from autonomous Agents."""

from typing import ClassVar, Literal
from pydantic import model_validator
from usctbench.algorithms.parameters import (
    Bounds,
    Nonnegative,
    Parameters,
    Positive,
    PositiveInt,
    parameter,
)


class WUSTParameters(Parameters):
    agent_schema_overrides: ClassVar[dict] = {
        "initialization": {"enum": ["reference", "scalar"]}
    }
    initialization: Literal["reference", "scalar", "map"] = parameter(
        "reference",
        "Reference or scalar initialization; map requires expert input.",
        exposure="agent",
    )
    initial_sound_speed_mps: Positive | None = parameter(
        None, "Required only for scalar initialization.", "m/s", "agent"
    )
    initial_map_mps: list[list[Positive]] | None = parameter(
        None,
        "Expert resolved initialization, exact case grid; never GT-derived.",
        "m/s",
    )
    sound_speed_bounds_mps: Bounds | None = parameter(
        None,
        "Explicit ordered projection bounds; no universal physiological defaults.",
        "m/s",
        "agent",
    )
    frequency_schedule_hz: list[Positive] | None = parameter(
        None,
        "One entry per update; null means one update per canonical frequency.",
        "Hz",
        "agent",
    )
    max_update_mps: Positive = parameter(
        12.0,
        "Numerical per-update speed cap, not a calibrated prescription.",
        "m/s",
        "agent",
    )
    step_damping: Positive = parameter(0.25, "WUST linearized-step damping.")
    filter_cutoff: Nonnegative = parameter(
        0.0, "Relative gradient filter cutoff; zero disables filtering."
    )
    filter_order: PositiveInt = parameter(4, "Gradient filter order.")
    pml_m: Positive | None = parameter(
        None, "Required numerical PML thickness for this computational domain.", "m"
    )
    pml_strength: Positive = parameter(10.0, "Numerical PML strength.")
    stencil_bounds_mps: Bounds | None = parameter(
        None, "Frozen stencil bounds; null uses full sound-speed bounds.", "m/s"
    )
    wavenumber_model: Literal["continuum", "kwave-ldr9"] = parameter(
        "continuum", "Helmholtz wavenumber model."
    )
    dispersion: dict[str, Positive] | None = parameter(
        None, "Explicit simulator facts for LDR9.", exposure="internal"
    )
    source_batch_size: PositiveInt = parameter(1, "Number of source RHS per batch.")
    update_mask: list[list[bool]] | None = parameter(
        None, "Expert mask, never inferred from GT; WUST validates PML clearance."
    )
    runtime_root: str | None = parameter(
        None, "Approved WUST checkout; null uses USCT_WUST_ROOT.", exposure="internal"
    )
    matlab_executable: str = parameter(
        "matlab", "Deployment-owned MATLAB executable.", exposure="internal"
    )
    cuda_visible_devices: str | None = parameter(
        None, "Deployment GPU selection; no CPU fallback.", exposure="internal"
    )
    backend: Literal["gpu", "cpu"] = parameter(
        "gpu", "GPU production; CPU reference/debug only.", exposure="internal"
    )
    scratch_root: str | None = parameter(
        None, "Deployment-owned artifact directory.", exposure="internal"
    )

    @model_validator(mode="after")
    def consistent(self):
        if self.initialization == "scalar" and self.initial_sound_speed_mps is None:
            raise ValueError("scalar initialization requires initial_sound_speed_mps")
        if self.initialization != "scalar" and self.initial_sound_speed_mps is not None:
            raise ValueError(
                "initial_sound_speed_mps belongs only to scalar initialization"
            )
        if self.initialization == "map" and self.initial_map_mps is None:
            raise ValueError("map initialization requires an expert resolved map")
        if self.initialization != "map" and self.initial_map_mps is not None:
            raise ValueError("initial_map_mps belongs only to map initialization")
        if (
            self.stencil_bounds_mps is not None
            and self.stencil_bounds_mps[0] >= self.stencil_bounds_mps[1]
        ):
            raise ValueError("stencil bounds must increase")
        if self.wavenumber_model == "kwave-ldr9":
            if self.dispersion is None or set(self.dispersion) != {
                "time_step_s",
                "reference_speed_mps",
                "model_reference_speed_mps",
            }:
                raise ValueError(
                    "LDR9 requires original simulator dt and both reference speeds"
                )
        elif self.dispersion is not None:
            raise ValueError("dispersion is only supported with kwave-ldr9")
        return self
