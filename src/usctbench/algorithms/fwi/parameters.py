"""Typed boundaries, not a replacement for external FWI runtime validation.

Optional production fields are forwarded only when supplied. Their defaults
remain owned by the installed external pipeline; no local defaults are invented.
"""

from typing import Literal

from pydantic import TypeAdapter, model_validator

from usctbench.algorithms.parameters import (
    Bounds,
    Nonnegative,
    NonnegativeInt,
    Parameters,
    Positive,
    PositiveInt,
    SoundSpeedParameters,
    parameter,
)


def delegated(description, units="1", exposure="internal"):
    return parameter(
        None, description + " Null delegates to external runtime.", units, exposure
    )


class ControlledFWIParameters(SoundSpeedParameters):
    controlled_operator: Literal[True] = parameter(
        True, "Use exact-discrete MATLAB reference operator.", exposure="internal"
    )
    functions_path: str | None = parameter(
        None,
        "Deployment-approved WaveformInversionUST functions directory.",
        exposure="internal",
    )
    matlab_bin: str = parameter(
        "matlab", "Deployment-owned MATLAB executable.", exposure="internal"
    )
    pml_m: Positive = parameter(0.009, "PML thickness.", "m")
    pml_strength: Positive = parameter(10.0, "Reference Helmholtz PML strength.")
    operator_timeout_s: Positive = parameter(
        180.0, "Per-call runtime timeout, not a shared deadline.", "s", "internal"
    )
    initial_sound_speed_mps: Positive | list[list[Positive]] | None = parameter(
        None,
        "Reference speed by default; expert [y,x] initialization is supported.",
        "m/s",
    )
    assume_unit_source: bool = parameter(
        False, "Explicit synthetic unit-source assumption.", exposure="internal"
    )
    inner_iterations: PositiveInt = parameter(
        4, "Truncated Gauss-Newton inner budget.", "iterations"
    )
    regularization: Literal["identity", "laplacian"] = parameter(
        "laplacian", "Sound-speed penalty operator.", exposure="agent"
    )
    regularization_lambda: Nonnegative = parameter(
        0.0, "Penalty amplitude.", "objective/operator dependent"
    )
    max_update_mps: Positive = parameter(
        12.0, "Maximum sound-speed update per outer iteration.", "m/s", "agent"
    )
    step_length: Positive = parameter(1.0, "Initial line-search step.")
    max_backtracks: NonnegativeInt = parameter(
        8, "Line-search backtrack limit.", "trials"
    )
    roi_update_only: bool = parameter(True, "Restrict updates to supplied ROI.")


class ExternalFWIParameters(Parameters):
    controlled_operator: Literal[False] = parameter(
        False, "External artifact/pipeline boundary.", exposure="internal"
    )
    run_external: bool = parameter(
        False,
        "Launch deployment-approved external pipeline; false imports a result.",
        exposure="internal",
    )
    execution_mode: Literal["invert_existing_dataset", "full_pipeline"] = parameter(
        "invert_existing_dataset",
        "External data preparation operation.",
        exposure="internal",
    )
    result_path: str | None = delegated("Result MAT artifact")
    dataset_path: str | None = delegated("Input dataset MAT artifact")
    external_log_path: str | None = delegated("External process log")
    usct_kwave_root: str | None = delegated("External package root")
    python_bin: str | None = delegated("Python executable")
    pipeline_module: str = parameter(
        "openbreastus_diffusion.kwave_dps.run_full_pipeline",
        "Deployment-approved pipeline module.",
        exposure="internal",
    )
    pipeline_args: list[str] = parameter(
        [], "Expert-only external arguments; never Agent input.", exposure="internal"
    )
    timeout_s: Positive = parameter(
        3600.0,
        "Legacy per-process timeout, not a global compute budget.",
        "s",
        "internal",
    )
    reconstruction_iteration: str | int = parameter(
        "final",
        "Expert artifact selection. GT-based choices require explicit permission.",
    )
    allow_ground_truth_selection: bool = parameter(
        False,
        "Allow explicitly labeled oracle artifact selection.",
        exposure="internal",
    )
    baseline_sound_speed_mps: Positive | None = parameter(
        None,
        "Post-hoc baseline; null uses case reference speed, then 1500 m/s.",
        "m/s",
        "internal",
    )
    initial_sound_speed_mps: Positive | None = delegated(
        "Initial sound speed (legacy c_init)", "m/s", "agent"
    )
    sound_speed_bounds_mps: Bounds | None = delegated(
        "Velocity projection bounds", "m/s", "agent"
    )
    warm_start_builder: Literal["", "traveltime", "bulk_support"] = parameter(
        "",
        "Approved initialization builder; empty disables the builder.",
        exposure="agent",
    )
    warm_start_module: str | None = delegated("Initialization module override")
    warm_start_args: list[str] = parameter(
        [], "Expert-only initialization arguments.", exposure="internal"
    )
    warm_start_path: str | None = delegated("Initialization output artifact")
    warm_start_summary_path: str | None = delegated("Initialization summary artifact")
    warm_start_diagnostic_prefix: str | None = delegated(
        "Initialization diagnostic path"
    )
    warm_start_result: str | None = delegated("Existing initialization artifact")
    mat_path: str | None = delegated("Property map MAT path")
    mat_key: str | None = delegated("Property map MAT variable")
    sample_index: NonnegativeInt | None = delegated("Dataset sample index")
    preprocessed_data_root: str | None = delegated("Preprocessed data root")
    preprocessed_split: str | None = delegated("Preprocessed split")
    preprocessed_mat_path: str | None = delegated("Preprocessed MAT path")
    preprocessed_mat_key: str | None = delegated("Preprocessed MAT variable")
    preprocessed_output_size: PositiveInt | None = delegated(
        "Preprocessed output width", "pixels"
    )
    array_mode: str | None = delegated("External acquisition layout")
    object_scale: Positive | None = delegated("Object scale")
    object_pose: str | None = delegated("Object pose")
    object_rotation_deg: float | None = delegated("Object rotation", "degrees")
    background_speed: Positive | None = delegated("Simulation background speed", "m/s")
    ncalc: PositiveInt | None = delegated("Simulation grid width", "pixels")
    xmax_mm: Positive | None = delegated("Simulation extent", "mm")
    circle_radius_mm: Positive | None = delegated("Array radius", "mm")
    atten_bkgnd: Nonnegative | None = delegated(
        "Legacy background attenuation", "external legacy units"
    )
    sos2atten: float | None = delegated(
        "External speed-to-attenuation coefficient", "external legacy units"
    )
    y_atten: Nonnegative | None = delegated("External attenuation exponent")
    f_tx_mhz: Positive | None = delegated("Simulation transmit frequency", "MHz")
    downsample_factor: PositiveInt | None = delegated("Simulation downsampling")
    backend: str | None = delegated("External simulation backend")
    binary_path: str | None = delegated("Simulation binary")
    generation_mode: str | None = delegated("External generation mode")
    direct_num_workers: PositiveInt | None = delegated("Simulation workers")
    kwave_data_path: str | None = delegated("Waveform output root")
    kwave_data_name_prefix: str | None = delegated("Waveform output prefix")
    output_dir: str | None = delegated("Pipeline output root")
    siminfo_path: str | None = delegated("Simulation metadata artifact")
    scratch_dir: str | None = delegated("Runtime scratch directory")
    tx_downsample: PositiveInt | None = delegated("Transmitter subsampling")
    recon_dxi_mm: Positive | None = delegated(
        "Reconstruction spacing", "mm", "advanced"
    )
    c_geom: Positive | None = delegated(
        "Geometric preprocessing speed", "m/s", "advanced"
    )
    sign_conv: Literal[-1, 1] | None = delegated("Pressure sign convention")
    a0: float | None = delegated(
        "External attenuation initialization", "external legacy units", "advanced"
    )
    l_pml: Positive | None = delegated(
        "External PML setting", "external runtime units", "advanced"
    )
    exclude_neighbor_fraction: Nonnegative | None = delegated(
        "External near-neighbor exclusion fraction", exposure="advanced"
    )
    perc_outliers: Nonnegative | None = delegated(
        "External outlier percentile convention", exposure="advanced"
    )
    step_damping: Positive | None = delegated(
        "External step damping", exposure="advanced"
    )
    tof_pre_frac: Nonnegative | None = delegated(
        "Pre-arrival window fraction", exposure="advanced"
    )
    tof_post_frac: Nonnegative | None = delegated(
        "Post-arrival window fraction", exposure="advanced"
    )
    filter_cutoff: Positive | None = delegated(
        "External gradient filter cutoff", "external runtime units", "advanced"
    )
    filter_order: PositiveInt | None = delegated(
        "External gradient filter order", exposure="advanced"
    )
    save_raw_grad_iters: NonnegativeInt | None = delegated("Raw gradient save interval")
    max_update_mps: Positive | None = delegated(
        "External runtime update cap; deployment must verify support", "m/s", "advanced"
    )
    shared_engine_name: str | None = delegated("MATLAB shared engine")
    cuda_devices: list[int] | None = delegated("Simulation CUDA device ids")
    sos_frequencies_hz: list[Positive] | None = delegated(
        "Sound-speed frequency schedule", "Hz", "advanced"
    )
    attenuation_frequencies_hz: list[Positive] | None = delegated(
        "Joint speed/attenuation schedule", "Hz", "advanced"
    )
    sos_iters: list[NonnegativeInt] | None = delegated(
        "Iterations per speed stage", "stage iterations", "advanced"
    )
    atten_iters: list[NonnegativeInt] | None = delegated(
        "Iterations per attenuation stage", "stage iterations", "advanced"
    )
    crange: Bounds | None = delegated("External speed display range", "m/s")
    attenrange: tuple[float, float] | None = delegated(
        "External attenuation display range", "legacy units"
    )
    overwrite: bool = parameter(
        False, "Allow external artifact overwrite.", exposure="internal"
    )
    keep_kwave_h5: bool = parameter(
        False, "Retain external simulation scratch files.", exposure="internal"
    )
    start_matlab: bool = parameter(False, "Start MATLAB runtime.", exposure="internal")
    no_connect_existing: bool = parameter(
        False, "Do not connect to a shared MATLAB engine.", exposure="internal"
    )
    skip_inversion: bool = parameter(False, "Generate data only.", exposure="internal")
    use_preprocessed_field: bool = parameter(
        False, "Use configured preprocessing artifacts.", exposure="internal"
    )

    @model_validator(mode="before")
    @classmethod
    def external_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        values = dict(data)
        # The expert CLI historically accepts a scalar for sequence arguments.
        # Canonical Agent admission validates its own strict schema first.
        for key in (
            "cuda_devices",
            "sos_freqs_mhz",
            "sos_atten_freqs_mhz",
            "sos_iters",
            "atten_iters",
            "crange",
            "attenrange",
            "velocity_bounds",
        ):
            if key in values and values[key] is not None:
                value = values[key]
                values[key] = (
                    list(value) if isinstance(value, (list, tuple)) else [value]
                )
        for old, new, scale in (
            ("c_init", "initial_sound_speed_mps", 1),
            ("velocity_bounds", "sound_speed_bounds_mps", 1),
            ("sos_freqs_mhz", "sos_frequencies_hz", 1e6),
            ("sos_atten_freqs_mhz", "attenuation_frequencies_hz", 1e6),
        ):
            if old in values:
                value = values.pop(old)
                if value is None:
                    continue
                annotation = (
                    list[Positive]
                    if scale == 1e6
                    else Bounds if old == "velocity_bounds" else Positive
                )
                value = TypeAdapter(annotation).validate_python(value)
                value = (
                    [v * scale for v in value]
                    if isinstance(value, (list, tuple))
                    else value * scale
                )
                if values.get(new) is not None and listify(values[new]) != listify(
                    value
                ):
                    raise ValueError(f"conflicting aliases: {old} and {new}")
                values[new] = value
        if "execution_mode" in values:
            values["execution_mode"] = {
                "dataset": "invert_existing_dataset",
                "skip_simulation": "invert_existing_dataset",
                "full_pipeline_from_speed_map": "full_pipeline",
                "speed_map": "full_pipeline",
            }.get(values["execution_mode"], values["execution_mode"])
        if "warm_start_builder" in values:
            value = values["warm_start_builder"]
            values["warm_start_builder"] = {
                "rf_traveltime": "traveltime",
                "travel_time": "traveltime",
                "rf_travel_time": "traveltime",
                "bulk-support": "bulk_support",
                "bulk": "bulk_support",
                "support_bulk": "bulk_support",
            }.get(value, value)
        return values

    def backend_parameters(self):
        # Translate only at the external CLI boundary; typed state stays canonical.
        values = super().backend_parameters()
        for canonical, old, scale in (
            ("initial_sound_speed_mps", "c_init", 1),
            ("sound_speed_bounds_mps", "velocity_bounds", 1),
            ("sos_frequencies_hz", "sos_freqs_mhz", 1e6),
            ("attenuation_frequencies_hz", "sos_atten_freqs_mhz", 1e6),
        ):
            if canonical in values:
                value = values.pop(canonical)
                values[old] = (
                    [v / scale for v in value]
                    if isinstance(value, (list, tuple))
                    else value / scale
                )
        return values


def listify(value):
    return list(value) if isinstance(value, (tuple, list)) else value
