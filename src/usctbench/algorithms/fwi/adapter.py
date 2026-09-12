"""Adapter for k-Wave/WaveformInversionUST-style MATLAB FWI results."""

from __future__ import annotations

from usctbench.algorithms.configuration import validated_run

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.core.config import coerce_bool
from usctbench.metrics import (
    compute_baseline_improvement_metrics,
    compute_image_metrics,
)
from usctbench.core.schema import (
    AlgorithmConfig,
    ReconstructionResult,
    ResultStatus,
    USCTCase,
)

_INVERT_EXISTING_DATASET_MODES = {
    "invert_existing_dataset",
    "dataset",
    "skip_simulation",
}
_FULL_PIPELINE_MODES = {"full_pipeline", "full_pipeline_from_speed_map", "speed_map"}
_TRAVELTIME_WARM_START_BUILDERS = {
    "traveltime",
    "rf_traveltime",
    "travel_time",
    "rf_travel_time",
}
_BULK_SUPPORT_WARM_START_BUILDERS = {
    "bulk_support",
    "bulk-support",
    "bulk",
    "support_bulk",
}
_SUPPORTED_WARM_START_BUILDERS = (
    _TRAVELTIME_WARM_START_BUILDERS | _BULK_SUPPORT_WARM_START_BUILDERS
)
_CLI_PARAMS = {
    "mat_path": "--mat-path",
    "mat_key": "--mat-key",
    "sample_index": "--sample-index",
    "preprocessed_data_root": "--preprocessed-data-root",
    "preprocessed_split": "--preprocessed-split",
    "preprocessed_mat_path": "--preprocessed-mat-path",
    "preprocessed_mat_key": "--preprocessed-mat-key",
    "preprocessed_output_size": "--preprocessed-output-size",
    "array_mode": "--array-mode",
    "object_scale": "--object-scale",
    "object_pose": "--object-pose",
    "object_rotation_deg": "--object-rotation-deg",
    "background_speed": "--background-speed",
    "ncalc": "--ncalc",
    "xmax_mm": "--xmax-mm",
    "circle_radius_mm": "--circle-radius-mm",
    "f_tx_mhz": "--f-tx-mhz",
    "downsample_factor": "--downsample-factor",
    "backend": "--backend",
    "binary_path": "--binary-path",
    "generation_mode": "--generation-mode",
    "direct_num_workers": "--direct-num-workers",
    "kwave_data_path": "--kwave-data-path",
    "kwave_data_name_prefix": "--kwave-data-name-prefix",
    "output_dir": "--output-dir",
    "siminfo_path": "--siminfo-path",
    "scratch_dir": "--scratch-dir",
    "tx_downsample": "--tx-downsample",
    "recon_dxi_mm": "--recon-dxi-mm",
    "c_geom": "--c-geom",
    "c_init": "--c-init",
    "sign_conv": "--sign-conv",
    "a0": "--a0",
    "l_pml": "--l-pml",
    "exclude_neighbor_fraction": "--exclude-neighbor-fraction",
    "perc_outliers": "--perc-outliers",
    "warm_start_result": "--warm-start-result",
    "step_damping": "--step-damping",
    "tof_pre_frac": "--tof-pre-frac",
    "tof_post_frac": "--tof-post-frac",
    "filter_cutoff": "--filter-cutoff",
    "filter_order": "--filter-order",
    "save_raw_grad_iters": "--save-raw-grad-iters",
    "max_update_mps": "--max-update-mps",
    "shared_engine_name": "--shared-engine-name",
}
_CLI_SEQUENCE_PARAMS = {
    "cuda_devices": "--cuda-devices",
    "sos_freqs_mhz": "--sos-freqs-mhz",
    "sos_iters": "--sos-iters",
    "crange": "--crange",
    "velocity_bounds": "--velocity-bounds",
}
_CLI_BOOL_FLAGS = {
    "overwrite": "--overwrite",
    "keep_kwave_h5": "--keep-kwave-h5",
    "start_matlab": "--start-matlab",
    "no_connect_existing": "--no-connect-existing",
    "skip_inversion": "--skip-inversion",
    "use_preprocessed_field": "--use-preprocessed-field",
}
_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


class KWaveFWIAdapterAlgorithm:
    """Load or launch an external k-Wave frequency-domain FWI run.

    The default path is intentionally non-invasive: it reads an existing
    MATLAB result MAT file and converts it to the standard ReconstructionResult.
    Set `run_external: true` only in environments that deliberately invoke an
    external k-Wave/FWI pipeline.
    """

    name = "fwi_kwave_adapter"

    @validated_run
    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        if coerce_bool(config.parameters.get("controlled_operator", False)):
            from usctbench.algorithms.fwi.controlled import run_controlled
            from usctbench.algorithms.ray import run_with_failure_capture

            return run_with_failure_capture(
                self.name, case, lambda: run_controlled(case, config)
            )
        if config.parameters.get("stopping") or config.parameters.get("evaluation"):
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason="legacy external FWI artifacts/commands cannot enforce online stopping or held-out fitting; use controlled_operator with a compatible pressure case",
            )
        result_path = _configured_path(config, "result_path")
        if result_path is None:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason="fwi_kwave_adapter requires parameters.result_path",
            )

        run_external = coerce_bool(config.parameters.get("run_external", False))
        if run_external:
            launched = _run_external_pipeline(case, config, result_path)
            if launched.status != ResultStatus.SUCCESS:
                return launched

        if not result_path.exists():
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason=f"k-Wave FWI result file not found: {result_path}",
            )

        try:
            external = read_kwave_fwi_result(result_path)
        except Exception as exc:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=f"failed to read k-Wave FWI result: {type(exc).__name__}: {exc}",
            )

        selected_iteration, selection_metrics = _configured_iteration(
            config, external, case
        )
        sound_speed = _selected_sound_speed(
            external, selected_iteration, case.grid.shape
        )
        c0 = float(
            config.parameters.get(
                "baseline_sound_speed_mps",
                case.metadata.get("reference_sound_speed_mps", 1500.0),
            )
        )
        metrics = _base_result_metrics(
            config,
            external,
            result_path=result_path,
            selected_iteration=selected_iteration,
        )
        metrics.update(selection_metrics)
        if "best_iteration" not in metrics or "final_iteration_rmse" not in metrics:
            _best_iteration, diagnostic_metrics = _best_iteration_by_rmse(
                external, case
            )
            metrics.update(diagnostic_metrics)
        _add_case_ground_truth_metrics(metrics, sound_speed, external, case, c0)
        _add_kwave_ground_truth_metrics(metrics, sound_speed, external, case, c0)
        artifacts = _external_artifacts(config, external, result_path)
        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            sound_speed_mps=sound_speed,
            metrics=metrics,
            artifacts=artifacts,
        )


def _selected_sound_speed(
    external: dict[str, Any], selected_iteration: int | None, shape: tuple[int, int]
) -> np.ndarray:
    selected_sound_speed = _select_iteration_image(
        external, "sound_speed_iter_mps", "sound_speed_mps", selected_iteration
    )
    return _resize_to_shape(selected_sound_speed, shape)


def _base_result_metrics(
    config: AlgorithmConfig,
    external: dict[str, Any],
    *,
    result_path: Path,
    selected_iteration: int | None,
) -> dict[str, Any]:
    log_path = _configured_path(config, "external_log_path")
    configured_dataset_path = _configured_path(config, "dataset_path")
    losses = external.get("losses", [])
    psnr_value = external.get("psnr_value")
    ssim_value = external.get("ssim_value")
    return {
        "online_stopping": False,
        "stop_reason": "external_stop_reason_unavailable",
        "stopping": {
            "online": False,
            "reason": None,
            "scope": "external_driver_not_instrumented",
            "optimization_variable": None,
            "update_variable": None,
            "update_units": None,
            "update_norm": None,
            "update_norm_scope": None,
            "update_normalization": None,
            "iteration_unit": "external_driver_iteration_unverified",
            "stage_id": None,
            "resolved_policy": None,
            "semantics_unavailable_reason": "Final artifacts do not certify runtime update semantics or the executed stopping policy.",
        },
        "external_result_loaded": True,
        "external_result_path": str(result_path),
        "external_dataset_path": external.get("dataset_path")
        or (str(configured_dataset_path) if configured_dataset_path else ""),
        "external_execution_mode": _external_execution_mode(config),
        "external_log_path": str(log_path) if log_path else "",
        "iterations": int(external.get("iterations", 0)),
        "selected_iteration": selected_iteration or int(external.get("iterations", 0)),
        "selected_loss": _loss_at_iteration(losses, selected_iteration),
        "initial_loss": external.get("initial_loss"),
        "final_loss": external.get("final_loss"),
        "loss_decreased": external.get("loss_decreased"),
        "kwave_native_psnr": psnr_value,
        "kwave_native_ssim": ssim_value,
        "matlab_psnr_value": psnr_value,
        "matlab_ssim_value": ssim_value,
        "warm_start_builder": _expand_text(
            config.parameters.get("warm_start_builder", "")
        ),
        "warm_start_module": _warm_start_module_for_builder(config),
        "warm_start_path": str(
            _configured_path(config, "warm_start_path")
            or _configured_path(config, "warm_start_result")
            or ""
        ),
    }


def _add_case_ground_truth_metrics(
    metrics: dict[str, Any],
    sound_speed: np.ndarray,
    external: dict[str, Any],
    case: USCTCase,
    baseline_sound_speed_mps: float,
) -> None:
    if case.ground_truth.sound_speed_mps is None:
        return
    truth = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
    metrics.update(compute_image_metrics(sound_speed, truth, mask=case.grid.roi_mask))
    metrics.update(
        compute_baseline_improvement_metrics(
            sound_speed, truth, baseline_sound_speed_mps, mask=case.grid.roi_mask
        )
    )
    if external.get("initial_sound_speed_mps") is not None:
        init = _resize_to_shape(
            np.asarray(external["initial_sound_speed_mps"], dtype=float),
            case.grid.shape,
        )
        metrics.update(
            compute_image_metrics(init, truth, mask=case.grid.roi_mask, prefix="init_")
        )
        metrics.update(
            compute_baseline_improvement_metrics(
                init,
                truth,
                baseline_sound_speed_mps,
                mask=case.grid.roi_mask,
                prefix="init_water_",
            )
        )


def _add_kwave_ground_truth_metrics(
    metrics: dict[str, Any],
    sound_speed: np.ndarray,
    external: dict[str, Any],
    case: USCTCase,
    baseline_sound_speed_mps: float,
) -> None:
    if external.get("ground_truth_sound_speed_mps") is None:
        return
    kwave_truth = _resize_to_shape(
        external["ground_truth_sound_speed_mps"], case.grid.shape
    )
    metrics.update(
        compute_image_metrics(
            sound_speed, kwave_truth, mask=case.grid.roi_mask, prefix="kwave_gt_"
        )
    )
    metrics.update(
        compute_baseline_improvement_metrics(
            sound_speed,
            kwave_truth,
            baseline_sound_speed_mps,
            mask=case.grid.roi_mask,
            prefix="kwave_gt_water_",
        )
    )
    if external.get("initial_sound_speed_mps") is not None:
        kwave_init = _resize_to_shape(
            np.asarray(external["initial_sound_speed_mps"], dtype=float),
            case.grid.shape,
        )
        metrics.update(
            compute_image_metrics(
                kwave_init,
                kwave_truth,
                mask=case.grid.roi_mask,
                prefix="kwave_gt_init_",
            )
        )
        metrics.update(
            compute_baseline_improvement_metrics(
                kwave_init,
                kwave_truth,
                baseline_sound_speed_mps,
                mask=case.grid.roi_mask,
                prefix="kwave_gt_init_water_",
            )
        )
    _add_kwave_iteration_improvement_metrics(metrics)


def _add_kwave_iteration_improvement_metrics(metrics: dict[str, Any]) -> None:
    initial_rmse = metrics.get("kwave_gt_init_rmse")
    final_rmse = metrics.get("final_iteration_rmse", metrics.get("kwave_gt_rmse"))
    if not _is_finite_number(initial_rmse):
        return
    initial = float(initial_rmse)
    if _is_finite_number(final_rmse):
        final = float(final_rmse)
        metrics["kwave_gt_final_absolute_rmse_improvement"] = initial - final
        metrics["kwave_gt_final_relative_rmse_improvement"] = (
            (initial - final) / initial if initial > 0.0 else 0.0
        )
        metrics["kwave_gt_final_improved"] = final < initial
    selected_rmse = metrics.get("kwave_gt_rmse")
    if _is_finite_number(selected_rmse):
        selected = float(selected_rmse)
        metrics["kwave_gt_selected_absolute_rmse_improvement"] = initial - selected
        metrics["kwave_gt_selected_relative_rmse_improvement"] = (
            (initial - selected) / initial if initial > 0.0 else 0.0
        )
        metrics["kwave_gt_selected_improved"] = selected < initial


def _external_artifacts(
    config: AlgorithmConfig, external: dict[str, Any], result_path: Path
) -> dict[str, str]:
    log_path = _configured_path(config, "external_log_path")
    configured_dataset_path = _configured_path(config, "dataset_path")
    return {
        "external_result_path": str(result_path),
        "external_dataset_path": external.get("dataset_path")
        or (str(configured_dataset_path) if configured_dataset_path else ""),
        "external_log_path": str(log_path) if log_path else "",
    }


def read_kwave_fwi_result(path: str | Path) -> dict[str, Any]:
    h5py = _h5py()
    result_path = Path(path).expanduser().resolve()
    with h5py.File(result_path, "r") as handle:
        sound_speed = _require_dataset(handle, "VEL_ESTIM")
        ground_truth = _read_dataset(handle, "C_INTERP")
        sound_speed_iter = _read_dataset(handle, "VEL_ESTIM_ITER")
        losses = _read_vector(handle, "LOSS_ITER")
        return {
            "sound_speed_mps": np.asarray(sound_speed, dtype=float),
            "ground_truth_sound_speed_mps": (
                np.asarray(ground_truth, dtype=float)
                if ground_truth is not None
                else None
            ),
            "sound_speed_iter_mps": (
                np.asarray(sound_speed_iter, dtype=float)
                if sound_speed_iter is not None
                else None
            ),
            "initial_sound_speed_mps": _read_dataset(handle, "VEL_INIT"),
            "losses": losses.tolist(),
            "iterations": int(losses.size),
            "initial_loss": float(losses[0]) if losses.size else None,
            "final_loss": float(losses[-1]) if losses.size else None,
            "loss_decreased": (
                bool(losses[-1] < losses[0]) if losses.size >= 2 else None
            ),
            "psnr_value": _read_scalar(handle, "psnr_value"),
            "ssim_value": _read_scalar(handle, "ssim_value"),
            "dataset_path": _read_matlab_string(handle, "datasetPath"),
        }


def _run_external_pipeline(
    case: USCTCase, config: AlgorithmConfig, result_path: Path
) -> ReconstructionResult:
    build = _build_external_pipeline_command(case, config, result_path)
    if build["error"]:
        return ReconstructionResult(
            algorithm=KWaveFWIAdapterAlgorithm.name,
            case_id=case.case_id,
            status=ResultStatus.SKIPPED,
            failure_reason=str(build["error"]),
        )

    root_text = _expand_text(
        config.parameters.get("usct_kwave_root", os.environ.get("USCT_KWAVE_ROOT", ""))
    ).strip()
    if not root_text or "$" in root_text:
        return ReconstructionResult(
            algorithm=KWaveFWIAdapterAlgorithm.name,
            case_id=case.case_id,
            status=ResultStatus.SKIPPED,
            failure_reason=(
                "run_external requires parameters.usct_kwave_root or " "USCT_KWAVE_ROOT"
            ),
        )
    usct_kwave_root = Path(root_text).expanduser()
    commands = [list(command) for command in build["commands"]]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(usct_kwave_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    log_path = _configured_path(config, "external_log_path")
    timeout_s = float(config.parameters.get("timeout_s", 3600.0))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    for step_index, command in enumerate(commands, start=1):
        stdout_target = subprocess.PIPE
        stderr_target = subprocess.STDOUT
        log_handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_mode = "w" if step_index == 1 else "a"
            log_handle = log_path.open(log_mode, encoding="utf-8")
            if step_index > 1:
                log_handle.write("\n\n")
            log_handle.write(f"# external step {step_index}/{len(commands)}\n")
            log_handle.write("$ " + " ".join(command) + "\n\n")
            log_handle.flush()
            stdout_target = log_handle
            stderr_target = subprocess.STDOUT
        try:
            proc = subprocess.run(
                command,
                cwd=usct_kwave_root,
                env=env,
                text=True,
                stdout=stdout_target,
                stderr=stderr_target,
                timeout=timeout_s,
                check=False,
            )
        except Exception as exc:
            return ReconstructionResult(
                algorithm=KWaveFWIAdapterAlgorithm.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=f"external k-Wave FWI step {step_index} failed: {type(exc).__name__}: {exc}",
            )
        finally:
            if log_handle is not None:
                log_handle.close()

        if proc.returncode != 0:
            detail = ""
            if log_path is not None:
                detail = f"; log={log_path}"
            elif proc.stdout:
                detail = f"; stdout={proc.stdout[-1000:]}"
            return ReconstructionResult(
                algorithm=KWaveFWIAdapterAlgorithm.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=f"external k-Wave FWI step {step_index} returned {proc.returncode}{detail}",
            )
    return ReconstructionResult(
        algorithm=KWaveFWIAdapterAlgorithm.name, case_id=case.case_id
    )


def _build_external_pipeline_command(
    case: USCTCase, config: AlgorithmConfig, result_path: Path
) -> dict[str, Any]:
    mode = _external_execution_mode(config)
    dataset_path = _configured_path(config, "dataset_path")
    if mode in _INVERT_EXISTING_DATASET_MODES and dataset_path is None:
        dataset_from_case = case.metadata.get("source_path")
        dataset_path = (
            Path(_expand_text(dataset_from_case)).expanduser()
            if dataset_from_case
            else None
        )
    if mode in _INVERT_EXISTING_DATASET_MODES and dataset_path is None:
        return {
            "commands": [],
            "dataset_path": None,
            "mode": mode,
            "error": "run_external in invert_existing_dataset mode requires parameters.dataset_path or case.metadata.source_path",
        }
    if mode not in _INVERT_EXISTING_DATASET_MODES and mode not in _FULL_PIPELINE_MODES:
        return {
            "commands": [],
            "dataset_path": dataset_path,
            "mode": mode,
            "error": f"unsupported k-Wave external execution_mode: {mode}",
        }

    python_bin = _configured_python_bin(config)
    module = _expand_text(
        config.parameters.get(
            "pipeline_module", "openbreastus_diffusion.kwave_dps.run_full_pipeline"
        )
    )
    command = [python_bin, "-m", module]
    if mode in _INVERT_EXISTING_DATASET_MODES:
        command.extend(["--skip-siminfo", "--skip-rf", "--skip-assemble"])

    if dataset_path is not None:
        command.extend(["--dataset-path", str(dataset_path)])
    command.extend(["--result-path", str(result_path)])

    for key, flag in _CLI_PARAMS.items():
        if key in config.parameters and config.parameters[key] not in (None, ""):
            command.extend([flag, _expand_text(config.parameters[key])])
    for key, flag in _CLI_SEQUENCE_PARAMS.items():
        if key in config.parameters and config.parameters[key] not in (None, ""):
            values = config.parameters[key]
            if isinstance(values, (list, tuple)):
                sequence = values
            else:
                sequence = [values]
            command.append(flag)
            command.extend(_expand_text(value) for value in sequence)
    for key, flag in _CLI_BOOL_FLAGS.items():
        if _as_bool(config.parameters.get(key, False)):
            command.append(flag)

    command.extend(
        _expand_text(value) for value in config.parameters.get("pipeline_args", [])
    )
    # The legacy pipeline otherwise defaults to nonzero absorption in simulation.
    # Keep sound-speed-only stages explicit; these are not user hyperparameters.
    command.extend(
        [
            "--atten-bkgnd",
            "0",
            "--sos2atten",
            "0",
            "--atten-iters",
            "0",
            "--sos-atten-freqs-mhz",
        ]
    )
    commands = _with_optional_warm_start_steps(
        command,
        mode=mode,
        dataset_path=dataset_path,
        result_path=result_path,
        config=config,
    )
    return {
        "commands": commands,
        "dataset_path": dataset_path,
        "mode": mode,
        "error": None,
    }


def _with_optional_warm_start_steps(
    inversion_command: list[str],
    *,
    mode: str,
    dataset_path: Path | None,
    result_path: Path,
    config: AlgorithmConfig,
) -> list[list[str]]:
    builder = (
        _expand_text(config.parameters.get("warm_start_builder", "")).strip().lower()
    )
    if builder not in _SUPPORTED_WARM_START_BUILDERS:
        return [inversion_command]
    if dataset_path is None:
        raise ValueError("warm_start_builder requires parameters.dataset_path")

    suffix = (
        "_bulk_support_init.mat"
        if builder in _BULK_SUPPORT_WARM_START_BUILDERS
        else "_traveltime_init.mat"
    )
    warm_start_path = _configured_path(
        config, "warm_start_path"
    ) or result_path.with_name(result_path.stem + suffix)
    warm_start_summary_path = _configured_path(
        config, "warm_start_summary_path"
    ) or warm_start_path.with_suffix(".json")
    diagnostic_prefix = _configured_path(
        config, "warm_start_diagnostic_prefix"
    ) or warm_start_path.with_suffix("")
    python_bin = _configured_python_bin(config)
    warm_module = _warm_start_module_for_builder(config)
    warm_command = [
        python_bin,
        "-m",
        warm_module,
        "--dataset-path",
        str(dataset_path),
        "--output-path",
        str(warm_start_path),
        "--summary-path",
        str(warm_start_summary_path),
        "--diagnostic-prefix",
        str(diagnostic_prefix),
    ]
    warm_command.extend(
        _expand_text(value) for value in config.parameters.get("warm_start_args", [])
    )

    final_inversion_command = list(inversion_command)
    if mode in _FULL_PIPELINE_MODES:
        generation_command = list(inversion_command)
        generation_command.append("--skip-inversion")
        final_inversion_command.extend(
            ["--skip-siminfo", "--skip-rf", "--skip-assemble"]
        )
    else:
        generation_command = None
    final_inversion_command.extend(["--warm-start-result", str(warm_start_path)])

    commands = []
    if generation_command is not None:
        commands.append(generation_command)
    commands.append(warm_command)
    commands.append(final_inversion_command)
    return commands


def _warm_start_module_for_builder(config: AlgorithmConfig) -> str:
    configured = config.parameters.get("warm_start_module")
    if configured:
        return _expand_text(configured)
    builder = (
        _expand_text(config.parameters.get("warm_start_builder", "")).strip().lower()
    )
    if builder in _BULK_SUPPORT_WARM_START_BUILDERS:
        return "openbreastus_diffusion.kwave_dps.make_bulk_support_init"
    return "openbreastus_diffusion.kwave_dps.make_traveltime_init"


def _configured_path(config: AlgorithmConfig, key: str) -> Path | None:
    value = config.parameters.get(key)
    if not value:
        return None
    expanded = _expand_text(value).strip()
    if not expanded or "$" in expanded:
        return None
    return Path(expanded).expanduser()


def _configured_python_bin(config: AlgorithmConfig) -> str:
    value = _expand_text(config.parameters.get("python_bin", "")).strip()
    return value if value and "$" not in value else sys.executable


def _configured_iteration(
    config: AlgorithmConfig, external: dict[str, Any], case: USCTCase
) -> tuple[int | None, dict[str, Any]]:
    value = config.parameters.get("reconstruction_iteration")
    if value in (None, "", "final", "last"):
        return None, {"selection_mode": "final"}
    expanded = _expand_text(value)
    if "$" in expanded:
        return None, {"selection_mode": "final_unresolved_env"}
    mode = expanded.strip().lower()
    if mode in {"best", "best_rmse", "best_kwave_gt_rmse", "auto"}:
        if not coerce_bool(
            config.parameters.get("allow_ground_truth_selection", False)
        ):
            raise ValueError(
                "GT-based FWI selection requires explicit allow_ground_truth_selection=true; use final or a predetermined iteration for truth-free evaluation"
            )
        best_iteration, best_metrics = _best_iteration_by_rmse(external, case)
        if best_iteration is None:
            return None, {"selection_mode": f"{mode}_fallback_final", **best_metrics}
        return best_iteration, {"selection_mode": mode, **best_metrics}
    if mode in {"first", "initial"}:
        return 1, {"selection_mode": mode}
    iteration = int(expanded)
    if iteration <= 0:
        return None, {"selection_mode": "final_nonpositive"}
    total = int(external.get("iterations", 0))
    if total and iteration > total:
        return total, {"selection_mode": "clamped_configured_iteration"}
    return iteration, {"selection_mode": "configured_iteration"}


def _best_iteration_by_rmse(
    external: dict[str, Any], case: USCTCase
) -> tuple[int | None, dict[str, Any]]:
    stack = external.get("sound_speed_iter_mps")
    if stack is None:
        return None, {"best_iteration_reason": "missing_sound_speed_iter"}
    images = _iteration_stack(stack, iterations=external.get("iterations"))
    if images is None or images.size == 0:
        return None, {"best_iteration_reason": "empty_sound_speed_iter"}

    truth = external.get("ground_truth_sound_speed_mps")
    target_source = "kwave_gt"
    if truth is None:
        truth = case.ground_truth.sound_speed_mps
        target_source = "case_gt"
    if truth is None:
        return None, {"best_iteration_reason": "missing_ground_truth"}
    target = _resize_to_shape(np.asarray(truth, dtype=float), case.grid.shape)
    finite = np.isfinite(target)
    if not np.any(finite):
        return None, {"best_iteration_reason": "nonfinite_ground_truth"}
    mask = (
        np.asarray(case.grid.roi_mask, dtype=bool)
        if case.grid.roi_mask is not None
        else None
    )

    rmses: list[float] = []
    ssims: list[float] = []
    for image in images:
        resized = _resize_to_shape(np.asarray(image, dtype=float), case.grid.shape)
        metrics = compute_image_metrics(resized, target, mask=mask)
        rmses.append(float(metrics["rmse"]))
        ssims.append(float(metrics["ssim"]))
    best_index = int(np.argmin(np.asarray(rmses, dtype=float)))
    return best_index + 1, {
        "best_iteration": best_index + 1,
        "best_iteration_metric": "rmse",
        "best_iteration_target": target_source,
        "best_iteration_eval_shape": list(case.grid.shape),
        "best_iteration_rmse": rmses[best_index],
        "best_iteration_ssim": ssims[best_index],
        "final_iteration_rmse": rmses[-1],
        "final_iteration_ssim": ssims[-1],
    }


def _select_iteration_image(
    external: dict[str, Any], iter_key: str, final_key: str, iteration: int | None
) -> np.ndarray | None:
    if iteration is None:
        value = external.get(final_key)
        return np.asarray(value, dtype=float) if value is not None else None
    stack = external.get(iter_key)
    if stack is None:
        value = external.get(final_key)
        return np.asarray(value, dtype=float) if value is not None else None
    array = _iteration_stack(stack, iterations=external.get("iterations"))
    if array is None or array.shape[0] == 0:
        value = external.get(final_key)
        return np.asarray(value, dtype=float) if value is not None else None
    index = max(0, min(int(iteration) - 1, array.shape[0] - 1))
    return array[index]


def _iteration_stack(stack: Any, iterations: Any = None) -> np.ndarray | None:
    array = np.asarray(stack, dtype=float)
    if array.ndim < 3:
        return None
    if array.ndim != 3:
        return array

    iteration_count = _positive_int(iterations)
    if iteration_count is not None:
        matching_axes = [
            axis
            for axis, length in enumerate(array.shape)
            if int(length) == iteration_count
        ]
        if len(matching_axes) == 1:
            return np.moveaxis(array, matching_axes[0], 0)
        if matching_axes:
            preferred_axis = 2 if 2 in matching_axes else matching_axes[0]
            return np.moveaxis(array, preferred_axis, 0)

    if array.shape[0] < min(array.shape[1], array.shape[2]):
        return array
    if array.shape[-1] < min(array.shape[0], array.shape[1]):
        return np.moveaxis(array, -1, 0)
    if array.shape[0] == array.shape[1]:
        return np.moveaxis(array, -1, 0)
    return array


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _loss_at_iteration(losses: Any, iteration: int | None) -> float | None:
    values = np.asarray(losses, dtype=float).reshape(-1)
    if values.size == 0:
        return None
    if iteration is None:
        return float(values[-1])
    index = max(0, min(int(iteration) - 1, values.size - 1))
    return float(values[index])


def _external_execution_mode(config: AlgorithmConfig) -> str:
    return (
        _expand_text(config.parameters.get("execution_mode", "invert_existing_dataset"))
        .strip()
        .lower()
    )


def _expand_text(value: Any) -> str:
    text = str(value)

    def _replace_default(match: re.Match[str]) -> str:
        env_value = os.environ.get(match.group(1))
        return env_value if env_value not in (None, "") else match.group(2)

    text = _ENV_DEFAULT_RE.sub(_replace_default, text)
    return os.path.expanduser(os.path.expandvars(text))


def _as_bool(value: Any) -> bool:
    return coerce_bool(value)


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _resize_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.shape == shape:
        return array
    y_idx = np.linspace(0, array.shape[0] - 1, shape[0])
    x_idx = np.linspace(0, array.shape[1] - 1, shape[1])
    y0 = np.floor(y_idx).astype(int)
    x0 = np.floor(x_idx).astype(int)
    y1 = np.clip(y0 + 1, 0, array.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, array.shape[1] - 1)
    wy = (y_idx - y0)[:, None]
    wx = (x_idx - x0)[None, :]
    top = (1.0 - wx) * array[np.ix_(y0, x0)] + wx * array[np.ix_(y0, x1)]
    bottom = (1.0 - wx) * array[np.ix_(y1, x0)] + wx * array[np.ix_(y1, x1)]
    return (1.0 - wy) * top + wy * bottom


def _require_dataset(handle: Any, name: str) -> np.ndarray:
    value = _read_dataset(handle, name)
    if value is None:
        raise KeyError(f"missing required dataset {name}")
    return value


def _read_dataset(handle: Any, name: str) -> np.ndarray | None:
    if name not in handle:
        return None
    return np.asarray(handle[name][()])


def _read_vector(handle: Any, name: str) -> np.ndarray:
    value = _read_dataset(handle, name)
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def _read_scalar(handle: Any, name: str) -> float | None:
    value = _read_vector(handle, name)
    if value.size == 0:
        return None
    return float(value[0])


def _read_matlab_string(handle: Any, name: str) -> str | None:
    if name not in handle:
        return None
    values = np.asarray(handle[name][()]).reshape(-1)
    try:
        return "".join(chr(int(value)) for value in values if int(value) != 0)
    except Exception:
        return None


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "h5py is required to read MATLAB v7.3 FWI result files"
        ) from exc
    return h5py
