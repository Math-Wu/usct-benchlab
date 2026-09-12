"""One production FWI path: USCTCase -> pinned WUST -> final sound speed."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from usctbench.core.run_controls import BudgetCaps, RunControls
from usctbench.core.schema import ReconstructionResult, ResultStatus
from usctbench.metrics import compute_regional_image_metrics

from .io import array_hash, file_hash, frequency_input, read_artifact, write_artifact
from .parameters import WUSTParameters
from .runtime import Deadline, RuntimeFailure, WUSTClient, verify_provenance


def validate_config(config, values=None):
    if config.name not in (None, "fwi_wust"):
        raise ValueError("Algorithm configuration must name fwi_wust")
    values = dict(config.parameters if values is None else values)
    auxiliary = {
        key: values.pop(key)
        for key in ("_run_output_dir", "image_evaluation")
        if key in values
    }
    p = WUSTParameters.model_validate(values)
    controls = config.run_controls or RunControls()
    caps = BudgetCaps.model_validate(config.budget_caps or {})
    if controls.update_rtol is not None or any(
        value is not None
        for value in (
            controls.max_forward_calls,
            controls.max_adjoint_calls,
            caps.max_forward_calls,
            caps.max_adjoint_calls,
        )
    ):
        raise ValueError(
            "WUST supports iteration/time budgets only; no call caps or update_rtol"
        )
    if "image_evaluation" in auxiliary:
        from usctbench.algorithms.configuration import ImageEvaluationParameters

        auxiliary["image_evaluation"] = ImageEvaluationParameters.model_validate(
            auxiliary["image_evaluation"]
        ).model_dump(exclude_unset=True)
    return config.model_copy(
        update={
            "name": "fwi_wust",
            "parameters": {**p.backend_parameters(), **auxiliary},
        }
    )


def initialization(case, p):
    if p.sound_speed_bounds_mps is None or p.pml_m is None:
        raise ValueError(
            "Explicit sound_speed_bounds_mps and pml_m required for execution"
        )
    if p.initialization == "map":
        initial = np.asarray(p.initial_map_mps, dtype=np.float64)
    else:
        value = (
            p.initial_sound_speed_mps
            if p.initialization == "scalar"
            else case.metadata.get("reference_sound_speed_mps")
        )
        if value is None:
            raise ValueError(
                "Reference initialization requires declared reference_sound_speed_mps"
            )
        initial = np.full(case.grid.shape, value, dtype=np.float64)
    low, high = p.sound_speed_bounds_mps
    if (
        initial.shape != case.grid.shape
        or not np.isfinite(initial).all()
        or np.any((initial < low) | (initial > high))
    ):
        raise ValueError(
            "Initialization must match exact grid and bounds; no resize/clipping"
        )
    if p.update_mask is not None:
        mask = np.asarray(p.update_mask, dtype=bool)
        policy = "explicit_expert_mask"
    else:
        # BenchLab's conservative physical-region policy, not a copy of WUST's
        # snapping or PML validator. WUST independently admits the requested mask.
        ny, nx = case.grid.shape
        dy, dx = case.grid.spacing_m
        clearance = p.pml_m + max(dy, dx)
        y = (np.arange(ny) + 0.5) * dy
        x = (np.arange(nx) + 0.5) * dx
        mask = ((y > clearance) & (ny * dy - y > clearance))[:, None] & (
            (x > clearance) & (nx * dx - x > clearance)
        )[None, :]
        policy = "edge_clearance_pml_plus_one_max_spacing"
    if mask.shape != case.grid.shape or not mask.any():
        raise ValueError(
            "Update mask must match grid and contain requested update pixels"
        )
    return initial, mask, policy


def resolve_schedule(frequencies, requested):
    frequencies = np.asarray(frequencies)
    if (
        frequencies.ndim != 1
        or not len(frequencies)
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0)
        or np.any(np.diff(frequencies) <= 0)
    ):
        raise ValueError(
            "WUST canonical frequencies must be positive strictly increasing"
        )
    hz = frequencies.tolist() if requested is None else list(requested)
    lookup = {value: i + 1 for i, value in enumerate(frequencies)}
    if any(value not in lookup for value in hz):
        raise ValueError(
            "Schedule frequency must exactly match WUST canonical frequencies"
        )
    return hz, [lookup[value] for value in hz]


def benchlab_identity():
    root = Path(__file__).resolve().parents[4]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


class WUSTFWIAlgorithm:
    name = "fwi_wust"

    def run(self, case, config):
        started = time.monotonic()
        folder = None
        try:
            try:
                config = validate_config(config)
            except (TypeError, ValueError) as exc:
                raise RuntimeFailure(
                    "incompatible_request", f"invalid configuration: {exc}"
                ) from exc
            p = WUSTParameters.model_validate(
                {
                    k: v
                    for k, v in config.parameters.items()
                    if k not in {"_run_output_dir", "image_evaluation"}
                }
            )
            controls = config.run_controls or RunControls()
            caps = BudgetCaps.model_validate(config.budget_caps or {})
            provisional = controls.capped(caps, default_iterations=0)
            deadline = Deadline(provisional.max_elapsed_s, started=started)
            base = p.scratch_root or config.parameters.get("_run_output_dir")
            if base is not None:
                Path(base).mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix="wust-", dir=base)).resolve()
            initial, mask, mask_policy = initialization(case, p)
            input_path, initial_path = (
                folder / "frequency.json",
                folder / "initial.json",
            )
            input_doc = frequency_input(case, input_path)
            write_artifact(
                initial_path,
                "wust.initial_model",
                {"initialization": p.initialization, "mask_policy": mask_policy},
                {"initial_mps": initial, "update_mask": mask},
                {"initial_mps": "y,x", "update_mask": "y,x"},
                {"initial_mps": "m/s", "update_mask": "1"},
            )
            client = WUSTClient(p, deadline, folder)
            admission = client.admit()
            canonical_path = folder / "measurements.json"
            client.invoke(
                {
                    "schema": "wust.request",
                    "schema_version": 1,
                    "operation": "ingest_frequency",
                    "input_manifest": str(input_path),
                    "output_manifest": str(canonical_path),
                    "config": {},
                }
            )
            canonical, arrays = read_artifact(canonical_path, "wust.measurements")
            verify_provenance(canonical["metadata"].get("runtime_provenance", {}))
            if (
                canonical["arrays"]["frequencies_hz"]["axes"] != "frequency"
                or canonical["arrays"]["frequencies_hz"]["units"] != "Hz"
            ):
                raise ValueError("Invalid canonical frequency axes/units")
            requested, full = resolve_schedule(
                arrays["frequencies_hz"], p.frequency_schedule_hz
            )
            resolved = controls.capped(caps, default_iterations=len(full))
            schedule = full[: min(len(full), resolved.max_iterations)]
            resolved = resolved.model_copy(update={"max_iterations": len(schedule)})
            runtime_config = {
                key: getattr(p, key)
                for key in (
                    "backend",
                    "max_update_mps",
                    "step_damping",
                    "source_batch_size",
                    "pml_strength",
                    "pml_m",
                    "filter_cutoff",
                    "filter_order",
                )
            }
            runtime_config.update(
                bounds_mps=list(p.sound_speed_bounds_mps),
                stencil_bounds=list(p.stencil_bounds_mps or p.sound_speed_bounds_mps),
                wavenumber=p.wavenumber_model,
            )
            if p.dispersion is not None:
                runtime_config["dispersion"] = p.dispersion
            output_path = folder / "reconstruction.json"
            client.invoke(
                {
                    "schema": "wust.request",
                    "schema_version": 1,
                    "operation": "reconstruct",
                    "input_manifest": str(canonical_path),
                    "initial_manifest": str(initial_path),
                    "output_manifest": str(output_path),
                    "config": runtime_config,
                    "schedule": schedule,
                    "planned_schedule_length": len(full),
                }
            )
            output, result_arrays = read_artifact(output_path, "wust.reconstruction")
            c = result_arrays["c_mps"]
            meta = output["metadata"]
            verify_provenance(meta.get("runtime_provenance", {}))
            if (
                output["arrays"]["c_mps"]["axes"] != "y,x"
                or output["arrays"]["c_mps"]["units"] != "m/s"
            ):
                raise ValueError("Invalid final-model axes/units")
            if (
                c.shape != case.grid.shape
                or c.dtype != np.float64
                or not np.isfinite(c).all()
                or np.any(c <= 0)
            ):
                raise ValueError(
                    "Invalid authoritative WUST final model; no resizing allowed"
                )
            if (
                meta.get("completed_updates") != len(schedule)
                or meta.get("executed_schedule") != schedule
            ):
                raise ValueError("WUST completed schedule differs from request")
            expected_reason = (
                "zero_updates"
                if not schedule
                else (
                    "caller_truncated_schedule"
                    if len(schedule) < len(full)
                    else "schedule_complete"
                )
            )
            if (
                meta.get("completion", {}).get("reason") != expected_reason
                or meta["completion"].get("converged") is not False
            ):
                raise ValueError("Invalid WUST completion semantics")
            if meta.get("environment", {}).get("backend") != p.backend:
                raise RuntimeFailure(
                    "incompatible_runtime",
                    "WUST backend differs; no fallback permitted",
                )
            metrics = {
                "forward_model": "WUST_Helmholtz",
                "final_data_residual": None,
                "stop_reason": (
                    "schedule_complete"
                    if expected_reason == "schedule_complete"
                    else "max_iterations"
                ),
                "termination_category": (
                    "completion" if expected_reason == "schedule_complete" else "budget"
                ),
                "converged": False,
                "completed_iterations": len(schedule),
                "wust": meta,
                "runtime_admission": admission,
                "requested_schedule_hz": requested,
                "canonical_frequencies_hz": arrays["frequencies_hz"].tolist(),
                "full_index_schedule": full,
                "executed_index_schedule": schedule,
                "run_controls": controls.model_dump(),
                "budget_caps": caps.model_dump(),
                "resolved_run_policy": resolved.model_dump(),
                "parameter_schema_version": "usct.wust.parameters.v1",
                "benchlab": benchlab_identity(),
                "input_hashes": {
                    "frequency_manifest": file_hash(input_path),
                    "canonical_manifest": file_hash(canonical_path),
                    "initial_model": array_hash(initial),
                    "update_mask": array_hash(mask),
                    "grid_geometry": hashlib_geometry(case),
                    "frequency_arrays": input_doc["arrays_sha256"],
                },
                "output_hashes": {
                    "final_model": array_hash(c),
                    "manifest": file_hash(output_path),
                    "arrays": output["arrays_sha256"],
                },
            }
            metrics.update(
                compute_regional_image_metrics(
                    c,
                    case.ground_truth.sound_speed_mps,
                    **config.parameters.get("image_evaluation", {}),
                )
            )
            deadline.remaining()
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                sound_speed_mps=c,
                metrics=metrics,
                runtime_s=time.monotonic() - started,
                artifacts={"wust_directory": str(folder)},
            )
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            reason = getattr(exc, "reason", "incompatible_request")
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=f"{reason}: {exc}",
                runtime_s=time.monotonic() - started,
                metrics={
                    "stop_reason": reason,
                    "termination_category": (
                        "budget" if reason == "time_budget" else "failure"
                    ),
                    "converged": False,
                },
                artifacts={} if folder is None else {"wust_directory": str(folder)},
            )


def hashlib_geometry(case):
    import hashlib

    facts = {
        "shape": case.grid.shape,
        "spacing_m": case.grid.spacing_m,
        "origin_m": case.grid.origin_m,
        "tx": array_hash(case.geometry.tx_pos_m),
        "rx": array_hash(case.geometry.rx_pos_m),
    }
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
