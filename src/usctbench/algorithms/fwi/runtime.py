"""Pinned WUST subprocess boundary; no numerical code or CPU fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

WUST_SHA = "79e347015be64cca88bacf591b4eed0952398800"
WUST_VERSION = "0.2.0-dev.1"
WUST_MANIFEST = "18fa31fd4cf5e44965bfc1250a4f73785d77f4af23ae40401a3e1afe73c94e4e"


def verify_provenance(provenance):
    expected = {
        "git_sha": WUST_SHA,
        "runtime_version": WUST_VERSION,
        "dirty": False,
        "source_manifest_verified": True,
        "source_manifest_sha256": WUST_MANIFEST,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise RuntimeFailure(
            "incompatible_runtime",
            "WUST result/probe identity differs from approved pin",
        )


class RuntimeFailure(RuntimeError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class Deadline:
    def __init__(self, seconds, *, started=None):
        if seconds is None:
            raise ValueError("fwi_wust requires an explicit elapsed-time budget")
        self.end = (time.monotonic() if started is None else started) + seconds

    def remaining(self):
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise RuntimeFailure("time_budget", "Global FWI deadline exhausted")
        return remaining


class WUSTClient:
    def __init__(self, parameters, deadline, folder):
        self.parameters = parameters
        self.deadline = deadline
        self.folder = Path(folder)
        root = parameters.runtime_root or os.environ.get("USCT_WUST_ROOT")
        if not root:
            raise RuntimeFailure("incompatible_runtime", "Set approved USCT_WUST_ROOT")
        self.root = Path(root).resolve(strict=True)
        self.launcher = self.root / "Runtime/python/wust_runtime.py"
        self.counter = 0

    def command(self, arguments):
        self.counter += 1
        log = self.folder / f"runtime-{self.counter}.stderr.log"
        env = os.environ.copy()
        if self.parameters.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.parameters.cuda_visible_devices
        cmd = [sys.executable, str(self.launcher), *arguments]
        remaining = self.deadline.remaining()
        handlers = {}

        def interrupted(signum, _frame):
            raise RuntimeFailure(
                "process_termination", f"BenchLab received signal {signum}"
            )

        with log.open("w") as stream:
            child = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stream,
                text=True,
                start_new_session=True,
                env=env,
            )
            try:
                if threading.current_thread() is threading.main_thread():
                    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                        handlers[signum] = signal.signal(signum, interrupted)
                stdout, _ = child.communicate(timeout=remaining)
            except BaseException as exc:
                for signum in handlers:
                    signal.signal(signum, signal.SIG_IGN)
                # WUST owns MATLAB's separate process group. Give its SIGTERM
                # handler time to reap that group before killing the launcher.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.communicate()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise RuntimeFailure(
                        "time_budget", "WUST deadline; process cleanup requested"
                    ) from exc
                raise
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
        if child.returncode:
            detail = {}
            for line in log.read_text().splitlines():
                try:
                    value = json.loads(line)
                    if value.get("schema") == "wust.execution_failure":
                        detail = value
                except (ValueError, AttributeError):
                    pass
            raise RuntimeFailure(
                detail.get("reason", "runtime_failure"),
                detail.get("message", f"WUST exit {child.returncode}; see {log}"),
            )
        self.deadline.remaining()
        try:
            return json.loads(stdout)
        except ValueError as exc:
            raise RuntimeFailure(
                "incompatible_runtime", "WUST stdout is not clean JSON"
            ) from exc

    def admit(self):
        info = self.command(["describe", "--json"])
        provenance = info.get("provenance", {})
        expected = {
            "wust.request",
            "wust.frequency_input",
            "wust.measurements",
            "wust.reconstruction",
        }
        if (
            info.get("schema") != "wust.capabilities"
            or info.get("schema_version") != 1
            or info.get("runtime_name") != "WUST"
            or info.get("runtime_version") != WUST_VERSION
            or any(info.get("schemas", {}).get(name) != 1 for name in expected)
            or provenance.get("git_sha") != WUST_SHA
            or provenance.get("dirty") is not False
            or provenance.get("source_manifest_verified") is not True
            or provenance.get("source_manifest_sha256") != WUST_MANIFEST
            or info.get("sound_speed_reconstruction", {}).get("supported") is not True
            or info.get("attenuation_reconstruction", {}).get("supported") is not False
            or not {"ingest_frequency", "reconstruct"}
            <= set(info.get("operations", []))
            or info.get("profiles", {}).get("gpu") != "gpu_complex64"
            or info.get("fourier_sign") != -1
            or info.get("source_scale")
            != {
                "method": "per_tx_per_frequency_complex_least_squares",
                "minimum_fitting_receivers": 2,
                "requires_water_reference": False,
                "requires_source_spectrum": False,
            }
            or info.get("budgets")
            != {
                "resolved_schedule": True,
                "hard_timeout_s": True,
                "forward_call_cap": False,
                "adjoint_call_cap": False,
                "update_rtol": False,
                "online_convergence": False,
            }
            or info.get("iteration_unit") != "frequency_schedule_update"
        ):
            raise RuntimeFailure(
                "incompatible_runtime", f"WUST must be clean approved SHA {WUST_SHA}"
            )
        probe = self.command(
            [
                "describe",
                "--json",
                "--probe",
                "--matlab",
                self.parameters.matlab_executable,
                "--timeout-s",
                str(self.deadline.remaining()),
            ]
        )
        verify_provenance(probe.get("provenance", {}))
        environment = probe.get("environment", {})
        backend = self.parameters.backend
        if environment.get(f"{backend}_available") is not True:
            raise RuntimeFailure(
                "incompatible_runtime",
                f"Requested {backend} unavailable: {environment}",
            )
        if backend == "gpu":
            mex = environment.get("mex", [])
            if len(mex) != 2 or any(
                not m.get("available")
                or not m.get("sha256")
                or Path(m["path"]).resolve().parent != self.root / "Runtime/solver"
                for m in mex
            ):
                raise RuntimeFailure(
                    "incompatible_runtime", "Approved runtime CUDA MEX required"
                )
        return {
            "static": info,
            "environment": environment,
            "role": "production" if backend == "gpu" else "reference",
            "approved_for_production": backend == "gpu",
        }

    def invoke(self, request):
        path = self.folder / f"{request['operation']}.request.json"
        path.write_text(json.dumps(request, allow_nan=False, indent=2))
        return self.command(
            [
                "run",
                str(path),
                "--matlab",
                self.parameters.matlab_executable,
                "--timeout-s",
                str(self.deadline.remaining()),
            ]
        )
