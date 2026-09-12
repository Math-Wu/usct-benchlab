"""Tiny synthetic waveform-inversion proof of life."""

from __future__ import annotations

from usctbench.algorithms.configuration import validated_run

import numpy as np

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


class TinyFWIAlgorithm:
    """A low-frequency synthetic gradient/loss-descent smoke test.

    This is not production FWI. It uses a one-dimensional path waveform to
    verify inversion plumbing before adding heavy full-wave solvers.
    """

    name = "fwi_tiny"

    @validated_run
    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        if case.ground_truth.sound_speed_mps is None:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason="fwi_tiny requires sound_speed_mps ground truth for the synthetic proof-of-life",
            )

        truth = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
        path_truth = _central_path(truth)
        frequencies = np.asarray(
            config.parameters.get("frequencies_hz", [1.0e5, 1.5e5, 2.0e5]), dtype=float
        )
        spacing_m = float(config.parameters.get("spacing_m", case.grid.spacing_m[1]))
        initial_speed = float(
            config.parameters.get(
                "initial_sound_speed_mps",
                case.metadata.get("reference_sound_speed_mps", 1500.0),
            )
        )
        steps = int(config.parameters.get("steps", 20))
        learning_rate = float(config.parameters.get("learning_rate", 1.0e6))
        bounds = config.parameters.get("sound_speed_bounds_mps", [1300.0, 1700.0])

        observed = waveform_from_speed(path_truth, frequencies, spacing_m)
        initial = np.full_like(path_truth, initial_speed, dtype=float)
        reconstructed_path, losses = gradient_descent(
            initial,
            observed,
            frequencies,
            spacing_m,
            steps=steps,
            learning_rate=learning_rate,
            bounds_mps=(float(bounds[0]), float(bounds[1])),
        )
        sound_speed = np.tile(reconstructed_path[None, :], (truth.shape[0], 1))
        metrics = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "iterations": steps,
            "stop_reason": "schedule_completed",
            "stopping": {
                "optimization_variable": "central_path_sound_speed",
                "update_variable": None,
                "update_units": None,
                "update_norm": None,
                "update_norm_scope": None,
                "update_normalization": None,
                "update_stopping_supported": False,
                "iteration_unit": "clipped_gradient_step",
                "stage_id": None,
                "resolved_policy": {
                    "max_iterations": max(0, steps),
                    "update_rtol": None,
                },
                "reason": "schedule_completed",
            },
        }
        metrics.update(
            compute_image_metrics(sound_speed, truth, mask=case.grid.roi_mask)
        )
        metrics.update(
            compute_baseline_improvement_metrics(
                sound_speed, truth, initial_speed, mask=case.grid.roi_mask
            )
        )
        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            sound_speed_mps=sound_speed,
            metrics=metrics,
        )


def _central_path(sound_speed_mps: np.ndarray) -> np.ndarray:
    if sound_speed_mps.ndim != 2:
        raise ValueError("fwi_tiny expects a 2-D sound-speed image")
    return np.asarray(sound_speed_mps[sound_speed_mps.shape[0] // 2, :], dtype=float)


def path_travel_time_s(sound_speed_mps: np.ndarray, spacing_m: float) -> float:
    speed = np.asarray(sound_speed_mps, dtype=float)
    if speed.ndim != 1:
        raise ValueError("tiny FWI path model must be one-dimensional")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    if np.any(speed <= 0):
        raise ValueError("sound speed must be positive")
    return float(np.sum(spacing_m / speed))


def waveform_from_speed(
    sound_speed_mps: np.ndarray, frequencies_hz: np.ndarray, spacing_m: float
) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty 1-D array")
    if np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must be positive")
    tau = path_travel_time_s(sound_speed_mps, spacing_m)
    phase = 2.0 * np.pi * frequencies * tau
    return np.concatenate([np.cos(phase), np.sin(phase)])


def loss_and_gradient(
    sound_speed_mps: np.ndarray,
    observed_waveform: np.ndarray,
    frequencies_hz: np.ndarray,
    spacing_m: float,
) -> tuple[float, np.ndarray]:
    speed = np.asarray(sound_speed_mps, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    observed = np.asarray(observed_waveform, dtype=float)
    tau = path_travel_time_s(speed, spacing_m)
    phase = 2.0 * np.pi * frequencies * tau
    predicted = np.concatenate([np.cos(phase), np.sin(phase)])
    if predicted.shape != observed.shape:
        raise ValueError("observed waveform shape does not match frequencies")

    residual = predicted - observed
    loss = 0.5 * float(np.mean(residual**2))
    d_loss_d_pred = residual / float(residual.size)
    d_pred_d_tau = np.concatenate(
        [
            -np.sin(phase) * (2.0 * np.pi * frequencies),
            np.cos(phase) * (2.0 * np.pi * frequencies),
        ]
    )
    d_loss_d_tau = float(np.dot(d_loss_d_pred, d_pred_d_tau))
    d_tau_d_speed = -spacing_m / (speed**2)
    return loss, d_loss_d_tau * d_tau_d_speed


def gradient_descent(
    initial_speed_mps: np.ndarray,
    observed_waveform: np.ndarray,
    frequencies_hz: np.ndarray,
    spacing_m: float,
    *,
    steps: int,
    learning_rate: float,
    bounds_mps: tuple[float, float] = (1300.0, 1700.0),
) -> tuple[np.ndarray, list[float]]:
    speed = np.asarray(initial_speed_mps, dtype=float).copy()
    low, high = bounds_mps
    if low <= 0 or high <= low:
        raise ValueError("bounds_mps must be positive and increasing")
    losses: list[float] = []
    for _ in range(max(0, int(steps))):
        loss, grad = loss_and_gradient(
            speed, observed_waveform, frequencies_hz, spacing_m
        )
        losses.append(loss)
        speed = np.clip(speed - float(learning_rate) * grad, low, high)
    loss, _ = loss_and_gradient(speed, observed_waveform, frequencies_hz, spacing_m)
    losses.append(loss)
    return speed, losses
