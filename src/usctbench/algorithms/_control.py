"""Shared run bookkeeping. Only training samples enter an optimization update."""

from __future__ import annotations

import numpy as np

from usctbench.core.config import coerce_bool
from usctbench.core.run_controls import RunControls
from usctbench.core.stopping import StopMonitor, StopPolicy, WorkLedger
from usctbench.evaluation.data import make_data_split, residual_statistics
from usctbench.metrics import (
    compute_baseline_improvement_metrics,
    compute_image_metrics,
)


class InversionControl:
    def __init__(
        self,
        case,
        config,
        observed,
        *,
        default_iterations,
        weights=None,
        valid_mask=None,
        iteration_unit="iteration",
    ):
        self.case, self.config = case, config
        self.observed = np.asarray(observed)
        settings = dict(config.parameters.get("evaluation", {}))
        if "exclude_reciprocal" in settings:
            settings["exclude_reciprocal"] = coerce_bool(settings["exclude_reciprocal"])
        self.split = make_data_split(
            self.observed,
            valid_mask=valid_mask,
            weights=weights,
            tx_positions=case.geometry.tx_pos_m,
            rx_positions=case.geometry.rx_pos_m,
            **settings,
        )
        self.policy = (
            (config.run_controls or RunControls()).to_stop_policy(
                default_iterations=config.parameters.get(
                    "iterations", default_iterations
                ),
                budget=config.budget_caps,
            )
            if config.run_controls is not None or config.budget_caps is not None
            else StopPolicy.from_parameters(
                config.parameters, default_iterations=default_iterations
            )
        )
        if self.policy.validation_patience is not None and not np.any(
            self.split.validation
        ):
            raise ValueError("validation_patience requires a nonempty hold-out split")
        if (
            self.policy.target_rmse_mps is not None
            and case.ground_truth.sound_speed_mps is None
        ):
            raise ValueError("oracle quality stopping requires case ground truth")
        self.work = WorkLedger(self.policy)
        self.monitor = StopMonitor(
            self.policy, self.work, iteration_unit=iteration_unit
        )
        self.precision = np.where(self.split.train, self.split.weights, 0.0)
        self.safe_observed = np.where(self.split.train, self.observed, 0)
        self.last_prediction = self.best_prediction = None

    def call(self, kind, function, *args, **kwargs):
        return self.work.call(kind, function, *args, **kwargs)

    def weighted_residual(self, prediction):
        # Index first: 0 * NaN must never poison gradients on excluded channels.
        return (
            np.where(self.split.train, self.safe_observed - prediction, 0)
            * self.precision
        )

    def observe(
        self,
        iteration,
        state,
        prediction,
        *,
        objective=None,
        update_relative=None,
        sound_speed=None,
        optimizer_diagnostics=None,
    ):
        if self.monitor.reason is not None:
            return self.monitor.reason
        if optimizer_diagnostics is not None and any(
            value is not None and not np.isfinite(value)
            for value in optimizer_diagnostics.values()
        ):
            return self.monitor.finish("numerical_failure")
        train = residual_statistics(
            prediction, self.observed, mask=self.split.train, weights=self.split.weights
        )
        validation = residual_statistics(
            prediction,
            self.observed,
            mask=self.split.validation,
            weights=self.split.weights,
        )
        quality = None
        if self.policy.target_rmse_mps is not None:
            quality = compute_image_metrics(
                sound_speed,
                self.case.ground_truth.sound_speed_mps,
                mask=self.case.grid.roi_mask,
            )["rmse"]
        previous_best = self.monitor.best_iteration
        previous_history_length = len(self.monitor.history)
        reason = self.monitor.observe(
            iteration,
            state,
            residual_norm=train["weighted_residual_norm"],
            observed_norm=train["weighted_observed_norm"],
            objective=(
                0.5 * train["weighted_residual_norm"] ** 2
                if objective is None
                else objective
            ),
            update_relative=update_relative,
            validation_relative=validation["weighted_relative_residual"],
            quality_rmse_mps=quality,
        )
        # A rejected nonfinite state/objective cannot advance the data cache.
        # Its image and its prediction must refer to the same complete iterate.
        if len(self.monitor.history) > previous_history_length:
            if optimizer_diagnostics is not None:
                self.monitor.history[-1]["optimizer_diagnostics"] = dict(
                    optimizer_diagnostics
                )
            self.last_prediction = np.array(prediction, copy=True)
            if self.monitor.best_iteration != previous_best:
                self.best_prediction = self.last_prediction.copy()
        return reason

    def output(self, fallback):
        state, _ = self.monitor.selected_state()
        if state is None:
            state = np.array(fallback, copy=True)
        selected_best = (
            self.policy.restore_best_validation and self.best_prediction is not None
        )
        prediction = self.best_prediction if selected_best else self.last_prediction
        stop = self.monitor.record()
        metrics = {
            "stop_reason": stop["reason"],
            "stopping": stop,
            "iterations": stop["completed_iterations"],
            "iteration_history": self.monitor.history,
            "evaluation_split": self.split.metadata,
            "heldout_role": "validation_for_stopping_and_checkpoint_selection_not_untouched_test",
            "roi_mask_provided": self.case.grid.roi_mask is not None,
            "roi_update_restriction_active": bool(
                coerce_bool(self.config.parameters.get("roi_update_only", False))
                and self.case.grid.roi_mask is not None
            ),
            "residual_curve": [row["residual_norm"] for row in self.monitor.history],
        }
        if getattr(self, "inner_solver_history", None):
            metrics["inner_solver_history"] = self.inner_solver_history
        if prediction is not None:
            evaluation = self.split.evaluate(prediction, self.observed)
            metrics["evaluation"] = evaluation
            train = evaluation["train"]
            metrics["data_relative_residual"] = train["weighted_relative_residual"]
            metrics["data_residual_norm"] = train["weighted_residual_norm"]
            first = self.monitor.history[0]["residual_norm"]
            metrics["initial_data_residual_norm"] = first
            metrics["data_residual_reduction"] = (
                1 - train["weighted_residual_norm"] / first if first else 0.0
            )
        return state, metrics


def add_image_metrics(metrics, sound_speed, case, c0):
    """Ground truth is only read after reconstruction unless oracle stopping is explicit."""
    if case.ground_truth.sound_speed_mps is not None:
        metrics.update(
            compute_image_metrics(
                sound_speed, case.ground_truth.sound_speed_mps, mask=case.grid.roi_mask
            )
        )
        metrics.update(
            compute_baseline_improvement_metrics(
                sound_speed,
                case.ground_truth.sound_speed_mps,
                c0,
                mask=case.grid.roi_mask,
            )
        )
