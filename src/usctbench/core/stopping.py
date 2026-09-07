"""OR stopping rules, explicit work budgets and auditable termination records."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from numbers import Real

import numpy as np

from usctbench.core.config import coerce_bool


@dataclass(frozen=True)
class StopPolicy:
    max_iterations: int = 50
    min_iterations: int = 1
    max_elapsed_s: float | None = None
    max_forward_calls: int | None = None
    max_adjoint_calls: int | None = None
    target_relative_residual: float | None = None
    noise_norm: float | None = None
    discrepancy_factor: float = 1.05
    update_rtol: float | None = 1e-6
    objective_rtol: float | None = 1e-6
    objective_patience: int = 5
    validation_patience: int | None = None
    validation_rtol: float = 1e-4
    restore_best_validation: bool = True
    target_rmse_mps: float | None = None
    allow_ground_truth_stopping: bool = False

    @classmethod
    def from_parameters(cls, parameters, *, default_iterations=50):
        supplied = dict(parameters.get("stopping", {}))
        unknown = supplied.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown stopping settings: {sorted(unknown)}")
        supplied.setdefault(
            "max_iterations", parameters.get("iterations", default_iterations)
        )
        for name in ("restore_best_validation", "allow_ground_truth_stopping"):
            if name in supplied:
                supplied[name] = coerce_bool(supplied[name])
        return cls(**supplied)

    def __post_init__(self):
        for name in ("restore_best_validation", "allow_ground_truth_stopping"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("max_iterations", "min_iterations", "objective_patience"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        integer_fields = (
            "max_iterations",
            "min_iterations",
            "max_forward_calls",
            "max_adjoint_calls",
            "objective_patience",
            "validation_patience",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
            if value is not None:
                object.__setattr__(self, name, int(value))
        for name in ("objective_patience", "validation_patience"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "max_elapsed_s",
            "target_relative_residual",
            "noise_norm",
            "update_rtol",
            "objective_rtol",
            "validation_rtol",
            "target_rmse_mps",
        ):
            value = getattr(self, name)
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be finite and nonnegative")
                object.__setattr__(self, name, float(value))
        if not np.isfinite(self.discrepancy_factor) or self.discrepancy_factor < 1:
            raise ValueError("discrepancy_factor must be finite and >= 1")
        if self.target_rmse_mps is not None and not self.allow_ground_truth_stopping:
            raise ValueError(
                "target_rmse_mps requires explicit allow_ground_truth_stopping=True"
            )


class BudgetExhausted(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class WorkLedger:
    """Count complete array operator calls, including validation and line search.

    Limits are checked *before* starting a call. A running external/NumPy/PDE
    solve cannot be preempted here; elapsed-time limits act between calls.
    Extra source-solve/subset counters are descriptive, not equivalent FLOPs.
    """

    def __init__(self, policy: StopPolicy, *, clock=time.perf_counter):
        self.policy, self.clock = policy, clock
        self.started = clock()
        self.counts = {"forward_calls": 0, "adjoint_calls": 0}

    @property
    def elapsed_s(self):
        return max(0.0, float(self.clock() - self.started))

    def call(self, kind, function, *args, **kwargs):
        if (
            self.policy.max_elapsed_s is not None
            and self.elapsed_s >= self.policy.max_elapsed_s
        ):
            raise BudgetExhausted("time_budget")
        counter = "adjoint_calls" if kind == "adjoint" else "forward_calls"
        limit = getattr(self.policy, f"max_{counter}")
        if limit is not None and self.counts[counter] >= limit:
            raise BudgetExhausted(f"{counter}_budget")
        self.counts[counter] += 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return function(*args, **kwargs)


class StopMonitor:
    """Record complete iterates and the first OR trigger in deterministic order."""

    def __init__(
        self, policy: StopPolicy, work: WorkLedger, *, iteration_unit="iteration"
    ):
        self.policy, self.work = policy, work
        self.iteration_unit = iteration_unit
        self.history = []
        self.reason = None
        self.triggers = []
        self.plateau_count = 0
        self.validation_bad_count = 0
        self.best_validation = None
        self.significant_validation = None
        self.best_state = None
        self.best_iteration = None
        self.last_state = None

    def observe(
        self,
        iteration,
        state,
        *,
        residual_norm,
        observed_norm,
        objective,
        update_relative=None,
        validation_relative=None,
        quality_rmse_mps=None,
    ):
        if self.reason is not None:
            return self.reason
        if (
            isinstance(iteration, (bool, np.bool_))
            or not isinstance(iteration, (int, np.integer))
            or iteration < 0
        ):
            raise ValueError("iteration must be a nonnegative integer")
        if self.history and iteration <= self.history[-1]["iteration"]:
            raise ValueError("complete iteration indices must be strictly increasing")
        values = [residual_norm, observed_norm, objective]
        values += [
            v
            for v in (update_relative, validation_relative, quality_rmse_mps)
            if v is not None
        ]
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(state)):
            return self.finish("numerical_failure")
        if self.policy.target_rmse_mps is not None and quality_rmse_mps is None:
            raise ValueError(
                "ground-truth quality stopping requested but no quality metric was supplied"
            )
        relative = residual_norm / observed_norm if observed_norm > 0 else None
        if self.history and self.policy.objective_rtol is not None:
            previous = self.history[-1]["objective"]
            improvement = (previous - objective) / max(
                abs(previous), np.finfo(float).tiny
            )
            self.plateau_count = (
                self.plateau_count + 1
                if improvement <= self.policy.objective_rtol
                else 0
            )
        if validation_relative is not None:
            if (
                self.best_validation is None
                or validation_relative < self.best_validation
            ):
                self.best_validation = float(validation_relative)
                self.best_state = np.array(state, copy=True)
                self.best_iteration = int(iteration)
            meaningful = self.significant_validation is None or validation_relative < (
                self.significant_validation * (1 - self.policy.validation_rtol)
            )
            if meaningful:
                self.significant_validation = float(validation_relative)
                self.validation_bad_count = 0
            else:
                self.validation_bad_count += 1
        self.last_state = np.array(state, copy=True)
        self.history.append(
            {
                "iteration": int(iteration),
                "elapsed_s": self.work.elapsed_s,
                "residual_norm": float(residual_norm),
                "relative_residual": relative,
                "objective": float(objective),
                "relative_update": update_relative,
                "validation_relative_residual": validation_relative,
                "work": dict(self.work.counts),
            }
        )
        p = self.policy
        triggers = []
        if p.max_elapsed_s is not None and self.work.elapsed_s >= p.max_elapsed_s:
            triggers.append("time_budget")
        for counter in ("forward_calls", "adjoint_calls"):
            limit = getattr(p, f"max_{counter}")
            if limit is not None and self.work.counts[counter] >= limit:
                triggers.append(f"{counter}_budget")
        # Exact fit and safety budgets do not wait for min_iterations.
        if residual_norm == 0:
            triggers.append("exact_data_fit")
        if iteration >= p.min_iterations:
            if (
                p.target_relative_residual is not None
                and relative is not None
                and relative <= p.target_relative_residual
            ):
                triggers.append("target_residual")
            if (
                p.noise_norm is not None
                and residual_norm <= p.discrepancy_factor * p.noise_norm
            ):
                triggers.append("noise_discrepancy")
            if p.target_rmse_mps is not None and quality_rmse_mps <= p.target_rmse_mps:
                triggers.append("oracle_quality_target")
            if (
                p.update_rtol is not None
                and update_relative is not None
                and update_relative <= p.update_rtol
            ):
                triggers.append("small_model_update")
            if (
                p.objective_rtol is not None
                and self.plateau_count >= p.objective_patience
            ):
                triggers.append("objective_plateau")
            if (
                p.validation_patience is not None
                and self.validation_bad_count >= p.validation_patience
            ):
                triggers.append("validation_plateau")
        if iteration >= p.max_iterations:
            triggers.append("max_iterations")
        if triggers:
            self.reason, self.triggers = triggers[0], triggers
        return self.reason

    def finish(self, reason):
        if self.reason is None:
            self.reason, self.triggers = reason, [reason]
        return self.reason

    def selected_state(self):
        if self.policy.restore_best_validation and self.best_state is not None:
            return self.best_state.copy(), self.best_iteration
        if self.last_state is None:
            return None, None
        return self.last_state.copy(), self.history[-1]["iteration"]

    def record(self):
        _, selected = self.selected_state()
        return {
            "reason": self.reason or "not_terminated",
            "triggered_rules": list(self.triggers),
            "completed_iterations": (
                self.history[-1]["iteration"] if self.history else 0
            ),
            "selected_iteration": selected,
            "iteration_unit": self.iteration_unit,
            "elapsed_s": self.work.elapsed_s,
            "work": dict(self.work.counts),
            "policy": asdict(self.policy),
            "budget_scope": "operator_calls; time_checked_between_calls",
            "ground_truth_used_for_stopping": self.policy.target_rmse_mps is not None,
            "has_complete_checkpoint": self.last_state is not None,
            "quality_target_met": any(
                rule in self.triggers
                for rule in (
                    "exact_data_fit",
                    "target_residual",
                    "noise_discrepancy",
                    "oracle_quality_target",
                )
            ),
            "termination_category": (
                "not_terminated"
                if self.reason is None
                else (
                    "budget"
                    if "budget" in self.reason or self.reason == "max_iterations"
                    else (
                        "failure"
                        if self.reason
                        in {
                            "numerical_failure",
                            "linear_solver_breakdown",
                            "line_search_failed",
                        }
                        else (
                            "quality_target"
                            if self.reason
                            in {
                                "exact_data_fit",
                                "target_residual",
                                "noise_discrepancy",
                                "oracle_quality_target",
                            }
                            else "stagnation"
                        )
                    )
                )
            ),
        }
