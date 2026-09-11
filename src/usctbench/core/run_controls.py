"""Minimal production stopping and budget admission, separate from hyperparameters.

The agent owns the outer process deadline; array/PDE calls are cooperatively
interruptible only. A smaller library budget can never enlarge an agent cap.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from usctbench.core.stopping import StopPolicy

NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonnegativeFloat = Annotated[float, Field(strict=True, ge=0)]


class RunControls(BaseModel):
    """Complete iterations, optional time/call caps, and small-update patience.

    A CGLS step, a complete SART sweep and a GN outer step are different units.
    Call caps use the existing WorkLedger charged-operation convention, which
    includes setup/calibration and Jacobian calls; they are not PDE-solve FLOPs.
    """

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )

    max_iterations: NonnegativeInt = Field(
        default=50,
        description="Complete method-specific iterations; zero permits no updates.",
    )
    max_elapsed_s: NonnegativeFloat | None = Field(
        default=None,
        description="Cooperative elapsed-time cap in seconds; an outer process must enforce a hard timeout.",
    )
    min_iterations: NonnegativeInt = Field(
        default=3,
        description="Minimum complete iterations before update-based stopping; budgets take priority.",
    )
    update_rtol: NonnegativeFloat | None = Field(
        default=1e-6,
        description="Relative change in the documented model variable; null disables this criterion.",
    )
    update_patience: Annotated[int, Field(strict=True, ge=1)] = Field(
        default=2,
        description="Consecutive eligible small updates, not objective/residual plateau.",
    )
    max_forward_calls: NonnegativeInt | None = Field(
        default=None,
        description="Optional cap on WorkLedger forward-charged operations, including setup and Jv.",
    )
    max_adjoint_calls: NonnegativeInt | None = Field(
        default=None,
        description="Optional cap on WorkLedger adjoint-charged operations, including normalization.",
    )

    def to_stop_policy(self) -> StopPolicy:
        # Legacy YAML policies remain available explicitly. New production calls
        # do not silently enable objectives, oracle targets or validation selection.
        return StopPolicy(
            **self.model_dump(),
            objective_rtol=None,
            validation_patience=None,
            restore_best_validation=False,
        )

    def capped(self, budget: BudgetCaps | dict | None) -> RunControls:
        """Intersect requested controls with agent admission caps (including zero)."""
        caps = BudgetCaps.model_validate(budget or {})
        values = self.model_dump()
        for target, value in {
            "max_iterations": caps.max_iterations,
            "max_elapsed_s": caps.timeout_s,
            "max_forward_calls": caps.max_forward_calls,
            "max_adjoint_calls": caps.max_adjoint_calls,
        }.items():
            if value is not None:
                values[target] = (
                    value if values[target] is None else min(value, values[target])
                )
        return RunControls.model_validate(values)


class BudgetCaps(BaseModel):
    """Numerical subset of inverse_agent.ComputeBudget.

    Device, memory isolation and thread limits belong to the launching process.
    Unknown keys are rejected rather than silently pretending to enforce them.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_iterations: NonnegativeInt | None = None
    timeout_s: NonnegativeFloat | None = None
    max_forward_calls: NonnegativeInt | None = None
    max_adjoint_calls: NonnegativeInt | None = None
