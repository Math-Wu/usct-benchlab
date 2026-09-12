"""Straight-ray attenuation tomography baseline."""

from __future__ import annotations

from usctbench.algorithms.configuration import validated_run

import numpy as np

from usctbench.algorithms.ray import (
    configured_ray_weights,
    ray_weight_metrics,
    run_with_failure_capture,
    target_attenuation_integral,
)
from usctbench.core.registry import register_algorithm
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase
from usctbench.metrics import compute_image_metrics
from usctbench.operators.straight_ray import StraightRayProjector


class AttenuationSIRTAlgorithm:
    """SIRT reconstruction from log-amplitude attenuation features."""

    name = "attenuation_sirt"

    @validated_run
    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_attenuation_integral(case, projector)
            weights = configured_ray_weights(case, projector, mask, config)
            from usctbench.algorithms._control import InversionControl
            from usctbench.solvers.row_action import row_action

            if config.parameters.get("stopping", {}).get("target_rmse_mps") is not None:
                raise ValueError(
                    "target_rmse_mps is a sound-speed target, not an attenuation target"
                )
            relaxation = float(config.parameters.get("relaxation", 0.8))
            upper = float(config.parameters.get("attenuation_upper_np_per_m", 80.0))
            if not np.isfinite(upper) or upper <= 0:
                raise ValueError(
                    "attenuation_upper_np_per_m must be finite and positive"
                )
            control = InversionControl(
                case,
                config,
                target.reshape(projector.ray_shape),
                default_iterations=50,
                weights=weights.reshape(projector.ray_shape),
                valid_mask=mask.reshape(projector.ray_shape),
                iteration_unit="SIRT sweep",
            )
            control.declare_update(
                "legacy_attenuation",
                "legacy_attenuation_state",
                "legacy_Np/m_frequency_unspecified",
                normalization="norm(q_new-q_old)/norm(q_old); zero denominator is undefined and cannot stop",
            )
            attenuation, metrics = row_action(
                projector,
                control,
                initial=np.zeros(case.grid.shape),
                reference=np.zeros(case.grid.shape),
                project=lambda x: np.clip(x, 0.0, upper),
                to_image=lambda x: x,
                relaxation=relaxation,
            )
            initial_norm = metrics.get("initial_data_residual_norm")
            metrics.update(
                {
                    "attenuation_input_signal_norm": initial_norm,
                    "attenuation_input_has_signal": bool(
                        initial_norm is not None and initial_norm > 0.0
                    ),
                    "attenuation_input_is_surrogate": _is_surrogate_attenuation_case(
                        case
                    ),
                    **ray_weight_metrics(weights, mask, config),
                }
            )
            if case.ground_truth.attenuation_np_per_m is not None:
                metrics.update(
                    compute_image_metrics(
                        attenuation,
                        np.asarray(case.ground_truth.attenuation_np_per_m, dtype=float),
                        mask=case.grid.roi_mask,
                        prefix="attenuation_",
                    )
                )
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                attenuation_np_per_m=attenuation,
                metrics=metrics,
            )

        return run_with_failure_capture(self.name, case, _run)


def _is_surrogate_attenuation_case(case: USCTCase) -> bool:
    text = " ".join(
        str(item) for item in case.metadata.get("measurement_limitations", [])
    )
    text = f"{text} {case.metadata.get('attenuation_note', '')} {case.metadata.get('feature_provenance', '')}".lower()
    return "zero surrogate" in text or "surrogate" in text and "log_amp" in text


def register_attenuation_algorithm(*, replace: bool = False) -> None:
    from usctbench.core.algorithm_specs import SPECS

    register_algorithm(
        "attenuation_sirt",
        AttenuationSIRTAlgorithm,
        specification=SPECS["attenuation_sirt"],
        description="Straight-ray SIRT attenuation reconstruction.",
        tags=("ray", "attenuation"),
        replace=replace,
    )


__all__ = ["AttenuationSIRTAlgorithm", "register_attenuation_algorithm"]
