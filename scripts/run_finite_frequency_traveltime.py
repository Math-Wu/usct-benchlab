#!/usr/bin/env python3
"""Controlled multiband nonlinear traveltime experiment on existing raw pressure.

Not a renamed straight-ray or geometric Bent algorithm. No GT enters preparation,
weights, updates, or stopping. --audit-only uses GT only as an explicit forward
diagnostic. See --help; output must be a fresh directory outside the repository.
"""

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usctbench.algorithms._control import InversionControl
from usctbench.core.io import read_case_hdf5, write_result_hdf5
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, GeometrySpec
from usctbench.data.validation_acquisition import read_validation_acquisition
from usctbench.data.waveforms import _read_channels, pressure_spectrum
from usctbench.metrics import compute_regional_image_metrics
from usctbench.operators.base import Linearization
from usctbench.operators.model_space import (
    BilinearBasis,
    FineGridRegularizer,
    SpatialGradient,
)
from usctbench.operators.forward.band_delay import (
    BandCorrelationDelay,
    BandDelayLinearization,
    FiniteFrequencyTravelTimeForward,
)
from usctbench.operators.forward.ray_born import RayBornForward, RayBornOperator
from usctbench.solvers.nonlinear import nonlinear_least_squares


class BasisJacobian:
    def __init__(self, jacobian, basis):
        self.jacobian, self.basis = jacobian, basis
        self.grid = basis.grid

    def forward(self, values):
        return self.jacobian.forward(self.basis.forward(values))

    def adjoint(self, values):
        return self.basis.adjoint(self.jacobian.adjoint(values))


class CoefficientForward:
    """Fine wave propagation with fewer independent squared-slowness unknowns."""

    def __init__(self, forward, basis):
        self.physics, self.basis = forward, basis
        self.grid = basis.grid

    def linearize(self, coefficients):
        lin = self.physics.linearize(self.basis.forward(coefficients))
        return Linearization(
            lin.value,
            BasisJacobian(lin.jacobian, self.basis),
            derivative_kind=lin.derivative_kind + "_bilinear_basis",
        )

    def forward(self, coefficients):
        return self.linearize(coefficients).value

    def __getattr__(self, name):
        return getattr(self.physics, name)


class ObservedCorrelationDelay:
    """Data-dependent misfit: correlate predicted and observed calibrated pulses.

    The additive offset only preserves the existing nonzero residual denominator.
    Minimizing the residual means minimizing their direct correlation lag, not
    fitting a difference of two independently picked water-relative delays.
    """

    def __init__(self, base, observed_ratio, offset):
        self.base = base
        self.frequencies_hz = base.frequencies_hz
        self.data_shape = base.data_shape
        self.observed_ratio = np.array(observed_ratio, complex, copy=True)
        self.offset = np.array(offset, float, copy=True)

    def linearize(self, predicted_ratio):
        multiplier = self.observed_ratio.conj()
        active = np.any(self.base.power > 0, axis=0)
        multiplier = np.where(active, multiplier, 0)
        fit = self.base.linearize(predicted_ratio * multiplier)
        return BandDelayLinearization(
            fit.value + self.offset,
            fit.valid,
            fit.coherence,
            fit.peak_gap,
            fit.coefficient * multiplier[None],
            stationarity_error_s=fit.stationarity_error_s,
            local_concavity_margin=fit.local_concavity_margin,
        )


def physical_regularization(length_wavelengths, maximum_frequency, spacing):
    if not length_wavelengths:
        return "laplacian"
    length_m = length_wavelengths * 1500 / maximum_frequency
    return tuple(float((length_m / h) ** 2) for h in spacing)


def common_observable_evaluation(
    predicted, measured, water_picker, direct_picker, split
):
    """Post-hoc cross-score a selected model on fixed, observation-only masks.

    Inputs are pressure ratios to independent water references. No refitting,
    peak-dependent mask selection, or use of GT is permitted here.
    """
    observed = water_picker.linearize(measured)
    picked = water_picker.linearize(predicted)
    direct = direct_picker.linearize(predicted * measured.conj())
    report = {}
    for name, mask in (("train", split.train), ("validation", split.validation)):
        mask = np.asarray(mask, bool)
        pairs = np.any(mask, axis=0)
        item = {
            "pair_count": int(pairs.sum()),
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        }
        for key, difference, valid in (
            (
                "water_peak_difference",
                picked.value - observed.value,
                picked.valid & observed.valid,
            ),
            ("direct_correlation_lag", direct.value, direct.valid),
        ):
            accepted = mask & valid & np.isfinite(difference)
            item[key] = {
                "requested_count": int(mask.sum()),
                "valid_count": int(accepted.sum()),
                "invalid_fraction": (
                    float(1 - accepted.sum() / mask.sum()) if mask.any() else None
                ),
                "rmse_us_valid_only": (
                    float(1e6 * np.sqrt(np.mean(difference[accepted] ** 2)))
                    if accepted.any()
                    else None
                ),
                "rmse_us_all_requested": (
                    float(1e6 * np.sqrt(np.mean(difference[mask] ** 2)))
                    if mask.any() and np.all(accepted[mask])
                    else None
                ),
            }
        error = np.linalg.norm((predicted - measured)[:, pairs])
        for key, denominator in (
            (
                "water_calibrated_pressure_relative_residual",
                np.linalg.norm(measured[:, pairs]),
            ),
            (
                "water_calibrated_scattered_relative_residual",
                np.linalg.norm((measured - 1)[:, pairs]),
            ),
        ):
            item[key] = (
                float(error / denominator)
                if denominator > 0 and np.isfinite(error)
                else None
            )
        report[name] = item
    return report


def regularization_weight(diagonal, ratio, absolute=None):
    """Allow changing observations without implicitly changing the prior weight."""
    value = ratio if absolute is None else absolute
    if not np.isfinite(value) or value < 0:
        raise ValueError("regularization weight must be finite and nonnegative")
    if absolute is not None:
        return float(absolute)
    diagonal = np.asarray(diagonal)
    positive = diagonal[diagonal > 0]
    if not np.isfinite(diagonal).all() or not positive.size:
        raise ValueError("cannot scale regularization without finite sensitivity")
    return float(ratio * np.median(positive))


def continuation_identity(case, measured, water, frequencies, control, manifest):
    """Bind a staged solve to its physical data, split, prior and numerical model."""
    settings = manifest["arguments"]
    fixed = {
        key: settings[key]
        for key in (
            "delay_reference",
            "damping_absolute",
            "prior_reference",
            "regularization_length_wavelengths",
            "regularization_penalty",
            "tv_transition_mps",
            "smooth_sigma",
            "mean_data_loss",
            "model_size",
            "gradient_rtol",
        )
    }
    fixed["source_sha256"] = manifest["source_sha256"]
    digest = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode())
    arrays = [
        case.grid.shape,
        case.grid.spacing_m,
        case.grid.origin_m,
        case.geometry.tx_pos_m,
        case.geometry.rx_pos_m,
        frequencies,
        measured,
        water,
    ]
    for mask in (control.split.train, control.split.validation):
        if not np.array_equal(mask.any(axis=0), mask.all(axis=0)):
            raise ValueError("continuation requires the same pair mask in every band")
        arrays.append(mask.all(axis=0))
    arrays.append(control.precision.sum(axis=0))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def result_initialization(path, case, identity, *, bounds=(1300, 1700)):
    """Read the selected final image, not the possibly different last checkpoint."""
    previous = json.loads((path.parent / "manifest.json").read_text())
    if previous.get("continuation_identity_sha256") != identity:
        raise ValueError("continuation data/grid/split/prior identity mismatch")
    provenance = yaml.safe_load((path.parent / "metadata.yaml").read_text())
    if provenance.get("source_changed_during_run") is not False:
        raise ValueError("continuation requires verified unchanged numerical source")
    snapshot = path.read_bytes()
    with h5py.File(io.BytesIO(snapshot)) as handle:
        if (
            handle.attrs.get("record_type") != "ReconstructionResult"
            or handle.attrs.get("algorithm") != "finite_frequency_traveltime_gn"
            or handle.attrs.get("case_id") != case.case_id
            or handle.attrs.get("status") != "success"
        ):
            raise ValueError("continuation requires a matching finite-frequency result")
        speed = np.asarray(handle["sound_speed_mps"])
        metrics = json.loads(handle.attrs["metrics_json"])
    stop = metrics.get("stopping", {})
    if (
        stop.get("termination_category")
        not in {"budget", "stagnation", "quality_target"}
        or stop.get("selected_iteration", 0) < 1
        or stop.get("ground_truth_used_for_stopping") is not False
        or metrics.get("stop_reason")
        in {"numerical_failure", "line_search_failed", "linear_solver_breakdown"}
    ):
        raise ValueError(
            "continuation refuses failed, unstarted or GT-selected results"
        )
    if (
        speed.shape != case.grid.shape
        or np.iscomplexobj(speed)
        or not np.isfinite(speed).all()
        or np.any(speed < bounds[0])
        or np.any(speed > bounds[1])
    ):
        raise ValueError("continuation speed must match the grid and physical bounds")
    return 1 / speed**2, {
        "method": "selected_completed_result",
        "source_result": str(path.resolve()),
        "source_result_sha256": hashlib.sha256(snapshot).hexdigest(),
        "continuation_identity_sha256": identity,
        "source_stop_reason": metrics["stop_reason"],
        "source_selected_iteration": stop["selected_iteration"],
        "source_completed_iterations": stop["completed_iterations"],
        "source_budget_elapsed_s": stop["elapsed_s"],
        "source_work": stop["work"],
        "source_initialization": metrics.get("initialization_qc"),
        "ground_truth_used": False,
    }


def ring_pair_mask(distance, tx_indices, rx_indices, fraction=None, *, elements=128):
    """Exclude a symmetric fraction of the ring around each receiver.

    Indices refer to the ORIGINAL uniform ring, not subsampled array rows.
    A quarter means +/-45 degrees; boundary ties and self channels are excluded.
    None preserves the historical half-diameter distance mask exactly.
    """
    distance = np.asarray(distance)
    tx, rx = np.asarray(tx_indices), np.asarray(rx_indices)
    if (
        not isinstance(elements, int)
        or elements < 2
        or tx.ndim != 1
        or rx.ndim != 1
        or tx.dtype.kind not in "iu"
        or rx.dtype.kind not in "iu"
        or distance.shape != (tx.size, rx.size)
        or not np.isfinite(distance).all()
        or np.any(distance < 0)
        or not np.any(distance > 0)
        or np.any(tx >= elements)
        or np.any(rx >= elements)
        or np.any(tx < 0)
        or np.any(rx < 0)
    ):
        raise ValueError("require valid ring indices and pair distances")
    if fraction is None:
        return distance > distance.max() * 0.5
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("excluded neighbor fraction must be in [0, 1)")
    tx, rx = tx.astype(np.int64), rx.astype(np.int64)
    separation = np.abs(tx[:, None] - rx[None])
    separation = np.minimum(separation, elements - separation)
    return (separation > fraction * elements / 2) & (distance > 0)


def shared_channel_validation(valid, tx_parent, rx_parent):
    """Same physical validation pairs for nested 64/128 acquisitions.

    New transmitters remain usable in training, but not against validation
    receivers. Reciprocal exclusion is subsequently enforced by DataSplit.
    """
    receivers = np.sort(
        np.random.default_rng(42).choice(np.arange(0, 128, 2), 8, replace=False)
    )
    if not np.isin(receivers, rx_parent).all():
        raise ValueError("shared validation requires all reference receivers")
    common_tx = np.asarray(tx_parent) % 2 == 0
    heldout_rx = np.isin(rx_parent, receivers)
    keep = common_tx[:, None] | ~heldout_rx[None]
    return valid & keep, {
        "receiver_indices": np.flatnonzero(heldout_rx).tolist(),
        "seed": 42,
        "exclude_reciprocal": True,
    }


def normalize_training_weight(control):
    """Mean training data loss; apply the same scale to validation reporting."""
    total = float(control.precision.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("require a positive finite training weight sum")
    control.split = replace(control.split, weights=control.split.weights / total)
    control.precision = control.precision / total
    return total


def phase_initialization(case, measured, water, frequencies, distance, control, args):
    """Reuse the existing rWave phase-CGLS recipe, with this run's training split."""
    from types import SimpleNamespace
    from scipy.ndimage import gaussian_filter
    from usctbench.data.phase_delay import phase_slope_delays
    from usctbench.operators.forward.straight_ray import StraightRayProjector
    from usctbench.solvers.least_squares import normal_step

    # No frequency holdout is claimed: overlapping bands share the full spectrum.
    train = np.broadcast_to(np.all(control.split.train, axis=0), measured.shape)
    delays, precision, qc = phase_slope_delays(
        measured,
        water,
        frequencies,
        distance,
        train,
        bounds=(1300, 1700),
        max_phase_rms=0.2,
    )
    if not np.any(precision):
        raise ValueError("no valid training phase-delay channels for initialization")
    training = SimpleNamespace(
        observed=delays,
        precision=precision,
        call=control.call,
        work=control.work,
    )
    projector = control.call(
        "initialization_setup", StraightRayProjector.from_case, case
    )
    delta = normal_step(
        projector,
        np.where(precision > 0, np.nan_to_num(delays) * precision, 0),
        np.zeros(case.grid.shape),
        training,
        iterations=args.initialization_iterations,
        damping=args.initialization_lambda**2,
        regularization="laplacian",
    )
    sigma = args.initialization_smooth_mm * 1e-3 / np.array(case.grid.spacing_m)
    delta = gaussian_filter(delta, sigma, mode="nearest")
    qc.update(
        {
            "iterations": args.initialization_iterations,
            "lambda": args.initialization_lambda,
            "smooth_mm": args.initialization_smooth_mm,
            "train_pair_sha256": hashlib.sha256(train[0].tobytes()).hexdigest(),
        }
    )
    return np.clip(1 / 1500 + delta, 1 / 1700, 1 / 1300) ** 2, qc


class ProgressControl(InversionControl):
    def __init__(self, *args, out, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = out
        self.basis = None

    def observe(self, iteration, state, prediction, **kwargs):
        previous = len(self.monitor.history)
        reason = super().observe(iteration, state, prediction, **kwargs)
        if len(self.monitor.history) > previous:
            record = self.monitor.history[-1]
            print(json.dumps(record), flush=True)
            (self.out / "progress.json").write_text(
                json.dumps(self.monitor.history, indent=2)
            )
            partial = self.out / "checkpoint.partial.npz"
            image = state if self.basis is None else self.basis.forward(state)
            np.savez_compressed(
                partial,
                iteration=iteration,
                squared_slowness=image,
                coefficients=state,
                prediction=prediction,
            )
            partial.replace(self.out / "checkpoint.npz")
        return reason


def main():
    executed_source = Path(__file__).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bands", choices=("single", "multi", "low"), default="multi")
    parser.add_argument(
        "--delay-reference", choices=("water", "observed"), default="water"
    )
    parser.add_argument("--tx-stride", type=int, default=2, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--ring-stride",
        type=int,
        choices=(1, 2),
        default=2,
        help="parent 128-ring sampling: 2 gives 64 RX, 1 gives 128 RX; TX also uses tx-stride",
    )
    parser.add_argument(
        "--shared-64-validation",
        action="store_true",
        help="same even-parent TX and eight physical held-out RX in 64/128 comparisons",
    )
    parser.add_argument(
        "--mean-data-loss",
        action="store_true",
        help="normalize training precision to sum one; use an explicitly scaled prior coefficient",
    )
    parser.add_argument(
        "--green-cache-gib",
        type=float,
        default=4,
        help="field cache budget per CPU/GPU copy; 128-channel runs may require 8 GiB or more",
    )
    parser.add_argument(
        "--exclude-neighbor-fraction",
        type=float,
        help="total ring fraction excluded around each receiver (0.25: +/-45 deg); default: legacy half-diameter mask",
    )
    parser.add_argument(
        "--frequency-count",
        type=int,
        default=61,
        help="spectral quadrature points; verify kernel convergence for each acquisition",
    )
    parser.add_argument("--outer-iterations", type=int, default=15)
    parser.add_argument(
        "--optimizer", choices=("backtracking_gn", "trf"), default="backtracking_gn"
    )
    parser.add_argument(
        "--model-size",
        type=int,
        help="optional coefficient grid; propagation stays at case resolution",
    )
    parser.add_argument("--inner-iterations", type=int, default=12)
    parser.add_argument(
        "--inner-solver", choices=("normal_cg", "lsmr", "lsqr"), default="lsmr"
    )
    parser.add_argument("--inner-rtol", type=float, default=1e-3)
    parser.add_argument("--audit-inner-caps", nargs="+", type=int)
    parser.add_argument(
        "--audit-gradient-steps",
        nargs="+",
        type=float,
        help="saved-checkpoint audit: difference-step multipliers of a 0.002 relative model direction",
    )
    parser.add_argument(
        "--skip-gradient-fd",
        action="store_true",
        help="saved-checkpoint solver audit only; do not repeat nonlinear gradient FD",
    )
    parser.add_argument(
        "--audit-inner-methods",
        nargs="+",
        choices=("normal_cg", "lsmr", "lsqr", "lsmr_column_rms"),
    )
    parser.add_argument("--inner-atol", type=float, default=1e-10)
    parser.add_argument("--inner-btol", type=float, default=1e-10)
    parser.add_argument("--inner-conlim", type=float, default=1e8)
    parser.add_argument(
        "--inner-preconditioner", choices=("none", "column_rms"), default="none"
    )
    parser.add_argument(
        "--compare-inner-solvers",
        action="store_true",
        help="saved-checkpoint audit of CG, LSMR, LSQR at identical work caps",
    )
    parser.add_argument(
        "--gradient-rtol",
        type=float,
        default=1e-8,
        help="regularized gradient tolerance relative to initial gradient (TRF uses bound scaling); zero disables",
    )
    parser.add_argument("--seconds", type=float, default=7200)
    damping_group = parser.add_mutually_exclusive_group()
    damping_group.add_argument("--damping-ratio", type=float, default=0.02)
    damping_group.add_argument(
        "--damping-absolute",
        type=float,
        help="fixed coefficient of the spatial penalty; independent of band Jacobian",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0,
        help="direction smoothing in propagation-grid pixels",
    )
    parser.add_argument("--regularization-length-wavelengths", type=float, default=0)
    parser.add_argument(
        "--regularization-penalty",
        choices=("quadratic_laplacian", "smooth_tv"),
        default="quadratic_laplacian",
        help="smooth_tv uses a physical first-difference penalty and requires TRF",
    )
    parser.add_argument(
        "--tv-transition-mps",
        type=float,
        default=5.0,
        help="smooth-TV transition, converted to squared slowness at 1500 m/s",
    )
    parser.add_argument(
        "--initialization", choices=("water", "phase_cgls", "result"), default="water"
    )
    parser.add_argument(
        "--initial-result",
        type=Path,
        help="previous selected result.h5; requires result initialization, fixed water prior and absolute damping",
    )
    parser.add_argument(
        "--prior-reference",
        choices=("initial", "water"),
        default="initial",
        help="spatial penalty anchor; water holds the prior fixed when changing initialization",
    )
    parser.add_argument("--initialization-iterations", type=int, default=80)
    parser.add_argument("--initialization-lambda", type=float, default=0.02)
    parser.add_argument("--initialization-smooth-mm", type=float, default=3)
    parser.add_argument("--minimum-peak-gap", type=float, default=0.02)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--solver-audit-checkpoint",
        type=Path,
        help="GT-free objective/normal-equation audit of a saved checkpoint",
    )
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="water-background derivative quadrature audit; no reconstruction or GT",
    )
    parser.add_argument("--gpu", type=int, help="optional CuPy device; CPU by default")
    args = parser.parse_args()
    if (args.initial_result is not None) != (args.initialization == "result"):
        parser.error("initial-result requires initialization=result, and vice versa")
    if args.initial_result is not None and (
        args.prior_reference != "water"
        or args.damping_absolute is None
        or args.audit_only
        or args.kernel_only
        or args.solver_audit_checkpoint is not None
    ):
        parser.error(
            "result initialization requires inversion with fixed absolute damping and water prior"
        )
    if args.audit_gradient_steps is not None and (
        args.solver_audit_checkpoint is None
        or args.skip_gradient_fd
        or not np.isfinite(args.audit_gradient_steps).all()
        or np.any(np.asarray(args.audit_gradient_steps) <= 0)
    ):
        parser.error(
            "audit-gradient-steps requires a checkpoint, enabled gradient FD and positive finite steps"
        )
    damping_input = (
        args.damping_ratio if args.damping_absolute is None else args.damping_absolute
    )
    if (
        sum(
            [
                args.audit_only,
                args.kernel_only,
                args.solver_audit_checkpoint is not None,
            ]
        )
        > 1
    ):
        parser.error(
            "audit-only, kernel-only and solver-audit-checkpoint are mutually exclusive"
        )
    if args.model_size is not None and args.initialization != "water":
        parser.error("model-size control currently requires water initialization")
    if args.optimizer == "trf" and args.smooth_sigma:
        parser.error("TRF uses the explicit penalty; set smooth-sigma to zero")
    if args.regularization_penalty == "smooth_tv" and (
        args.optimizer != "trf"
        or args.regularization_length_wavelengths <= 0
        or damping_input <= 0
        or not np.isfinite(args.tv_transition_mps)
        or args.tv_transition_mps <= 0
        or args.audit_only
        or args.kernel_only
        or args.solver_audit_checkpoint is not None
    ):
        parser.error(
            "smooth_tv requires TRF inversion, positive length/damping/transition; quadratic audits are separate"
        )
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(repo) or args.frequency_count < 9:
        parser.error("use external output and at least 9 frequencies")
    if (
        not np.isfinite(damping_input)
        or damping_input < 0
        or not np.isfinite(args.gradient_rtol)
        or not 0 <= args.gradient_rtol < 1
        or 0 < args.gradient_rtol <= np.finfo(float).eps
        or not 0 <= args.minimum_peak_gap < 1
        or not np.isfinite(args.smooth_sigma)
        or args.smooth_sigma < 0
        or not np.isfinite(args.regularization_length_wavelengths)
        or args.regularization_length_wavelengths < 0
        or args.initialization_iterations < 1
        or not np.isfinite(args.initialization_lambda)
        or args.initialization_lambda < 0
        or not np.isfinite(args.initialization_smooth_mm)
        or args.initialization_smooth_mm < 0
    ):
        parser.error("invalid damping or feature/initialization setting")
    if args.exclude_neighbor_fraction is not None and (
        not np.isfinite(args.exclude_neighbor_fraction)
        or not 0 <= args.exclude_neighbor_fraction < 1
    ):
        parser.error("exclude-neighbor-fraction must be in [0, 1)")
    if not np.isfinite(args.green_cache_gib) or args.green_cache_gib <= 0:
        parser.error("green-cache-gib must be finite and positive")
    if args.mean_data_loss and args.damping_absolute is None:
        parser.error("mean-data-loss requires damping-absolute in mean-loss units")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "executed_experiment.py").write_bytes(executed_source)
    helper_hash = None
    if args.optimizer == "trf":
        helper_source = (
            Path(__file__).with_name("_finite_frequency_trf.py").read_bytes()
        )
        (args.out / "_finite_frequency_trf.py").write_bytes(helper_source)
        helper_hash = hashlib.sha256(helper_source).hexdigest()
    acquisition = read_validation_acquisition(args.acquisition)
    case = read_case_hdf5(args.case)
    if args.model_size is not None:
        if args.model_size < 2 or args.model_size > min(case.grid.shape):
            parser.error("model-size must be between 2 and the propagation size")
        if case.grid.shape[0] != case.grid.shape[1]:
            parser.error(
                "this square coefficient-grid experiment requires a square case"
            )
    if case.metadata.get("input_sha256") != acquisition.manifest["input_sha256"]:
        raise ValueError("case/acquisition identity mismatch")
    if not np.array_equal(
        case.ground_truth.sound_speed_mps, acquisition.image_speed_mps
    ):
        raise ValueError("property map identity mismatch")
    tx = np.arange(0, 128, args.ring_stride * args.tx_stride)
    rx = np.arange(0, 128, args.ring_stride)
    case.geometry = GeometrySpec(
        tx_pos_m=acquisition.positions_yx_m[tx], rx_pos_m=acquisition.positions_yx_m[rx]
    )
    case.grid.roi_mask = None
    distance = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    pair_valid = ring_pair_mask(distance, tx, rx, args.exclude_neighbor_fraction)
    aperture = {
        "policy": (
            "legacy_distance_greater_than_half_diameter"
            if args.exclude_neighbor_fraction is None
            else "symmetric_original_ring_neighbor_exclusion"
        ),
        "requested_fraction": args.exclude_neighbor_fraction,
        "parent_ring_elements": 128,
        "excluded_pair_count": int((~pair_valid).sum()),
        "excluded_pair_fraction": float((~pair_valid).mean()),
        "excluded_tx_per_receiver": (~pair_valid).sum(axis=0).tolist(),
        "pair_mask_sha256": hashlib.sha256(pair_valid.tobytes()).hexdigest(),
        "scope": "feature_QC_training_residual_initialization_and_adjoint",
    }
    with h5py.File(args.acquisition / "pressure.mat") as f:
        t = np.asarray(f["time"]).ravel()
        p = _read_channels(
            f["full_dataset"], ("tx", "rx", "time"), tx, rx, len(t), 2**31
        )
        w = _read_channels(
            f["water_dataset"], ("tx", "rx", "time"), tx, rx, len(t), 2**31
        )
    low = 0.4 * acquisition.manifest["source_frequency_hz"]
    high = acquisition.manifest["maximum_frequency_hz"]
    frequencies = np.linspace(low, high, args.frequency_count)
    measured, water = pressure_spectrum(p, t, frequencies), pressure_spectrum(
        w, t, frequencies
    )
    del p, w
    coordinate = np.linspace(0, 1, len(frequencies))
    all_windows = np.array(
        [np.maximum(0, 1 - np.abs(coordinate - c) / 0.5) for c in (0.2, 0.5, 0.8)]
    )
    if args.bands == "single":
        bands = np.ones((1, len(frequencies)))
    else:
        # Fixed overlapping spectral power windows, never selected using GT.
        bands = all_windows
        if args.bands == "low":
            bands = bands[:1]
    bounds = (1300, 1700)
    lower = distance * (1 / bounds[1] - 1 / 1500) - 1e-7
    upper = distance * (1 / bounds[0] - 1 / 1500) + 1e-7
    observation = BandCorrelationDelay(
        frequencies, np.where(pair_valid[None], water, 0), bands, lower, upper
    )
    water_picker = observation
    ratio = np.divide(
        measured, water, out=np.zeros_like(water), where=np.abs(water) > 0
    )
    observed = observation.linearize(ratio)
    # All objective modes share full-spectrum QC and equal pair weights.
    # "low" limits objective information, not the acquisition/QC budget.
    quality = BandCorrelationDelay(
        frequencies,
        np.where(pair_valid[None], water, 0),
        np.vstack([np.ones(len(frequencies)), all_windows]),
        lower,
        upper,
    ).linearize(ratio)
    common_pairs = pair_valid & np.all(
        quality.valid
        & (quality.coherence >= 0.6)
        & (quality.peak_gap >= args.minimum_peak_gap),
        axis=0,
    )
    valid = np.broadcast_to(common_pairs, observed.value.shape).copy()
    evaluation = {
        "receiver_fraction": 0.125,
        "seed": 42,
        "exclude_reciprocal": True,
    }
    if args.shared_64_validation:
        valid, evaluation = shared_channel_validation(valid, tx, rx)
    weights = valid.astype(float) / len(bands)
    relative_base = BandCorrelationDelay(
        frequencies,
        np.where(pair_valid[None], water, 0),
        bands,
        lower - upper,
        upper - lower,
    )
    if args.delay_reference == "observed":
        # Relative delay between two admissible media; bounds use no observed
        # values or GT and therefore cannot couple training/validation channels.
        observation = ObservedCorrelationDelay(relative_base, ratio, observed.value)
    manifest = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "acquisition_input_sha256": acquisition.manifest["input_sha256"],
        "experiment_script_sha256": hashlib.sha256(executed_source).hexdigest(),
        "trf_helper_sha256": helper_hash,
        "frequency_convention": "positive Fourier transform; time exp(-i omega t)",
        "frequencies_hz": frequencies.tolist(),
        "band_power_windows": bands.tolist(),
        "tx_parent_indices": tx.tolist(),
        "rx_parent_indices": rx.tolist(),
        "source_response": "fixed independent water ratio, no breast calibration",
        "gt_used_for_adaptation_or_stopping": False,
        "gt_role": (
            "forward_attribution_only"
            if args.audit_only
            else (
                "identity_check_only"
                if args.kernel_only
                else "posthoc_image_metrics_only"
            )
        ),
        "observable": (
            "bounded_predicted_observed_correlation_lag"
            if args.delay_reference == "observed"
            else "bounded_band_correlation_peak_not_geometric_first_arrival"
        ),
        "band_overlap": "not independent frequency holdout",
        "quality_policy": "common full-spectrum QC intersection across broad/three bands",
        "peak_derivative_qc": {
            "stationary_refined_peak_required": True,
            "local_concavity_certificate": "curvature > 2 * lag_step * bound_abs_third_derivative",
            "global_peak_uniqueness_certified": False,
            "peak_gap_scope": "refined_sampled_located_competing_maxima",
            "valid_channels_per_qc_band": quality.valid.sum(axis=(1, 2)).tolist(),
        },
        "weight_policy": "uniform shared pairs, normalized by band count; not noise precision",
        "aperture": aperture,
        "valid_fraction": float(valid.sum() / (len(bands) * pair_valid.sum())),
        "source_sha256": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (repo / "src/usctbench").rglob("*.py")
        },
    }
    print("prepared", args.bands, "valid", manifest["valid_fraction"], flush=True)
    if valid.sum() < 32:
        raise ValueError("too few reliable band observations")
    parameters = {
        "stopping": {
            "max_iterations": args.outer_iterations,
            "max_elapsed_s": args.seconds,
            "validation_patience": 5,
            "update_rtol": 1e-7,
            "objective_rtol": 1e-6,
        },
        "evaluation": evaluation,
        "roi_update_only": False,
    }
    config = AlgorithmConfig(
        name="finite_frequency_traveltime_gn", parameters=parameters
    )
    control = ProgressControl(
        case,
        config,
        observed.value,
        default_iterations=args.outer_iterations,
        weights=weights,
        valid_mask=valid,
        iteration_unit="finite-frequency traveltime accepted outer step",
        out=args.out,
    )
    weight_divisor = normalize_training_weight(control) if args.mean_data_loss else 1.0
    manifest["training_weight_divisor"] = weight_divisor
    manifest["weight_policy"] = (
        "uniform pairs/band count; sum training precision equals one; not noise precision"
        if args.mean_data_loss
        else manifest["weight_policy"]
    )
    manifest["validation_protocol"] = (
        "shared_even_parent_TX_and_eight_parent_RX"
        if args.shared_64_validation
        else "seeded_receiver_fraction"
    )
    manifest["continuation_identity_sha256"] = continuation_identity(
        case, measured, water, frequencies, control, manifest
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(
        args.out / "observations.npz",
        delay_s=observed.value,
        valid=valid,
        weights=weights / weight_divisor,
        coherence=observed.coherence,
        peak_gap=observed.peak_gap,
        stationarity_error_s=observed.stationarity_error_s,
        local_concavity_margin=observed.local_concavity_margin,
        train_mask=control.split.train,
        validation_mask=control.split.validation,
    )
    reference = RayBornOperator(case.grid, case.geometry, frequencies).background_data()
    pressure = RayBornForward(
        case.grid,
        case.geometry,
        frequencies,
        green_backend="volume_integral",
        green_solver_rtol=1e-6,
        green_solver_maxiter=30,
        max_cache_bytes=int(args.green_cache_gib * 2**30),
        budget_check=control.work.check_time,
        green_device=args.gpu,
    )
    forward = FiniteFrequencyTravelTimeForward(pressure, observation, reference)
    if args.kernel_only:
        lin = forward.linearize(np.full(case.grid.shape, 1 / 1500**2))
        # A deterministic near-diametric channel, independent of sample labels.
        source = 0
        receiver = int(np.argmax(distance[source]))
        y = np.zeros_like(observed.value)
        y[:, source, receiver] = 1 / len(bands)
        kernel = lin.jacobian.adjoint(y)
        gradient = lin.jacobian.adjoint(control.weighted_residual(lin.value))
        yy, xx = np.indices(case.grid.shape)
        points = np.stack([yy.ravel(), xx.ravel()], axis=1)
        points = np.asarray(case.grid.origin_m) + (points + 0.5) * case.grid.spacing_m
        excess = (
            np.linalg.norm(points - case.geometry.tx_pos_m[source], axis=1)
            + np.linalg.norm(points - case.geometry.rx_pos_m[receiver], axis=1)
            - distance[source, receiver]
        ) / 1500
        np.savez_compressed(
            args.out / "kernel_audit.npz", kernel=kernel, gradient=gradient
        )
        report = {
            "frequency_step_hz": float(frequencies[1] - frequencies[0]),
            "quadrature_replica_period_us": float(
                1e6 / (frequencies[1] - frequencies[0])
            ),
            "maximum_channel_excess_time_us": float(excess.max() * 1e6),
            "tx_index": source,
            "rx_index": receiver,
            "kernel_norm": float(np.linalg.norm(kernel)),
            "elapsed_s": control.work.elapsed_s,
            "gt_used_in_kernel_or_gradient": False,
        }
        (args.out / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    if args.audit_only:
        audit_model = 1 / acquisition.image_speed_mps**2
        water_model = np.full(case.grid.shape, 1 / 1500**2)
        linearization = control.call(
            "audit_linearization", forward.linearize, water_model
        )
        linear_prediction = linearization.value + control.call(
            "audit_jacobian", linearization.jacobian.forward, audit_model - water_model
        )
        del linearization
        prediction = control.call("audit_forward", forward.forward, audit_model)
        report = {
            "band_rmse_ns": [
                float(
                    np.sqrt(np.mean((prediction[k] - observed.value[k])[valid[k]] ** 2))
                    * 1e9
                )
                for k in range(len(bands))
            ],
            "evaluation": control.split.evaluate(prediction, observed.value),
            "water_linearization": {
                "model": "F(m_water) + J(m_water) * (m_GT - m_water)",
                "evaluation": control.split.evaluate(linear_prediction, observed.value),
                "difference_vs_nonlinear_prediction": control.split.evaluate(
                    linear_prediction, prediction
                ),
                "nonlinear_relinearization_used": False,
                "ground_truth_role": "forward_attribution_only_not_initialization",
            },
            "elapsed_s": control.work.elapsed_s,
            "work": dict(control.work.counts),
        }
        np.savez_compressed(
            args.out / "forward_at_gt.npz",
            prediction_s=prediction,
            water_linear_prediction_s=linear_prediction,
            observed_s=observed.value,
            valid=valid,
        )
        (args.out / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    initial = np.full(case.grid.shape, 1 / 1500**2)
    initialization_qc = None
    if args.initialization == "phase_cgls":
        initial, initialization_qc = phase_initialization(
            case, measured, water, frequencies, distance, control, args
        )
    elif args.initialization == "result":
        initial, initialization_qc = result_initialization(
            args.initial_result, case, manifest["continuation_identity_sha256"]
        )
    prior = (
        initial.copy()
        if args.prior_reference == "initial"
        else np.full(case.grid.shape, 1 / 1500**2)
    )
    np.savez_compressed(
        args.out / "initial_model.npz",
        squared_slowness=initial,
        prior_squared_slowness=prior,
    )
    water_lin = control.call("initial_jacobian", forward.linearize, initial)
    rng = np.random.default_rng(42)
    diagonal = np.zeros(case.grid.shape)
    for _ in range(4):
        probe = rng.choice([-1.0, 1.0], size=observed.value.shape) * np.sqrt(
            control.precision
        )
        diagonal += control.call("adjoint", water_lin.jacobian.adjoint, probe) ** 2 / 4
        control.work.counts["diagonal_probes"] = (
            control.work.counts.get("diagonal_probes", 0) + 1
        )
    damping = regularization_weight(diagonal, args.damping_ratio, args.damping_absolute)
    regularization = physical_regularization(
        args.regularization_length_wavelengths, frequencies[-1], case.grid.spacing_m
    )
    regularization_info = regularization
    tv_transition = None
    if args.regularization_penalty == "smooth_tv":
        length_m = args.regularization_length_wavelengths * 1500 / frequencies[-1]
        regularization = SpatialGradient(case.grid, length_m)
        tv_transition = 2 * args.tv_transition_mps / 1500**3
        regularization_info = {
            "kind": "smooth_anisotropic_tv",
            "length_m": float(length_m),
            "transition_mps": args.tv_transition_mps,
            "transition_squared_slowness": tv_transition,
            "transition_conversion": "linearized_at_1500_mps",
            "boundary": "interior_edges_only",
            "data_loss": "quadratic",
        }
    del water_lin
    basis = (
        None
        if args.model_size is None
        else BilinearBasis(case.grid, (args.model_size, args.model_size))
    )
    direction_sigma = args.smooth_sigma * (
        1 if basis is None else basis.shape[0] / case.grid.shape[0]
    )
    (args.out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                **parameters,
                "damping": float(damping),
                "damping_scaling": (
                    "four_training_only_rademacher_diagonal_probes"
                    if args.damping_absolute is None
                    else "fixed_absolute_coefficient"
                ),
                "regularization": regularization_info,
                "smooth_sigma": args.smooth_sigma,
                "direction_sigma_coefficient_pixels": direction_sigma,
                "initialization_qc": initialization_qc,
                "regularization_reference": (
                    "initial_model"
                    if args.prior_reference == "initial"
                    else "water_1500_mps"
                ),
                "model_parameterization": None if basis is None else basis.metadata(),
                "aperture": aperture,
            },
            sort_keys=False,
        )
    )
    if basis is not None:
        # Keep the SAME fine-grid penalty and diagonal damping scale. Only the
        # admissible model subspace changes; no GT-derived support is supplied.
        forward = CoefficientForward(forward, basis)
        regularization = FineGridRegularizer(basis, regularization)
        initial = np.full(basis.shape, 1 / 1500**2)
        prior = initial.copy()
        control.basis = basis
    print("inverting damping", damping, flush=True)
    if args.solver_audit_checkpoint is not None:
        from _inverse_solver_audit import audit_iterate

        helper_source = (
            Path(__file__).with_name("_inverse_solver_audit.py").read_bytes()
        )
        (args.out / "_inverse_solver_audit.py").write_bytes(helper_source)
        checkpoint_source = args.solver_audit_checkpoint.read_bytes()
        import io

        with np.load(io.BytesIO(checkpoint_source)) as saved:
            state = saved["squared_slowness" if basis is None else "coefficients"]
        if state.shape != initial.shape:
            raise ValueError("checkpoint shape does not match configured model space")
        report = audit_iterate(
            forward,
            control,
            state=state,
            reference=prior,
            bounds=bounds,
            damping=damping,
            regularization=regularization,
            smooth_sigma=direction_sigma,
            inner_iterations=args.inner_iterations,
            compare_inner_solvers=args.compare_inner_solvers,
            audit_methods=args.audit_inner_methods,
            skip_gradient_fd=args.skip_gradient_fd,
            audit_caps=args.audit_inner_caps,
            gradient_steps=(
                (1.0, 0.5, 0.25)
                if args.audit_gradient_steps is None
                else args.audit_gradient_steps
            ),
            inner_options=dict(
                rtol=args.inner_rtol,
                atol=args.inner_atol,
                btol=args.inner_btol,
                conlim=args.inner_conlim,
            ),
        )
        report["checkpoint_sha256"] = hashlib.sha256(checkpoint_source).hexdigest()
        report["audit_helper_sha256"] = hashlib.sha256(helper_source).hexdigest()
        report["work"] = control.work.counts
        (args.out / "solver_audit.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    start = time.perf_counter()
    solver_parameters = dict(
        initial=initial,
        prior_reference=None if args.prior_reference == "initial" else prior,
        bounds=bounds,
        inner_iterations=args.inner_iterations,
        damping=damping,
        regularization=regularization,
    )
    if args.optimizer == "trf":
        from _finite_frequency_trf import solve_trust_region

        selected, metrics = solve_trust_region(
            forward,
            control,
            **solver_parameters,
            tv_transition=tv_transition,
            gradient_rtol=args.gradient_rtol,
        )
    else:
        selected, metrics = nonlinear_least_squares(
            forward,
            control,
            **solver_parameters,
            smooth_sigma=direction_sigma,
            max_update_mps=12,
            max_backtracks=8,
            gradient_rtol=args.gradient_rtol,
            inner_solver=args.inner_solver,
            inner_options=dict(
                rtol=args.inner_rtol,
                atol=args.inner_atol,
                btol=args.inner_btol,
                conlim=args.inner_conlim,
                preconditioner=args.inner_preconditioner,
            ),
        )
    speed = 1 / np.sqrt(selected if basis is None else basis.forward(selected))
    metrics["initialization_qc"] = initialization_qc
    metrics["model_parameterization"] = None if basis is None else basis.metadata()
    metrics.update(
        compute_regional_image_metrics(
            speed, case.ground_truth.sound_speed_mps, water_speed_mps=1500
        )
    )
    metrics["elapsed_s"] = time.perf_counter() - start
    result = ReconstructionResult(
        algorithm=config.name,
        case_id=case.case_id,
        sound_speed_mps=speed,
        metrics=metrics,
    )
    write_result_hdf5(result, args.out / "result.h5")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    current = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (repo / "src/usctbench").rglob("*.py")
    }
    (args.out / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "finite_frequency_traveltime",
                "source_changed_during_run": current != manifest["source_sha256"],
                "stop_reason": metrics["stop_reason"],
            }
        )
    )
    # Recompute at the validation-selected model, never the last trial cache.
    # Assessment is outside optimization and cannot affect its selected state.
    assessment_start = time.perf_counter()
    pressure.budget_check = None
    try:
        predicted_ratio = pressure.linearize(1 / speed**2).value / reference
        assessment = common_observable_evaluation(
            predicted_ratio, ratio, water_picker, relative_base, control.split
        )
        assessment.update(
            status="success",
            selected_iteration=metrics["stopping"]["selected_iteration"],
            result_sha256=hashlib.sha256(
                (args.out / "result.h5").read_bytes()
            ).hexdigest(),
            scope="posthoc_not_used_for_stopping_or_selection",
            extra_pressure_forward_calls=1,
            elapsed_s=time.perf_counter() - assessment_start,
        )
        np.savez_compressed(
            args.out / "selected_pressure_ratios.npz",
            predicted=predicted_ratio,
            observed=ratio,
            frequencies_hz=frequencies,
        )
    except (FloatingPointError, RuntimeError) as exc:
        assessment = {
            "status": "failed",
            "error": str(exc),
            "elapsed_s": time.perf_counter() - assessment_start,
        }
    (args.out / "common_evaluation.json").write_text(json.dumps(assessment, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    gt = case.ground_truth.sound_speed_mps
    for ax, image, name in zip(
        axes, (gt, speed), ("GT", "Finite-frequency traveltime")
    ):
        ax.imshow(image, cmap="gray", origin="lower", vmin=gt.min(), vmax=gt.max())
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(
        f"{args.bands}: RMSE {metrics['rmse']:.2f} / PSNR {metrics['psnr']:.2f} / SSIM {metrics['ssim']:.3f}"
    )
    fig.savefig(args.out / "preview.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                k: metrics.get(k)
                for k in (
                    "rmse",
                    "psnr",
                    "ssim",
                    "stop_reason",
                    "elapsed_s",
                    "evaluation",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
