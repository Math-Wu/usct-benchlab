"""Budgeted SIRT/SART with training-only normalization and atomic checkpoints."""

from __future__ import annotations

import numpy as np

from usctbench.core.stopping import BudgetExhausted


def row_action(
    operator,
    control,
    *,
    initial,
    reference,
    project,
    to_image,
    relaxation,
    subsets=1,
    postprocess=None,
):
    """One iteration is a full sweep; each subset's actual operator work is counted.

    SART subsets use disjoint training rays. Normalization excludes held-out
    channels. An interrupted partial sweep returns the last *complete* sweep.
    Physical projection precedes residual evaluation and checkpoint selection.
    """
    x = np.array(initial, dtype=float, copy=True)
    start = x.copy()
    if not np.isfinite(relaxation) or not 0 < relaxation < 2:
        raise ValueError("relaxation must be finite and in (0, 2)")
    if isinstance(subsets, bool) or int(subsets) != subsets or subsets <= 0:
        raise ValueError("subsets must be a positive integer")
    try:
        prediction = control.call("forward", operator.forward, x).reshape(
            control.observed.shape
        )
        if control.observe(0, x, prediction, sound_speed=to_image(x)):
            return control.output(start)
        ids = np.flatnonzero(control.split.train)
        groups = np.array_split(ids, min(int(subsets), len(ids)))
        row_sum = np.maximum(
            operator.row_norms(power=1).reshape(control.observed.shape),
            np.finfo(float).tiny,
        )
        normalization = []
        for group in groups:
            precision = np.zeros_like(control.precision)
            precision.flat[group] = control.precision.flat[group]
            col = control.call("adjoint", operator.adjoint, precision)
            normalization.append((precision, np.where(col > 0, col, 1.0)))
        for iteration in range(1, control.policy.max_iterations + 1):
            previous = x.copy()
            for index, (precision, col) in enumerate(normalization):
                if index:
                    prediction = control.call("forward", operator.forward, x).reshape(
                        control.observed.shape
                    )
                residual = np.where(
                    precision > 0, control.safe_observed - prediction, 0.0
                )
                update = (
                    control.call(
                        "adjoint", operator.adjoint, precision * residual / row_sum
                    )
                    / col
                )
                x = project(x + relaxation * update)
                if not np.all(np.isfinite(x)):
                    raise FloatingPointError("nonfinite projected iterate")
                control.work.counts["subset_updates"] = (
                    control.work.counts.get("subset_updates", 0) + 1
                )
            if postprocess is not None:
                x = project(postprocess(x, iteration))
            prediction = control.call("forward", operator.forward, x).reshape(
                control.observed.shape
            )
            relative_update = float(
                np.linalg.norm(x - previous)
                / max(np.linalg.norm(reference + previous), np.finfo(float).tiny)
            )
            if control.observe(
                iteration,
                x,
                prediction,
                update_relative=relative_update,
                sound_speed=to_image(x),
            ):
                break
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    except FloatingPointError:
        control.monitor.finish("numerical_failure")
    return control.output(start)
