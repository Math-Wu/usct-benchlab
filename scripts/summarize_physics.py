#!/usr/bin/env python3
"""Render small evidence artifacts from completed validate_physics runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from usctbench.core.io import read_case_hdf5, read_result_hdf5

ALGORITHMS = {
    "straight_cgls": "CGLS",
    "straight_sirt": "SIRT",
    "straight_sart": "SART",
    "bent_ray_gn": "Eikonal",
    "rwave_adapter": "Ray-Born",
}


def summarize(root, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    rows = []
    cases = sorted(root.glob("*/pressure_case.h5"))
    if len(cases) != 8:
        raise ValueError("the four-class paired report requires eight completed cases")
    loaded = {path.parent.name: read_case_hdf5(path) for path in cases}
    for stage in ("reconstructions", "equal_ray_regularization"):
        for path in sorted((root / stage).glob("*/*/metrics.json")):
            metrics = json.loads(path.read_text())
            stop = metrics["stopping"]
            row = {
                "stage": stage,
                "case": path.parent.name,
                "algorithm": path.parent.parent.name,
            }
            row.update(
                {
                    k: metrics.get(k)
                    for k in (
                        "rmse",
                        "ssim",
                        "psnr",
                        "data_relative_residual",
                        "stop_reason",
                    )
                }
            )
            row.update(
                {
                    k: stop.get(k)
                    for k in (
                        "completed_iterations",
                        "selected_iteration",
                        "elapsed_s",
                        "termination_category",
                    )
                }
            )
            for group in ("receiver", "frequency", "joint"):
                row[group + "_relative"] = (
                    metrics.get("evaluation", {})
                    .get(group, {})
                    .get("weighted_relative_residual")
                )
            for group in ("train", "receiver", "frequency", "joint"):
                row["contrast_" + group + "_relative"] = (
                    metrics.get("contrast_evaluation", {})
                    .get(group, {})
                    .get("weighted_relative_residual")
                )
            counts = stop.get("work", {})
            row["forward_calls"] = counts.get("forward_calls")
            row["adjoint_calls"] = counts.get("adjoint_calls")
            rows.append(row)
    if len(rows) != 48:
        raise ValueError(
            f"expected 40 initial plus 8 matched-penalty runs, found {len(rows)}"
        )
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for dataset, ids in (
        ("NBPslice2D", [key for key in loaded if not key.startswith("breast_")]),
        ("OpenBreastUS", [key for key in loaded if key.startswith("breast_")]),
    ):
        fig, axes = plt.subplots(4, 6, figsize=(13.5, 9.5), layout="constrained")
        for r, case_id in enumerate(ids):
            case = loaded[case_id]
            gt = case.ground_truth.sound_speed_mps
            label = case_id[0] if dataset == "NBPslice2D" else f"Class {r+1}"
            vmin, vmax = float(np.min(gt)), float(np.max(gt))
            panels = [("GT", gt, None)]
            for algorithm, title in ALGORITHMS.items():
                stage = (
                    "equal_ray_regularization"
                    if algorithm == "bent_ray_gn"
                    else "reconstructions"
                )
                path = root / stage / algorithm / case_id
                result = read_result_hdf5(path / "result.h5")
                metrics = json.loads((path / "metrics.json").read_text())
                panels.append((title, result.sound_speed_mps, metrics))
            for c, (title, pixels, metrics) in enumerate(panels):
                ax = axes[r, c]
                im = ax.imshow(
                    pixels,
                    origin="lower",
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontweight="bold", fontsize=12)
                if c == 0:
                    ax.set_ylabel(label, fontweight="bold", fontsize=11)
                    ax.set_xlabel(f"{vmin:.0f}..{vmax:.0f} m/s")
                if metrics:
                    reason = metrics.get("stop_reason", "unknown")
                    reason = {
                        "validation_plateau": "validation plateau",
                        "max_iterations": "iteration cap",
                        "target_residual": "residual target",
                        "line_search_failed": "LINE SEARCH FAILED",
                        "time_budget": "time budget",
                    }.get(reason, reason)
                    ax.set_xlabel(
                        f"PSNR {metrics['psnr']:.2f}  SSIM {metrics['ssim']:.3f}\n{reason}",
                        fontsize=8,
                        color="#a32020" if "FAILED" in reason else "black",
                    )
            fig.colorbar(im, ax=axes[r, :], shrink=0.65, pad=0.01, label="m/s")
        fig.suptitle(
            f"{dataset} | independent k-Wave validation\n16 transmitters / 48 x 48 reconstruction; Eikonal matched penalty",
            fontsize=13,
        )
        fig.savefig(out / f"{dataset.lower()}_physics.png", dpi=150)
        plt.close(fig)
    # The plotted residual is training-domain dependent; do not rank different
    # physical models by comparing its absolute value across algorithms.
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for ax, case_id in zip(axes.flat, loaded):
        for algorithm in ("bent_ray_gn", "rwave_adapter"):
            stage = (
                "equal_ray_regularization"
                if algorithm == "bent_ray_gn"
                else "reconstructions"
            )
            metrics = json.loads(
                (root / stage / algorithm / case_id / "metrics.json").read_text()
            )
            history = metrics["iteration_history"]
            if not history:
                continue
            first = history[0]["residual_norm"]
            ax.plot(
                [h["iteration"] for h in history],
                [h["residual_norm"] / first for h in history],
                marker=".",
                label=ALGORITHMS[algorithm],
            )
        ax.set_title(case_id.replace("breast_train_speed_", ""), fontsize=9)
        ax.set_xlabel("Completed outer steps")
        ax.set_ylabel("Training residual / initial")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "residual_curves.png", dpi=150)
    plt.close(fig)
    print(out / "metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.root, args.out)
