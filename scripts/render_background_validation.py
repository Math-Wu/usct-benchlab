#!/usr/bin/env python3
"""Render measured background-fix evidence without loading raw pressure tensors."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

from usctbench.metrics import compute_image_metrics


def render(root, out, case_ids, stages):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = [item.split(":", 2) for item in stages]
    fig, axes = plt.subplots(
        len(case_ids),
        len(methods) + 1,
        squeeze=False,
        figsize=(2.55 * (len(methods) + 1), 3.15 * len(case_ids)),
        layout="constrained",
    )
    for row_index, case_id in enumerate(case_ids):
        with h5py.File(root / case_id / "pressure_case.h5") as handle:
            truth = handle["ground_truth/sound_speed_mps"][()]
        water = np.isclose(truth, 1500, atol=0.1, rtol=0)
        # Evaluation only: this mask is never exported to an inverse algorithm.
        tissue = binary_fill_holes(~water)
        gt_highpass = truth - gaussian_filter(truth, 2.0)
        panels = [("GT", truth, None)]
        for stage, algorithm, title in methods:
            path = root / stage / algorithm / case_id
            with h5py.File(path / "result.h5") as handle:
                pixels = handle["sound_speed_mps"][()]
            metrics = json.loads((path / "metrics.json").read_text())
            stop = metrics["stopping"]
            record = dict(case_id=case_id, stage=stage, algorithm=algorithm)
            record.update(
                {
                    key: metrics.get(key)
                    for key in (
                        "rmse",
                        "ssim",
                        "psnr",
                        "data_relative_residual",
                        "stop_reason",
                    )
                }
            )
            record.update(
                {
                    key: stop.get(key)
                    for key in (
                        "selected_iteration",
                        "completed_iterations",
                        "elapsed_s",
                    )
                }
            )
            record["water_rmse_mps_posthoc_GT_mask"] = (
                float(np.sqrt(np.mean((pixels[water] - truth[water]) ** 2)))
                if water.any()
                else None
            )
            if tissue.any():
                record.update(
                    compute_image_metrics(
                        pixels, truth, mask=tissue, prefix="tissue_posthoc_"
                    )
                )
                hp = pixels - gaussian_filter(pixels, 2.0)
                record["tissue_highpass_corr_posthoc_sigma2px"] = (
                    float(np.corrcoef(hp[tissue], gt_highpass[tissue])[0, 1])
                    if np.std(hp[tissue]) > 0 and np.std(gt_highpass[tissue]) > 0
                    else None
                )
            for group in ("receiver", "frequency", "joint"):
                record[group + "_relative_residual"] = (
                    metrics.get("evaluation", {})
                    .get(group, {})
                    .get("weighted_relative_residual")
                )
            rows.append(record)
            panels.append((title, pixels, metrics))
        for col, (title, pixels, metrics) in enumerate(panels):
            ax = axes[row_index, col]
            im = ax.imshow(
                pixels,
                origin="lower",
                interpolation="nearest",
                cmap="gray",
                vmin=truth.min(),
                vmax=truth.max(),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title, fontweight="bold", fontsize=11)
            if metrics is None:
                ax.set_ylabel(case_id.replace("breast_train_speed_", ""), fontsize=9)
                ax.set_xlabel(f"{truth.shape[0]} x {truth.shape[1]}")
            else:
                reason = metrics["stop_reason"].replace("_", " ")
                ax.set_xlabel(
                    f"PSNR {metrics['psnr']:.2f}  SSIM {metrics['ssim']:.3f}\n{reason}",
                    fontweight="bold",
                    fontsize=9,
                )
        fig.colorbar(im, ax=axes[row_index], shrink=0.8, pad=0.01, label="m/s")
    fig.suptitle("Same k-Wave pressure | 128 transmitters / 128 receivers", fontsize=13)
    fig.savefig(out / "reconstructions.png", dpi=160)
    plt.close(fig)
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        required=True,
        help="stage:registered_algorithm:display_title",
    )
    args = parser.parse_args()
    render(args.root, args.out, args.case_ids, args.stages)
