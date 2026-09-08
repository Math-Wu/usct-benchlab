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
from scipy.ndimage import gaussian_filter

from usctbench.metrics import compute_regional_image_metrics, non_water_tissue_mask


def render(root, out, case_ids, stages, metric_region="tissue"):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    policies = {}
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
            metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
        water_speed = float(metadata.get("reference_sound_speed_mps", 1500.0))
        # Evaluation only: this mask is never exported to an inverse algorithm.
        tissue = non_water_tissue_mask(truth, water_speed_mps=water_speed)
        plt.imsave(
            out / f"{case_id}_evaluation_mask.png", tissue, cmap="gray", vmin=0, vmax=1
        )
        gt_highpass = truth - gaussian_filter(truth, 2.0)
        panels = [("GT", truth, None)]
        for stage, algorithm, title in methods:
            path = root / stage / algorithm / case_id
            with h5py.File(path / "result.h5") as handle:
                pixels = handle["sound_speed_mps"][()]
            metrics = json.loads((path / "metrics.json").read_text())
            regional = compute_regional_image_metrics(
                pixels, truth, primary_region=metric_region, water_speed_mps=water_speed
            )
            policies[case_id] = regional["image_evaluation"]
            metrics = {**metrics, **regional}
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
            record.update(
                {k: v for k, v in regional.items() if k != "image_evaluation"}
            )
            record["evaluation_mask_sha256"] = regional["image_evaluation"][
                "mask_sha256"
            ]
            for key in (
                "version",
                "water_speed_mps",
                "data_range_mps",
                "tissue_pixels",
            ):
                record["evaluation_" + key] = regional["image_evaluation"][key]
            if tissue.any():
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

                def score(name):
                    value = metrics.get(name)
                    return f"{value:.3f}" if value is not None else "N/A"

                ax.set_xlabel(
                    f"RMSE {score('rmse')} m/s\nPSNR {score('psnr')} dB  SSIM {score('ssim')}",
                    fontweight="bold",
                    fontsize=9,
                )
        fig.colorbar(im, ax=axes[row_index], shrink=0.8, pad=0.01, label="m/s")
    fig.suptitle(
        f"Same k-Wave pressure | {metric_region} metrics | full field of view",
        fontsize=13,
    )
    fig.savefig(out / "reconstructions.png", dpi=160)
    plt.close(fig)
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "evaluation_policy.json").write_text(
        json.dumps(
            {
                "primary_region": metric_region,
                "cases": policies,
                "old_reconstruction_metrics_overwritten": False,
            },
            indent=2,
        )
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument(
        "--metric-region", choices=["tissue", "full_image"], default="tissue"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        required=True,
        help="stage:registered_algorithm:display_title",
    )
    args = parser.parse_args()
    render(args.root, args.out, args.case_ids, args.stages, args.metric_region)
