#!/usr/bin/env python3
"""Render final results, or explicitly labelled intermediate checkpoints."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usctbench.core.io import read_case_hdf5
from usctbench.metrics import compute_regional_image_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--allow-checkpoint", action="store_true")
    parser.add_argument(
        "--variant",
        nargs=2,
        action="append",
        metavar=("LABEL", "RUN_TEMPLATE"),
        help="explicit comparison column; run template contains {sample}",
    )
    parser.add_argument("--out", type=Path, help="PNG output, with a JSON sidecar")
    parser.add_argument(
        "--suffix", default="", help="run-directory suffix, e.g. _stabilized_r2"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("single", "multi", "low"),
        default=["single", "multi"],
    )
    args = parser.parse_args()
    titles = {
        "single": "One broad band",
        "multi": "Three frequency bands",
        "low": "Low band",
    }
    variants = args.variant or [
        (titles[m], "{sample}_" + m + args.suffix) for m in args.modes
    ]
    for _, template in variants:
        if "{sample}" not in template:
            parser.error("each run template must contain {sample}")
    output = args.out or args.runs / f"multiband_comparison{args.suffix}.png"
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
    fig, axes = plt.subplots(
        2,
        1 + len(variants),
        figsize=(3.7 * (1 + len(variants)), 8),
        layout="constrained",
    )
    records = []
    for row, (name, path, label) in enumerate(
        [
            (
                "high_d",
                "cases/high_band/D510022534/envelope_case.h5",
                "NBP D / 200-800 kHz",
            ),
            (
                "low_ob",
                "cases/low_band/breast_train_speed_class_1_000000/pressure_case.h5",
                "OpenBreastUS HET / 80-250 kHz",
            ),
        ]
    ):
        truth = read_case_hdf5(args.handoff / path).ground_truth.sound_speed_mps
        axes[row, 0].imshow(
            truth, cmap="gray", origin="lower", vmin=truth.min(), vmax=truth.max()
        )
        axes[row, 0].set_ylabel(label, fontweight="bold")
        for col, (variant_label, template) in enumerate(variants, 1):
            out = args.runs / template.format(sample=name)
            if (out / "result.h5").exists():
                with h5py.File(out / "result.h5") as f:
                    image = np.asarray(f["sound_speed_mps"])
                metrics = json.loads((out / "metrics.json").read_text())
                status = metrics["stop_reason"]
            elif args.allow_checkpoint:
                image = 1 / np.sqrt(np.load(out / "checkpoint.npz")["squared_slowness"])
                metrics = compute_regional_image_metrics(image, truth)
                iteration = json.loads((out / "progress.json").read_text())[-1][
                    "iteration"
                ]
                status = f"Intermediate iteration {iteration}, not final"
            else:
                raise ValueError(f"unfinished run: {out}")
            if image.shape != truth.shape or not np.isfinite(image).all():
                raise ValueError(f"invalid reconstructed image: {out}")
            axes[row, col].imshow(
                image, cmap="gray", origin="lower", vmin=truth.min(), vmax=truth.max()
            )
            axes[row, col].set_xlabel(
                f"RMSE {metrics['rmse']:.2f} / PSNR {metrics['psnr']:.2f}\nSSIM {metrics['ssim']:.3f}\n{status}",
                fontsize=10,
                fontweight="bold",
            )
            records.append(
                {
                    "sample": name,
                    "variant": variant_label,
                    "run": out.name,
                    "status": status,
                    **{k: metrics[k] for k in ("rmse", "psnr", "ssim")},
                }
            )
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[0], ["GT"] + [label for label, _ in variants]):
        ax.set_title(title, fontweight="bold")
    fig.suptitle(
        "Finite-frequency traveltime / 64 TX x 64 RX / 256 x 256\nTissue-region metrics; common raw pressure and QC"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    sidecar = (
        output.with_suffix(".json")
        if args.out
        else args.runs / f"image_comparison{args.suffix}.json"
    )
    sidecar.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
