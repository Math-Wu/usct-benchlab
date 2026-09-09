#!/usr/bin/env python3
"""Render measured adaptation audits and all frozen comparison outputs."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from usctbench.core.io import read_case_hdf5


def read(root, name):
    return json.loads((root / name / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--comparison-prefix", default="comparison_final")
    args = parser.parse_args()
    try:
        findfont("Times New Roman", fallback_to_default=False)
        font = "Times New Roman"
    except ValueError:
        font = "DejaVu Serif"
    plt.rcParams.update({"font.family": font, "font.size": 11})
    root = args.runs
    groups = ["high_d", "low_d", "low_ob"]
    names = ["NBP D / 500 kHz", "NBP D / 200 kHz", "OpenBreastUS / 200 kHz"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
    for k, picker in enumerate(("xcorr", "envelope_0.1", "aic")):
        values = [
            np.mean(
                [
                    read(root, g + "_r2")["results"][picker][str(c)]["rms_ns"]
                    for c in (1480, 1520)
                ]
            )
            for g in groups
        ]
        axes[0].plot(np.arange(3), values, "o-", label=picker)
    axes[0].set(
        yscale="log", ylabel="Delay RMSE (ns)", title="Independent uniform validation"
    )
    axes[0].set_xticks(np.arange(3), ["D high", "D low", "OB low"])
    axes[0].legend(fontsize=9)
    for g, name in zip(groups, names):
        report = read(root, "eikonal_" + g + "_4")
        levels = report["refinements"]
        axes[1].plot(
            [r["grid_shape"][0] for r in levels],
            [r["successive_delta_change_ns"]["rms"] for r in levels],
            "o-",
            label=name,
        )
    axes[1].set(
        yscale="log",
        xlabel="Refined grid width",
        ylabel="Successive delay change (ns)",
        title="Second-order Eikonal refinement",
    )
    axes[1].legend(fontsize=9)
    keys = [
        "straight_vs_observed_ns",
        "born_vs_observed_ns",
        "full_green_vs_observed_ns",
    ]
    for j, g in enumerate(("high_d", "low_ob")):
        report = read(root, "finite_verified_" + g)
        axes[2].bar(
            np.arange(3) + (j - 0.5) * 0.35,
            [report[k]["rms"] for k in keys],
            width=0.35,
            label="NBP D" if j == 0 else "OpenBreastUS",
        )
    axes[2].set(
        yscale="log",
        ylabel="Observable RMSE (ns)",
        title="Same finite-frequency functional",
    )
    axes[2].set_xticks(np.arange(3), ["Straight", "Linear Born", "Full Green"])
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(root / "adaptation_evidence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(18, 7), layout="constrained")
    labels = ["GT", "CGLS", "SIRT", "SART", "Bent (1st order)", "Bent (2nd order)"]
    for row, (g, casepath, title) in enumerate(
        (
            (
                "high_d",
                "cases/high_band/D510022534/envelope_case.h5",
                "NBP D / 500 kHz",
            ),
            (
                "low_ob",
                "cases/low_band/breast_train_speed_class_1_000000/pressure_case.h5",
                "OpenBreastUS HET / 200 kHz",
            ),
        )
    ):
        case = read_case_hdf5(args.handoff / casepath)
        truth = case.ground_truth.sound_speed_mps
        low, high = float(truth.min()), float(truth.max())
        rows = read(root, args.comparison_prefix + "_" + g)
        if len(rows) != 5:
            raise ValueError("comparison is incomplete")
        for col in range(6):
            image = truth
            if col:
                item = rows[col - 1]
                destination = (
                    root
                    / (args.comparison_prefix + "_" + g)
                    / item["label"]
                    / case.case_id
                )
                with h5py.File(destination / "result.h5") as f:
                    image = np.asarray(f["sound_speed_mps"])
                axes[row, col].set_xlabel(
                    f"RMSE {item['rmse']:.2f}\nPSNR {item['psnr']:.2f}   SSIM {item['ssim']:.3f}",
                    fontweight="bold",
                    fontsize=11,
                )
            im = axes[row, col].imshow(
                image, cmap="gray", origin="lower", vmin=low, vmax=high
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(
                    labels[col], fontweight="bold", fontsize=14, pad=12
                )
        axes[row, 0].set_ylabel(title, fontweight="bold", fontsize=12, labelpad=12)
        fig.colorbar(im, ax=axes[row, :], shrink=0.78, aspect=28, label="m/s")
    fig.suptitle(
        "Common k-Wave-derived xcorr / 64 TX x 64 RX / 256 x 256 images\n"
        "Tissue-region metrics; GT masks used only for evaluation",
        fontsize=16,
    )
    fig.savefig(root / "frozen_ray_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
