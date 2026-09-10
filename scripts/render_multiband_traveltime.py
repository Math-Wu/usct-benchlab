#!/usr/bin/env python3
"""Render final results, or explicitly labelled intermediate checkpoints."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usctbench.core.io import read_case_hdf5
from usctbench.metrics import compute_regional_image_metrics


def load_run(out, truth, allow_checkpoint):
    """Use a single image snapshot; never infer its iteration from a live log."""
    final = (out / "result.h5").exists() and (out / "metrics.json").exists()
    artifact = out / ("result.h5" if final else "checkpoint.npz")
    if not final and not allow_checkpoint:
        raise ValueError(f"unfinished run: {out}")
    snapshot = artifact.read_bytes()
    if final:
        with h5py.File(io.BytesIO(snapshot)) as handle:
            image = np.asarray(handle["sound_speed_mps"])
            recorded = json.loads(handle.attrs["metrics_json"])
        iteration = recorded.get("stopping", {}).get("selected_iteration")
        status = f"Final: {recorded['stop_reason']}"
    else:
        with np.load(io.BytesIO(snapshot)) as checkpoint:
            image = 1 / np.sqrt(checkpoint["squared_slowness"])
            iteration = (
                int(checkpoint["iteration"]) if "iteration" in checkpoint else None
            )
        recorded = {}
        status = (
            f"Intermediate iteration {iteration}, not final"
            if iteration is not None
            else "Intermediate checkpoint, iteration unverified"
        )
    if image.shape != truth.shape or not np.isfinite(image).all():
        raise ValueError(f"invalid reconstructed image: {out}")
    metrics = compute_regional_image_metrics(image, truth, water_speed_mps=1500)
    for key in ("rmse", "psnr", "ssim"):
        if final and not np.isclose(recorded[key], metrics[key], rtol=1e-8, atol=1e-10):
            raise ValueError(f"recorded {key} disagrees with image/GT: {out}")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return image, {
        "status": status,
        "final": final,
        "selected_iteration" if final else "checkpoint_iteration": iteration,
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(snapshot).hexdigest(),
        "n_tx": len(manifest.get("tx_parent_indices", [])) or None,
        "n_rx": len(manifest.get("rx_parent_indices", [])) or None,
        "image_shape_yx": list(image.shape),
        "acquisition_input_sha256": manifest.get("acquisition_input_sha256"),
        "observable": manifest.get("observable"),
        "elapsed_s": recorded.get("elapsed_s"),
        "budget_elapsed_s": recorded.get("stopping", {}).get("elapsed_s"),
        "evaluation": recorded.get("evaluation"),
        **metrics,
    }


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
    plt.rcParams.update(
        {"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 11}
    )
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
        displayed = axes[row, 0].imshow(
            truth, cmap="gray", origin="lower", vmin=truth.min(), vmax=truth.max()
        )
        axes[row, 0].set_ylabel(label, fontweight="bold")
        for col, (variant_label, template) in enumerate(variants, 1):
            out = args.runs / template.format(sample=name)
            image, record = load_run(out, truth, args.allow_checkpoint)
            axes[row, col].imshow(
                image, cmap="gray", origin="lower", vmin=truth.min(), vmax=truth.max()
            )
            counts = (
                f"{record['n_tx']} TX x {record['n_rx']} RX"
                if record["n_tx"] and record["n_rx"]
                else "Channel count not recorded"
            )
            axes[row, col].set_xlabel(
                f"RMSE {record['rmse']:.2f} / PSNR {record['psnr']:.2f}\n"
                f"SSIM {record['ssim']:.3f}\n{counts}\n{record['status']}",
                fontsize=10,
                fontweight="bold",
            )
            records.append(
                {
                    "sample": name,
                    "variant": variant_label,
                    "run": out.name,
                    **record,
                }
            )
        fig.colorbar(
            displayed, ax=axes[row].tolist(), fraction=0.02, pad=0.02, label="m/s"
        )
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[0], ["GT"] + [label for label, _ in variants]):
        ax.set_title(title, fontweight="bold")
    fig.suptitle(
        "Finite-frequency traveltime comparison\nTissue-region metrics; GT used only for evaluation"
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
