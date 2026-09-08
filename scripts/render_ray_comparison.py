"""Render the frozen 64-channel campaign, including all negative results.

Expected handoff output directories are documented in the 64-channel report.
This script is evaluation/display only; it never modifies reconstruction arrays.
"""

from pathlib import Path
import argparse
import json
import csv
import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from PIL import Image, ImageDraw, ImageFont
from usctbench.core.io import read_case_hdf5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--handoff", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = args.handoff.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
figures = out / "figures"
figures.mkdir(exist_ok=True)
algorithms = ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"]
names = ["CGLS", "SIRT", "SART", "Bent"]
groups = {
    "high_d": (
        "High-band NBP D510022534",
        "cases/high_band/D510022534/envelope_case.h5",
        {
            "pixels": "final_high_pixels",
            "basis32": "final_high_basis32",
            "source_phase": "phase_high",
        },
    ),
    "low_d": (
        "Low-band NBP D510022534",
        "cases/low_band/D510022534/pressure_case.h5",
        {"pixels": "low_d_pixels", "basis32": "low_d_basis32"},
    ),
    "low_ob": (
        "Low-band OpenBreastUS class 1",
        "cases/low_band/breast_train_speed_class_1_000000/pressure_case.h5",
        {"pixels": "low_ob_pixels", "basis32": "low_ob_basis32"},
    ),
}

font_path = findfont("DejaVu Sans")
font = ImageFont.truetype(font_path, 16)
small = ImageFont.truetype(font_path, 13)
big = ImageFont.truetype(font_path, 22)
rows = []
paired = []
manifest = {}
for key, (title, case_path, variants) in groups.items():
    case = read_case_hdf5(root / case_path)
    truth = case.ground_truth.sound_speed_mps.copy()
    case.measurement.time_data = case.measurement.water_reference_time = None
    lo, hi = float(truth.min()), float(truth.max())
    allmetrics = {}
    allimages = {}
    split = None
    for variant, directory in variants.items():
        assert (root / directory / "COMPLETE.json").exists(), directory
        feature = "source_phase" if variant == "source_phase" else "existing"
        for alg in algorithms:
            location = root / directory / feature / alg / case.case_id
            m = json.loads((location / "metrics.json").read_text())
            with h5py.File(location / "result.h5") as h:
                image = h["sound_speed_mps"][()]
            assert image.shape == truth.shape and np.isfinite(image).all()
            assert m["primary_image_region"] == "tissue"
            if split is None:
                split = m["evaluation_split"]
            assert split == m["evaluation_split"], "split changed"
            assert not m["stopping"]["ground_truth_used_for_stopping"]
            assert not m["implementation"]["source_changed_during_run"]
            allmetrics[variant, alg] = m
            allimages[variant, alg] = image
            stop = m["stopping"]
            work = stop["work"]
            row = {
                "group": key,
                "case_id": case.case_id,
                "variant": variant,
                "algorithm": alg,
                "rmse_mps": m["rmse"],
                "psnr_db": m["psnr"],
                "ssim": m["ssim"],
                "water_rmse_mps": m["water_background_rmse"],
                "water_bias_mps": m["water_background_bias"],
                "full_image_rmse_mps": m["full_image_rmse"],
                "train_relative_residual": m["data_relative_residual"],
                "validation_relative_residual": m["evaluation"]["receiver"][
                    "weighted_relative_residual"
                ],
                "stop_reason": stop["reason"],
                "selected_iteration": stop["selected_iteration"],
                "completed_iterations": stop["completed_iterations"],
                "loop_elapsed_s": stop["elapsed_s"],
                "logical_forward_calls": work.get("forward_calls", 0),
                "logical_adjoint_calls": work.get("adjoint_calls", 0),
                "forward_evaluations": work.get("forward", 0),
                "jacobian_products": work.get("jacobian", 0),
                "line_search_evaluations": work.get("line_search", 0),
                "inner_iterations": work.get("inner_iterations", 0),
                "subset_updates": work.get("subset_updates", 0),
                "gt_range_min": lo,
                "gt_range_max": hi,
                "result": str(location.relative_to(root)),
            }
            rows.append(row)
    for alg in algorithms:
        a, b = allmetrics["pixels", alg], allmetrics["basis32", alg]
        paired.append(
            {
                "group": key,
                "algorithm": alg,
                "delta_rmse_mps": b["rmse"] - a["rmse"],
                "delta_ssim": b["ssim"] - a["ssim"],
                "delta_water_rmse_mps": b["water_background_rmse"]
                - a["water_background_rmse"],
                "delta_validation_relative_residual": b["evaluation"]["receiver"][
                    "weighted_relative_residual"
                ]
                - a["evaluation"]["receiver"]["weighted_relative_residual"],
            }
        )
        for horizontal, xlabel in [
            ("iteration", "Completed method-specific iteration"),
            ("elapsed_s", "Measured solver elapsed time (s)"),
        ]:
            fig, ax = plt.subplots(figsize=(7.7, 4.8))
            for variant, linestyle in [("pixels", "-"), ("basis32", "--")]:
                m = allmetrics[variant, alg]
                history = m["iteration_history"]
                x = [i[horizontal] for i in history]
                ax.plot(
                    x,
                    [i["relative_residual"] for i in history],
                    linestyle=linestyle,
                    label=f"{variant}: train",
                )
                ax.plot(
                    x,
                    [i["validation_relative_residual"] for i in history],
                    linestyle=linestyle,
                    label=f"{variant}: validation",
                )
                selected = m["stopping"]["selected_iteration"]
                mark = next(i for i in history if i["iteration"] == selected)
                ax.scatter(
                    [mark[horizontal]],
                    [mark["validation_relative_residual"]],
                    marker="o",
                    s=38,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Weighted relative ToF residual")
            ax.set_title(title + " / " + names[algorithms.index(alg)])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
            foot = " | ".join(
                v
                + ": "
                + allmetrics[v, alg]["stop_reason"]
                + "; selected "
                + str(allmetrics[v, alg]["stopping"]["selected_iteration"])
                for v in ("pixels", "basis32")
            )
            fig.text(0.03, 0.015, foot, fontsize=8)
            fig.tight_layout(rect=(0, 0.045, 1, 1))
            fig.savefig(figures / f"{key}_{alg}_{horizontal}.png", dpi=135)
            plt.close(fig)
    # Arrays displayed directly on one common linear gray scale; no denoising or sharpening.
    canvas = Image.new("RGB", (1424, 145 + len(variants) * 338), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, font=big, fill="black")
    draw.text(
        (24, 52),
        f"All panels: {lo:.2f} to {hi:.2f} m/s, same linear grayscale; GT used only for evaluation.",
        font=font,
        fill="black",
    )
    draw.text(
        (24, 77),
        "Values outside this display range are clipped visually only; scores use unaltered float64 arrays.",
        font=small,
        fill="black",
    )
    for j, label in enumerate(["Ground truth"] + names):
        draw.text((24 + j * 280, 110), label, font=font, fill="black")
    for i, variant in enumerate(variants):
        y = 145 + i * 338
        for j, alg in enumerate([None] + algorithms):
            image = truth if alg is None else allimages[variant, alg]
            gray = np.rint(np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(
                "uint8"
            )
            canvas.paste(Image.fromarray(gray), (24 + j * 280, y))
            if alg is None:
                text = variant + " / 256 x 256"
            else:
                m = allmetrics[variant, alg]
                text = f'RMSE {m["rmse"]:.3f} / SSIM {m["ssim"]:.4f}'
            draw.text((24 + j * 280, y + 262), text, font=small, fill="black")
            if alg is not None:
                draw.text(
                    (24 + j * 280, y + 283),
                    f'selected/completed {m["stopping"]["selected_iteration"]}/{m["stopping"]["completed_iterations"]}',
                    font=small,
                    fill="black",
                )
    canvas.save(figures / f"{key}_comparison.png")
    manifest[key] = {
        "case": case_path,
        "variants": variants,
        "same_split": True,
        "evaluation_split": split,
        "gt_display_range": [lo, hi],
    }
with (out / "metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(out / "metrics.json").write_text(json.dumps(rows, indent=2))
(out / "paired_differences.json").write_text(json.dumps(paired, indent=2))
(out / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2))
print("rendered", len(rows), "runs and", len(list(figures.glob("*.png"))), "figures")
