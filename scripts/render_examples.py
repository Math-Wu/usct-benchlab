#!/usr/bin/env python3
"""Render README-friendly GT/algorithm comparison panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.core.io import read_case_hdf5, read_result_hdf5

DEFAULT_ALGORITHMS = [
    "straight_cgls",
    "straight_sirt",
    "straight_sart",
    "bent_ray_gn",
    "rwave_adapter",
    "fwi_wust",
]
DEFAULT_LABELS = ["CGLS", "SIRT", "SART", "Bent-ray", "rWave", "FWI"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--row-labels", nargs="+", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--algorithm-labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument(
        "--display-transform",
        action="append",
        default=[],
        help="Display-only image transform as algorithm=transpose|rot90cw|rot90ccw|flipud|fliplr.",
    )
    parser.add_argument("--cmap", default="gray")
    parser.add_argument("--unit", default="m/s")
    args = parser.parse_args()

    if len(args.row_labels) != len(args.case_ids):
        raise SystemExit("--row-labels must have the same length as --case-ids")
    if len(args.algorithm_labels) != len(args.algorithms):
        raise SystemExit("--algorithm-labels must have the same length as --algorithms")

    transforms = _parse_transforms(args.display_transform)
    cases = _load_cases(Path(args.cases_dir), args.case_ids)
    rows = []
    summary_rows: list[dict[str, Any]] = []
    for row_label, case in zip(args.row_labels, cases, strict=True):
        gt = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
        row: list[dict[str, Any]] = [
            {"label": "GT", "image": gt, "metrics": {}, "row_label": row_label}
        ]
        for algorithm, column_label in zip(
            args.algorithms, args.algorithm_labels, strict=True
        ):
            result_path = Path(args.run_dir) / algorithm / case.case_id / "result.h5"
            if result_path.exists():
                result = read_result_hdf5(result_path)
                image = result.sound_speed_mps
                if image is None:
                    raise SystemExit(
                        f"{result_path} does not contain sound_speed_mps; "
                        "render_examples.py only renders sound-speed comparison panels"
                    )
                metrics = dict(result.metrics)
                row.append(
                    {
                        "label": column_label,
                        "image": _apply_transform(
                            np.asarray(image, dtype=float),
                            transforms.get(algorithm, ""),
                        ),
                        "metrics": metrics,
                    }
                )
                summary_rows.append(
                    {
                        "case_id": case.case_id,
                        "row_label": row_label,
                        "algorithm": algorithm,
                        **_metric_subset(metrics),
                    }
                )
            else:
                row.append(
                    {"label": column_label, "image": None, "metrics": {"missing": True}}
                )
                summary_rows.append(
                    {
                        "case_id": case.case_id,
                        "row_label": row_label,
                        "algorithm": algorithm,
                        "missing": True,
                    }
                )
        rows.append(row)

    _write_panel(rows, Path(args.out), title=args.title, cmap=args.cmap, unit=args.unit)
    _write_summary_csv(summary_rows, Path(args.out).with_suffix(".summary.csv"))
    Path(args.out).with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "cases_dir": str(Path(args.cases_dir)),
                "run_dir": str(Path(args.run_dir)),
                "case_ids": args.case_ids,
                "row_labels": args.row_labels,
                "algorithms": args.algorithms,
                "algorithm_labels": args.algorithm_labels,
                "display_transform": args.display_transform,
                "out": str(Path(args.out)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(args.out)
    return 0


def _parse_transforms(values: list[str]) -> dict[str, str]:
    transforms: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"display transform must be algorithm=transform, got {value!r}"
            )
        algorithm, transform = value.split("=", 1)
        transforms[algorithm.strip()] = transform.strip().lower()
    return transforms


def _apply_transform(image: np.ndarray, transform: str) -> np.ndarray:
    if not transform or transform == "none":
        return image
    if transform == "transpose":
        return np.swapaxes(image, 0, 1)
    if transform == "rot90cw":
        return np.rot90(image, k=-1)
    if transform == "rot90ccw":
        return np.rot90(image, k=1)
    if transform == "flipud":
        return np.flipud(image)
    if transform == "fliplr":
        return np.fliplr(image)
    raise SystemExit(f"unknown display transform: {transform}")


def _load_cases(cases_dir: Path, case_ids: list[str]) -> list[Any]:
    by_id = {}
    for path in sorted(cases_dir.glob("*.h5")):
        case = read_case_hdf5(path)
        by_id[case.case_id] = case
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"missing cases: {', '.join(missing)}")
    return [by_id[case_id] for case_id in case_ids]


def _write_panel(
    rows: list[list[dict[str, Any]]], out: Path, *, title: str, cmap: str, unit: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["mathtext.fontset"] = "dejavuserif"

    out.parent.mkdir(parents=True, exist_ok=True)
    nrows = len(rows)
    ncols = max(len(row) for row in rows)
    images = [
        cell["image"] for row in rows for cell in row if cell["image"] is not None
    ]
    vmin, vmax = _shared_limits(images)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(13.0, 2.35 * ncols + 1.0), max(7.8, 2.40 * nrows + 1.0)),
        dpi=220,
        squeeze=False,
    )
    image_artist = None
    for row_idx, row in enumerate(rows):
        for col_idx in range(ncols):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx >= len(row):
                ax.set_axis_off()
                continue
            cell = row[col_idx]
            image = cell["image"]
            if image is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
            else:
                image_artist = ax.imshow(
                    image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
                )
            if row_idx == 0:
                ax.set_title(
                    str(cell["label"]),
                    fontsize=11,
                    pad=11,
                    fontweight="bold",
                    color="black",
                )
            if col_idx == 0:
                ax.text(
                    -0.13,
                    0.5,
                    str(cell.get("row_label", "")),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    linespacing=1.05,
                )
            metric_text = _metric_text(cell["metrics"])
            if metric_text:
                ax.text(
                    0.5,
                    -0.075,
                    metric_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7.6,
                    fontweight="bold",
                    color="black",
                    linespacing=1.0,
                )

    if title:
        fig.suptitle(title, fontsize=15, y=0.988, fontweight="bold", color="black")
    fig.subplots_adjust(
        left=0.105,
        right=0.935 if image_artist is not None else 0.985,
        bottom=0.050,
        top=0.895 if title else 0.94,
        wspace=0.18,
        hspace=0.42,
    )
    if image_artist is not None:
        colorbar = fig.colorbar(
            image_artist, ax=axes.ravel().tolist(), fraction=0.016, pad=0.012
        )
        colorbar.set_label(unit, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _metric_text(metrics: dict[str, Any]) -> str:
    if metrics.get("missing"):
        return "missing"
    psnr_key = "psnr"
    ssim_key = "ssim"
    if _is_number(metrics.get(psnr_key)):
        text = f"PSNR {float(metrics[psnr_key]):.1f}"
        if _is_number(metrics.get(ssim_key)):
            text += f"  SSIM {float(metrics[ssim_key]):.3f}"
        return text
    return ""


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rmse",
        "ssim",
        "psnr",
        "data_relative_residual",
        "data_residual_reduction",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _shared_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite_parts = [
        np.asarray(image, dtype=float)[np.isfinite(image)].reshape(-1)
        for image in images
    ]
    finite = (
        np.concatenate([part for part in finite_parts if part.size])
        if finite_parts
        else np.asarray([])
    )
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "row_label",
        "case_id",
        "algorithm",
        "rmse",
        "ssim",
        "kwave_gt_rmse",
        "kwave_gt_ssim",
    ]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
