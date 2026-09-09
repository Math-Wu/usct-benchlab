#!/usr/bin/env python3
"""Compare actual derivative arrays across spectral quadrature resolutions."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--sample", default="high_d")
    parser.add_argument("--counts", nargs="+", type=int, default=[15, 31, 61, 121])
    args = parser.parse_args()
    counts = sorted(set(args.counts))
    arrays, summaries = [], []
    for n in counts:
        run = args.runs / f"kernel_{args.sample}_f{n}"
        with np.load(run / "kernel_audit.npz") as data:
            arrays.append({k: np.array(data[k]) for k in ("kernel", "gradient")})
        summaries.append(json.loads((run / "summary.json").read_text()))
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
    fig, axes = plt.subplots(
        2,
        len(counts),
        figsize=(4 * len(counts), 8),
        layout="constrained",
        squeeze=False,
    )
    records = []
    for col, (n, data, summary) in enumerate(zip(counts, arrays, summaries)):
        record = {"frequencies": n, **summary}
        for row, key in enumerate(("kernel", "gradient")):
            reference = arrays[-1][key]
            error = float(
                np.linalg.norm(data[key] - reference) / np.linalg.norm(reference)
            )
            record[f"{key}_relative_difference_vs_f{counts[-1]}"] = error
            limit = max(float(np.max(np.abs(a[key]))) for a in arrays)
            artist = axes[row, col].imshow(
                data[key], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xlabel(
                f"Relative difference vs {counts[-1]}: {error:.3g}"
            )
            if col == 0:
                axes[row, col].set_ylabel(
                    "Single-channel kernel" if row == 0 else "Initial training gradient"
                )
            if col == len(counts) - 1:
                fig.colorbar(artist, ax=list(axes[row]), shrink=0.7)
        axes[0, col].set_title(
            f"{n} frequencies / df {summary['frequency_step_hz']/1e3:.2f} kHz"
        )
        records.append(record)
    fig.suptitle(
        f"Spectral sampling audit: {args.sample}\nSame 256 x 256 propagation grid; no GT in derivatives"
    )
    fig.savefig(args.runs / f"frequency_sampling_{args.sample}.png", dpi=160)
    plt.close(fig)
    (args.runs / f"frequency_sampling_{args.sample}.json").write_text(
        json.dumps(records, indent=2)
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
