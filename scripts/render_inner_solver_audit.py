"""Plot fixed-checkpoint numerical accuracy versus measured operator work."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    styles = {
        "CG": ("#0072B2", "o"),
        "LSMR": ("#D55E00", "s"),
        "LSQR": ("#009E73", "^"),
        "LSMR + column scaling": ("#CC79A7", "D"),
    }
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    rows = []
    for ax, case, title in zip(
        axes, ("high_d", "low_ob"), ("NBP-D", "OpenBreastUS HET")
    ):
        report = json.loads(
            (
                args.runs / f"augmented_{case}_all_caps_r2" / "solver_audit.json"
            ).read_text()
        )
        grouped = {name: [] for name in styles}
        for entry in report["normal_subproblems"]:
            inner = entry["inner"]
            method = {
                "real_parameter_normal_cg": "CG",
                "augmented_lsmr": "LSMR",
                "augmented_lsqr": "LSQR",
            }[inner["method"]]
            if inner.get("preconditioner") == "column_rms":
                method = "LSMR + column scaling"
            calls = entry["work"]["forward_calls"] + entry["work"]["adjoint_calls"]
            relative = inner["true_relative_residual"]
            grouped[method].append((calls, relative))
            rows.append(
                dict(
                    case=case,
                    checkpoint_sha256=report["checkpoint_sha256"],
                    method=method,
                    cap=inner["iteration_limit"],
                    iterations=inner["iterations"],
                    operator_calls=calls,
                    normal_relative_residual=relative,
                    quadratic_objective=entry["quadratic_objective"],
                    elapsed_s=entry["elapsed_s"],
                    stop_reason=inner["stop_reason"],
                    scipy_stop_code=inner.get("scipy_stop_code"),
                )
            )
        for name, points in grouped.items():
            color, marker = styles[name]
            x, y = zip(*sorted(set(points)))
            ax.loglog(
                x,
                y,
                color=color,
                marker=marker,
                label=name,
                linewidth=1.6,
                markersize=5,
            )
        ax.axhline(1e-3, color="0.5", linestyle="--", linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Operator calls (J and J*)")
        ax.grid(True, which="major", color="0.9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recomputed relative normal residual")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Fixed linearization: 256 x 256, 64 TX/RX, 61 frequencies", fontweight="bold"
    )
    fig.text(
        0.5,
        0.11,
        "Dashed: normal target 1e-3. CG uses a weaker stopping rule; no image-quality claim.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
