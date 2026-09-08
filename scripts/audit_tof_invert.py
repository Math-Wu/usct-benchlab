"""Frozen ToF invert diagnosis. Outputs are development controls, not clinical evidence."""

import argparse
import json
import yaml
from pathlib import Path
from usctbench.benchmark.runner import run_algorithm_case
from usctbench.cli import register_builtin_algorithms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="shared audit directory, outside the source repository",
    )
    parser.add_argument(
        "--handoff", type=Path, required=True, help="extracted usct_handoff_64"
    )
    parser.add_argument("--group", choices=["high_d", "low_d", "low_ob"], required=True)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"],
        default=["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"],
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["kwave", "matched_straight", "matched_eikonal"],
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve() == repo or repo in args.out.resolve().parents:
        raise ValueError("private outputs must stay outside the repository")

    register_builtin_algorithms()
    R = args.out.resolve()
    H = args.handoff.resolve()
    group = args.group
    algs = args.algorithms
    features = args.features or (
        ["matched_straight", "matched_eikonal"]
        if group == "high_d"
        else ["kwave", "matched_straight", "matched_eikonal"]
    )
    rows = []
    for feature in features:
        for alg in algs:
            path = R / "controls" / group / (feature + ".h5")
            if not path.exists():
                raise FileNotFoundError(path)
            out = R / "runs" / group / feature / alg
            if out.exists():
                raise FileExistsError(out)
            print("START", group, feature, alg, flush=True)
            dest = run_algorithm_case(alg, path, H / "configs" / (alg + ".yaml"), out)
            status = yaml.safe_load((dest / "metadata.yaml").read_text())
            if status["status"] != "success":
                raise RuntimeError(status["failure_reason"])
            m = json.loads((dest / "metrics.json").read_text())
            row = {
                "group": group,
                "feature": feature,
                "algorithm": alg,
                **{
                    k: m.get(k)
                    for k in [
                        "rmse",
                        "ssim",
                        "psnr",
                        "water_background_rmse",
                        "data_relative_residual",
                        "stopping",
                    ]
                },
                "validation": m.get("evaluation", {}).get("receiver"),
                "split": m.get("evaluation_split"),
                "result_dir": str(dest),
            }
            rows.append(row)
            (out.parent / (alg + "_summary.json")).write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
