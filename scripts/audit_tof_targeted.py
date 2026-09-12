"""Frozen ToF targeted diagnosis. Outputs are development controls, not clinical evidence."""

import argparse
from pathlib import Path
import json
import numpy as np
import yaml
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.core.schema import GroundTruthSpec, MeasurementDomain
from usctbench.cli import register_builtin_algorithms
from usctbench.benchmark.runner import run_algorithm_case
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.operators.eikonal import EikonalForward


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
    parser.add_argument("--mode", choices=["warm", "smooth"], required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve() == repo or repo in args.out.resolve().parents:
        raise ValueError("private outputs must stay outside the repository")

    R = args.out.resolve()
    H = args.handoff.resolve()
    register_builtin_algorithms()

    def run(case_path, alg, config, out):
        out.mkdir(parents=True, exist_ok=False)
        cp = out / "config.yaml"
        cp.write_text(yaml.safe_dump(config))
        dest = run_algorithm_case(alg, case_path, cp, out)
        meta = yaml.safe_load((dest / "metadata.yaml").read_text())
        assert meta["status"] == "success", meta["failure_reason"]
        m = json.loads((dest / "metrics.json").read_text())
        print(
            json.dumps(
                {
                    "out": str(out),
                    **{
                        k: m[k]
                        for k in ["rmse", "ssim", "data_relative_residual", "stopping"]
                    },
                }
            ),
            flush=True,
        )

    if args.mode == "warm":
        for feature in ["matched_eikonal", "kwave"]:
            cfg = yaml.safe_load((H / "configs/bent_ray_gn.yaml").read_text())
            cfg["parameters"].update(
                initialization="cgls", initialization_iterations=80
            )
            run(
                R / "controls/high_d" / (feature + ".h5"),
                "bent_ray_gn",
                cfg,
                R / "warm" / feature,
            )
    else:
        c = read_case_hdf5(R / "controls/high_d/kwave.h5")
        y, x = np.indices(c.grid.shape)
        y = c.grid.origin_m[0] + (y + 0.5) * c.grid.spacing_m[0]
        x = c.grid.origin_m[1] + (x + 0.5) * c.grid.spacing_m[1]

        def bump(cx, cy, r):
            rr = ((x - cx) ** 2 + (y - cy) ** 2) / r**2
            f = np.zeros_like(rr)
            inside = rr < 1
            f[inside] = np.exp(1 - 1 / (1 - rr[inside]))
            return f

        ds = (
            -35 * bump(-0.012, 0.004, 0.018) + 25 * bump(0.012, -0.008, 0.013)
        ) / 1500**2
        speed = 1 / (1 / 1500 + ds)
        c.ground_truth = GroundTruthSpec(sound_speed_mps=speed)
        mask = c.measurement.valid_mask & np.isfinite(c.measurement.delta_tof_s)
        c.metadata.update(
            measurement_provenance="oracle_travel_time",
            benchmark_type="smooth_analytic_matched_control",
            uses_kwave_wavefield=False,
            diagnostic_control="smooth_analytic_control",
            ground_truth_used_for_measurement=True,
            uses_gt_generated_measurement=True,
        )
        out = R / "smooth"
        out.mkdir(exist_ok=False)
        a = StraightRayProjector.from_case(c).forward(ds).reshape((64, 64))
        distance = np.linalg.norm(
            c.geometry.tx_pos_m[:, None] - c.geometry.rx_pos_m[None], axis=-1
        )
        e = (
            EikonalForward(c.grid, c.geometry).forward(1 / speed).reshape((64, 64))
            - distance / 1500
        )
        for alg in ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"]:
            c.measurement.delta_tof_s = np.where(
                mask, e if alg == "bent_ray_gn" else a, np.nan
            )
            c.measurement.domain = MeasurementDomain.FEATURES
            cp = out / (alg + ".h5")
            write_case_hdf5(c, cp)
            cfg = yaml.safe_load((H / "configs" / (alg + ".yaml")).read_text())
            run(cp, alg, cfg, out / alg)


if __name__ == "__main__":
    main()
