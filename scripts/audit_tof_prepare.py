"""Frozen ToF prepare diagnosis. Outputs are development controls, not clinical evidence."""

import argparse
from pathlib import Path
import gc
import json
import time
import numpy as np
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.operators.forward.straight_ray import StraightRayProjector
from tof_audit_common import diagnostic_case, streamed_eikonal_delta
from usctbench.evaluation.data import make_data_split


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
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve() == repo or repo in args.out.resolve().parents:
        raise ValueError("private outputs must stay outside the repository")

    R = args.out.resolve()
    H = args.handoff.resolve()
    paths = {
        "high_d": "high_band/D510022534/envelope_case.h5",
        "low_d": "low_band/D510022534/pressure_case.h5",
        "low_ob": "low_band/breast_train_speed_class_1_000000/pressure_case.h5",
    }

    def stats(a, b, mask, w):
        r = (a - b)[mask] * 1000000.0
        w = w[mask]
        signal = np.linalg.norm(b[mask] * 1000000.0)
        return {
            "count": int(mask.sum()),
            "rms_us": float(np.sqrt(np.mean(r * r))),
            "bias_us": float(r.mean()),
            "weighted_rms_us": float(np.sqrt(np.sum(w * r * r) / np.sum(w))),
            "relative": float(np.linalg.norm(r) / signal) if signal else None,
            "p95_abs_us": float(np.quantile(np.abs(r), 0.95)),
        }

    for group, p in paths.items():
        t = time.perf_counter()
        c = read_case_hdf5(H / "cases" / p)
        m = c.measurement
        m.time_data = m.water_reference_time = None
        speed = c.ground_truth.sound_speed_mps
        if c.grid.shape != (256, 256) or c.grid.roi_mask is not None:
            raise ValueError(
                "this frozen audit requires the original 256-square grid and no ROI"
            )
        if len(c.geometry.tx_pos_m) != 64 or len(c.geometry.rx_pos_m) != 64:
            raise ValueError("this frozen audit requires the original 64 TX/RX")
        op = StraightRayProjector.from_case(c)
        s = op.forward(1 / speed - 1 / 1500).reshape((64, 64))
        e = streamed_eikonal_delta(c.grid, c.geometry, speed)
        obs = m.delta_tof_s
        mask = m.valid_mask & np.isfinite(obs)
        w = m.ray_weights if m.ray_weights is not None else np.ones((64, 64))
        split = make_data_split(
            obs,
            valid_mask=mask,
            weights=w,
            receiver_fraction=0.125,
            seed=42,
            tx_positions=c.geometry.tx_pos_m,
            rx_positions=c.geometry.rx_pos_m,
        )
        dest = R / "controls" / group
        dest.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            dest / "predictions.npz",
            straight=s,
            eikonal=e,
            observed=obs,
            valid=mask,
            weights=w,
            train=split.train,
            validation=split.validation,
        )
        report = {
            "group": group,
            "case": p,
            "split": split.metadata,
            "forward_error_at_gt": {},
            "elapsed_s": time.perf_counter() - t,
        }
        for domain, sel in [
            ("all", mask),
            ("train", split.train),
            ("validation", split.validation),
        ]:
            report["forward_error_at_gt"][domain] = {
                "straight_vs_observed": stats(s, obs, sel, w),
                "eikonal_vs_observed": stats(e, obs, sel, w),
                "straight_vs_eikonal": stats(s, e, sel, w),
            }
        report["reciprocity"] = {
            name: stats(a, a.T, mask & mask.T, w)
            for name, a in [("straight", s), ("eikonal", e), ("observed", obs)]
        }
        for label, target in [
            ("matched_straight", s),
            ("matched_eikonal", e),
            ("kwave", obs),
        ]:
            write_case_hdf5(diagnostic_case(c, target, label), dest / (label + ".h5"))
        (dest / "diagnostic.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        del c, op
        gc.collect()


if __name__ == "__main__":
    main()
