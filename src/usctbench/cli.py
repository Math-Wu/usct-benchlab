"""Command-line entry point for the USCT benchmark harness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from usctbench.benchmark.runner import (
    evaluate_run,
    run_algorithm_case,
    run_benchmark_suite,
)
from usctbench.data.nbpslice2d import (
    inspect_nbp_slice2d_zip,
    make_nbp_slice2d_quality_subset,
    make_nbp_slice2d_smoke_subset,
)
from usctbench.data.openbreastus import (
    inspect_openbreastus,
    make_quality_subset,
    make_smoke_subset,
    write_schema_report,
)
from usctbench.data.synthetic import make_synthetic_smoke_subset
from usctbench.core.registry import get_algorithm_entry, list_algorithms


def register_builtin_algorithms() -> None:
    """Register built-in algorithms exactly once for CLI use."""

    from usctbench.algorithms.bent_ray import register_bent_ray_algorithm
    from usctbench.algorithms.fwi import register_fwi_algorithms
    from usctbench.algorithms.ray import register_ray_algorithms
    from usctbench.algorithms.rwave import register_rwave_algorithm

    register_ray_algorithms(replace=True)
    register_bent_ray_algorithm(replace=True)
    register_rwave_algorithm(replace=True)
    register_fwi_algorithms(replace=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usct", description="USCT benchmark command-line interface."
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser(
        "list-algorithms", help="List registered reconstruction algorithms."
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit authoritative machine-readable specifications.",
    )
    describe_parser = subparsers.add_parser(
        "describe-algorithm",
        help="Describe a registered algorithm and approved physical variants.",
    )
    describe_parser.add_argument("algorithm_id")
    describe_parser.add_argument(
        "--variant", help="Approved variant id from list-algorithms --json."
    )
    describe_parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON only."
    )
    describe_parser.add_argument(
        "--case",
        help="Optionally append case-bound capabilities from HDF5; does not run a solver.",
    )

    data_parser = subparsers.add_parser(
        "data", help="Data inspection and subset commands."
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    inspect_parser = data_subparsers.add_parser(
        "inspect-openbreastus", help="Inspect a local OpenBreastUS tree."
    )
    inspect_parser.add_argument("--root", required=True, help="OpenBreastUS data root.")
    inspect_parser.add_argument(
        "--out", default="openbreastus_index.json", help="Output index JSON path."
    )

    smoke_parser = data_subparsers.add_parser(
        "make-smoke", help="Create a smoke benchmark subset."
    )
    smoke_parser.add_argument("--root", required=True, help="OpenBreastUS data root.")
    smoke_parser.add_argument("--out", required=True, help="Output smoke subset root.")
    smoke_parser.add_argument("--cases-per-density", type=int, default=1)
    smoke_parser.add_argument(
        "--converted-shape",
        type=int,
        default=64,
        help="Square image size for converted speed-map cases.",
    )
    smoke_parser.add_argument(
        "--n-transducers",
        type=int,
        default=32,
        help="Synthetic ring transducers for converted speed-map cases.",
    )
    smoke_parser.add_argument(
        "--spacing-m",
        type=float,
        default=1.0e-3,
        help="Assumed pixel spacing for converted speed-map cases.",
    )
    smoke_parser.add_argument(
        "--no-convert-speed-mat",
        action="store_true",
        help="Only write the manifest; do not create HDF5 cases.",
    )

    quality_parser = data_subparsers.add_parser(
        "make-quality", help="Create 256x256 OpenBreastUS quality-comparison cases."
    )
    quality_parser.add_argument("--root", required=True, help="OpenBreastUS data root.")
    quality_parser.add_argument(
        "--out", required=True, help="Output quality subset root."
    )
    quality_parser.add_argument("--cases-per-density", type=int, default=1)
    quality_parser.add_argument(
        "--converted-shape",
        type=int,
        default=256,
        help="Square image size for quality comparison.",
    )
    quality_parser.add_argument(
        "--n-transducers",
        type=int,
        default=128,
        help="Synthetic ring transducers for quality comparison.",
    )
    quality_parser.add_argument(
        "--spacing-m",
        type=float,
        default=1.0e-3,
        help="Assumed pixel spacing for converted speed-map cases.",
    )
    quality_parser.add_argument(
        "--no-convert-speed-mat",
        action="store_true",
        help="Only write the manifest; do not create HDF5 cases.",
    )

    nbp_inspect_parser = data_subparsers.add_parser(
        "inspect-nbpslice2d", help="Inspect an NBPslices2D ZIP archive."
    )
    nbp_inspect_parser.add_argument(
        "--zip", required=True, help="NBPslices2D ZIP path."
    )
    nbp_inspect_parser.add_argument(
        "--out", default="nbpslice2d_index.json", help="Output index JSON path."
    )

    nbp_smoke_parser = data_subparsers.add_parser(
        "make-nbp-smoke", help="Create an NBPslices2D smoke benchmark subset."
    )
    nbp_smoke_parser.add_argument("--zip", required=True, help="NBPslices2D ZIP path.")
    nbp_smoke_parser.add_argument(
        "--out", required=True, help="Output smoke subset root."
    )
    nbp_smoke_parser.add_argument(
        "--cases-per-type",
        type=int,
        default=1,
        help="Cases to convert for each A/B/C/D density label.",
    )
    nbp_smoke_parser.add_argument(
        "--converted-shape",
        type=int,
        default=64,
        help="Square image size for converted cases.",
    )
    nbp_smoke_parser.add_argument(
        "--n-transducers",
        type=int,
        default=32,
        help="Synthetic ring transducers for converted cases.",
    )
    nbp_smoke_parser.add_argument(
        "--reference-sound-speed-mps", type=float, default=1500.0
    )

    nbp_quality_parser = data_subparsers.add_parser(
        "make-nbp-quality", help="Create 256x256 NBPslice2D quality-comparison cases."
    )
    nbp_quality_parser.add_argument(
        "--zip", required=True, help="NBPslices2D ZIP path."
    )
    nbp_quality_parser.add_argument(
        "--out", required=True, help="Output quality subset root."
    )
    nbp_quality_parser.add_argument(
        "--cases-per-type",
        type=int,
        default=1,
        help="Cases to convert for each A/B/C/D density label.",
    )
    nbp_quality_parser.add_argument(
        "--converted-shape",
        type=int,
        default=256,
        help="Square image size for quality comparison.",
    )
    nbp_quality_parser.add_argument(
        "--n-transducers",
        type=int,
        default=128,
        help="Synthetic ring transducers for quality comparison.",
    )
    nbp_quality_parser.add_argument(
        "--reference-sound-speed-mps", type=float, default=1500.0
    )

    synthetic_smoke_parser = data_subparsers.add_parser(
        "make-synthetic-smoke", help="Create deterministic local synthetic smoke cases."
    )
    synthetic_smoke_parser.add_argument(
        "--out",
        default="data/synthetic_smoke",
        help="Output synthetic smoke subset root.",
    )
    synthetic_smoke_parser.add_argument(
        "--shape", type=int, default=24, help="Square image size."
    )
    synthetic_smoke_parser.add_argument("--n-transducers", type=int, default=24)

    run_parser = subparsers.add_parser("run", help="Run one algorithm on one case.")
    run_parser.add_argument("algorithm", help="Registered algorithm name.")
    run_parser.add_argument("--case", required=True, help="Input case HDF5 path.")
    run_parser.add_argument(
        "--config", required=True, help="Algorithm YAML config path."
    )
    run_parser.add_argument("--out", required=True, help="Output run directory.")

    eval_parser = subparsers.add_parser("eval", help="Evaluate one run directory.")
    eval_parser.add_argument("--run", required=True, help="Run directory.")
    eval_parser.add_argument(
        "--protocol", required=True, help="Benchmark protocol YAML path."
    )

    bench_parser = subparsers.add_parser("bench", help="Run a benchmark suite.")
    bench_parser.add_argument(
        "--suite", required=True, help="Benchmark suite YAML path."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    register_builtin_algorithms()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "list-algorithms":
        entries = list_algorithms()
        if args.json:
            print(
                json.dumps(
                    [entry.describe() for entry in entries], indent=2, allow_nan=False
                )
            )
            return 0
        if not entries:
            print("No algorithms registered.")
            return 0
        for entry in entries:
            tags = f" [{' '.join(entry.tags)}]" if entry.tags else ""
            suffix = f" - {entry.description}" if entry.description else ""
            print(f"{entry.name}{tags}{suffix}")
        return 0

    if args.command == "describe-algorithm":
        try:
            description = get_algorithm_entry(args.algorithm_id).describe(args.variant)
            if args.case:
                from usctbench.core.algorithm_specs import case_capabilities
                from usctbench.core.io import read_case_hdf5

                description["case_capabilities"] = case_capabilities(
                    args.algorithm_id, read_case_hdf5(args.case), variant=args.variant
                )
        except (KeyError, ValueError, OSError) as exc:
            parser.error(str(exc))
        if args.json:
            print(json.dumps(description, indent=2, allow_nan=False))
        else:
            print(f"{description['algorithm_id']}: {description['description']}")
            print(json.dumps(description, indent=2, allow_nan=False))
        return 0

    if args.command == "data":
        if args.data_command is None:
            parser.parse_args(["data", "--help"])
            return 0
        if args.data_command == "inspect-openbreastus":
            index = inspect_openbreastus(args.root, args.out)
            write_schema_report(index, Path(args.out).with_suffix(".schema_report.md"))
            print(args.out)
            return 0
        if args.data_command == "make-smoke":
            make_smoke_subset(
                args.root,
                args.out,
                cases_per_density=args.cases_per_density,
                convert_speed_mat=not args.no_convert_speed_mat,
                converted_shape=(args.converted_shape, args.converted_shape),
                spacing_m=(args.spacing_m, args.spacing_m),
                n_transducers=args.n_transducers,
            )
            print(args.out)
            return 0
        if args.data_command == "make-quality":
            make_quality_subset(
                args.root,
                args.out,
                cases_per_density=args.cases_per_density,
                convert_speed_mat=not args.no_convert_speed_mat,
                converted_shape=(args.converted_shape, args.converted_shape),
                spacing_m=(args.spacing_m, args.spacing_m),
                n_transducers=args.n_transducers,
            )
            print(args.out)
            return 0
        if args.data_command == "inspect-nbpslice2d":
            inspect_nbp_slice2d_zip(args.zip, args.out)
            print(args.out)
            return 0
        if args.data_command == "make-nbp-smoke":
            make_nbp_slice2d_smoke_subset(
                args.zip,
                args.out,
                cases_per_type=args.cases_per_type,
                converted_shape=(args.converted_shape, args.converted_shape),
                n_transducers=args.n_transducers,
                reference_sound_speed_mps=args.reference_sound_speed_mps,
            )
            print(args.out)
            return 0
        if args.data_command == "make-nbp-quality":
            make_nbp_slice2d_quality_subset(
                args.zip,
                args.out,
                cases_per_type=args.cases_per_type,
                converted_shape=(args.converted_shape, args.converted_shape),
                n_transducers=args.n_transducers,
                reference_sound_speed_mps=args.reference_sound_speed_mps,
            )
            print(args.out)
            return 0
        if args.data_command == "make-synthetic-smoke":
            make_synthetic_smoke_subset(
                args.out,
                shape=(args.shape, args.shape),
                n_transducers=args.n_transducers,
            )
            print(args.out)
            return 0

    if args.command == "run":
        out_dir = run_algorithm_case(args.algorithm, args.case, args.config, args.out)
        print(out_dir / "result.h5")
        return 0 if not (out_dir / "failure_report.md").exists() else 1

    if args.command == "eval":
        result = evaluate_run(args.run, args.protocol)
        print(result["report_md"])
        return 0 if _benchmark_passed(result) else 1

    if args.command == "bench":
        result = run_benchmark_suite(args.suite)
        print(result["run_root"])
        return 0 if _benchmark_passed(result) else 1

    parser.error(f"unknown command: {args.command}")
    return 2


def _benchmark_passed(result: dict) -> bool:
    return (
        bool(result["records"])
        and all(record.get("pass") for record in result["records"])
        and bool(result.get("run_checks", {}).get("passed"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
