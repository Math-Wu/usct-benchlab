#!/usr/bin/env python3
"""Repository-level release audit for the package."""

import re
import subprocess
from pathlib import Path

REQUIRED = [
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
    ".env.example",
    ".gitignore",
    ".github/workflows/tests.yml",
    "scripts/audit_release.py",
    "src/usctbench/cli.py",
    "src/usctbench/core/config.py",
    "src/usctbench/core/schema.py",
    "src/usctbench/core/io.py",
    "src/usctbench/core/registry.py",
    "src/usctbench/algorithms/ray.py",
    "src/usctbench/algorithms/bent_ray.py",
    "src/usctbench/algorithms/rwave.py",
    "src/usctbench/algorithms/fwi/algorithm.py",
    "src/usctbench/algorithms/fwi/runtime.py",
    "src/usctbench/algorithms/fwi/io.py",
    "src/usctbench/algorithms/fwi/parameters.py",
    "configs/algorithms/bent_ray.yaml",
    "configs/algorithms/cgls.yaml",
    "configs/algorithms/fwi_wust.yaml",
    "src/usctbench/algorithms/fwi/tiny.py",
    "configs/algorithms/rwave.yaml",
    "configs/algorithms/sart.yaml",
    "configs/algorithms/sirt.yaml",
    "configs/benchmarks/synthetic_demo.yaml",
    "configs/benchmarks/nbpslice2d_demo.yaml",
    "configs/benchmarks/openbreastus_demo.yaml",
    "configs/benchmarks/fwi_wust_demo.yaml",
    "docs/README.md",
    "docs/math_formulation.md",
    "docs/usage.md",
    "docs/algorithms.md",
    "docs/datasets.md",
    "docs/fwi.md",
    "docs/development.md",
    "docs/references.bib",
    "docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png",
    "docs/assets/openbreastus_readme_fwi_vs_surrogate.png",
    "examples/README.md",
    "examples/synthetic_quickstart.sh",
    "tests/algorithms/test_fwi_wust.py",
    "tests/release/test_release_integrity.py",
]

FORBIDDEN_SUFFIXES = (
    ".h5",
    ".hdf5",
    ".mat",
    ".npy",
    ".npz",
    ".zarr",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
)
FORBIDDEN_DIRS = [
    "src/usctbench/features",
    "src/usctbench/sim",
    "src/usctbench/adapters",
    "configs/benchmarks/archive",
    "configs/algorithms/experimental",
    "scripts/experimental",
    "scripts/matlab_adapters",
    "docs/internal",
]
FORBIDDEN_FILES = [
    "src/usctbench/algorithms/fwi/adapter.py",
    "src/usctbench/algorithms/fwi/controlled.py",
    "src/usctbench/algorithms/fwi/diffusion_adapter.py",
    "AGENTS.md",
    "CODEX_GOAL_PROMPT.md",
    "codex_goal_prompt.md",
]
MAX_TRACKED_FILE_BYTES = 1_000_000
ALLOWED_LARGE_FILES = {
    "docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png",
    "docs/assets/openbreastus_readme_fwi_vs_surrogate.png",
}
ABSOLUTE_USER_PATH_RE = re.compile(r"(?<![\w$])/(?:Users|home)/[A-Za-z0-9._-]+")
ABSOLUTE_PATH_ALLOWLIST = (
    "/Users/example",
    "/home/example",
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    missing = [path for path in REQUIRED if not (root / path).exists()]
    if missing:
        failures.append("missing required files: " + ", ".join(missing))

    existing_forbidden_dirs = [
        path for path in FORBIDDEN_DIRS if (root / path).exists()
    ]
    if existing_forbidden_dirs:
        failures.append(
            "forbidden directories still exist: " + ", ".join(existing_forbidden_dirs)
        )

    existing_forbidden_files = [
        path for path in FORBIDDEN_FILES if (root / path).exists()
    ]
    if existing_forbidden_files:
        failures.append(
            "forbidden development files still exist: "
            + ", ".join(existing_forbidden_files)
        )

    tracked = _git_ls_files(root)
    data_files = [path for path in tracked if path.endswith(FORBIDDEN_SUFFIXES)]
    if data_files:
        failures.append("tracked data/checkpoint files: " + ", ".join(data_files))

    large_files = _large_tracked_files(root, tracked)
    if large_files:
        failures.append("unexpected large tracked files: " + ", ".join(large_files))

    old_terms = _grep(
        root,
        ["diagnostic-only", "retired", "observable mismatch", "kwave_unified"],
        ["README.md", "configs", "src", "tests"],
    )
    if old_terms:
        failures.append("old internal experiment terms found in user-facing paths")

    personal_paths = _grep_regex(
        root,
        ABSOLUTE_USER_PATH_RE,
        ["README.md", "docs", "configs", "scripts", "src", "tests", ".env.example"],
        allowed_substrings=ABSOLUTE_PATH_ALLOWLIST,
    )
    if personal_paths:
        failures.append("personal absolute paths found: " + ", ".join(personal_paths))

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS release audit")
    return 0


def _git_ls_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip())
    return proc.stdout.splitlines()


def _large_tracked_files(root: Path, tracked: list[str]) -> list[str]:
    large: list[str] = []
    for path in tracked:
        if path in ALLOWED_LARGE_FILES:
            continue
        file_path = root / path
        if file_path.is_file() and file_path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            large.append(f"{path}:{file_path.stat().st_size}")
    return large


def _grep(root: Path, patterns: list[str], targets: list[str]) -> list[str]:
    results: list[str] = []
    for target in targets:
        path = root / target
        if not path.exists():
            continue
        files = (
            [path]
            if path.is_file()
            else [item for item in path.rglob("*") if item.is_file()]
        )
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in patterns:
                if pattern in text:
                    results.append(f"{file_path.relative_to(root)}:{pattern}")
    return results


def _grep_regex(
    root: Path,
    pattern: re.Pattern[str],
    targets: list[str],
    *,
    allowed_substrings: tuple[str, ...] = (),
) -> list[str]:
    results: list[str] = []
    for target in targets:
        path = root / target
        if not path.exists():
            continue
        files = (
            [path]
            if path.is_file()
            else [item for item in path.rglob("*") if item.is_file()]
        )
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if any(allowed in line for allowed in allowed_substrings):
                    continue
                match = pattern.search(line)
                if match:
                    results.append(
                        f"{file_path.relative_to(root)}:{line_number}:{match.group(0)}"
                    )
    return results


if __name__ == "__main__":
    raise SystemExit(main())
