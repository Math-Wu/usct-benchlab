"""NBPslices2D numerical breast phantom smoke-subset helpers."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .conversion import NBP_DENSITY_CLASSES, convert_nbp_slice2d_zip


def inspect_nbp_slice2d_zip(
    zip_path: str | Path, out: str | Path | None = None
) -> dict[str, Any]:
    """Inspect a local NBPslices2D ZIP archive without extracting data."""

    source = Path(zip_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"NBPslices2D ZIP does not exist: {source}")
    with zipfile.ZipFile(source) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".mat") and "__MACOSX" not in Path(name).parts
        ]
    label_counts = Counter(Path(name).stem[:1].upper() or "unknown" for name in members)
    index = {
        "schema_version": "0.1",
        "source_zip": str(source),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "num_cases": len(members),
            "density_label_counts": dict(sorted(label_counts.items())),
            "density_class_counts": {
                NBP_DENSITY_CLASSES.get(label, "unknown"): count
                for label, count in sorted(label_counts.items())
            },
        },
        "cases": [
            {
                "member": name,
                "case_id": Path(name).stem,
                "density_label": Path(name).stem[:1].upper(),
                "density_class": NBP_DENSITY_CLASSES.get(
                    Path(name).stem[:1].upper(), "unknown"
                ),
            }
            for name in sorted(members)
        ],
    }
    if out is not None:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
        )
    return index


def make_nbp_slice2d_smoke_subset(
    zip_path: str | Path,
    out: str | Path,
    *,
    cases_per_type: int = 1,
    converted_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
    subset_role: str = "interface_smoke",
) -> dict[str, Any]:
    """Create standard USCTCase smoke cases from NBPslices2D."""

    out_path = Path(out).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    index = inspect_nbp_slice2d_zip(zip_path, out_path / "nbpslice2d_index.json")
    cases_root = out_path / "cases"
    converted_cases = convert_nbp_slice2d_zip(
        zip_path,
        cases_root,
        cases_per_type=cases_per_type,
        output_shape=converted_shape,
        n_transducers=n_transducers,
        reference_sound_speed_mps=reference_sound_speed_mps,
    )
    manifest = {
        "schema_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_zip": str(Path(zip_path).expanduser().resolve()),
        "subset_root": str(out_path),
        "cases_per_type": cases_per_type,
        "converted_shape": list(converted_shape),
        "n_transducers": n_transducers,
        "subset_role": subset_role,
        "reference_sound_speed_mps": reference_sound_speed_mps,
        "index_summary": index["summary"],
        "converted_cases": converted_cases,
        "notes": [
            "NBPslices2D contains numerical phantom property maps, not measured RF data.",
            "Converted cases use straight-ray surrogate travel-time features.",
            "Sound speed is converted from mm/us to m/s.",
        ],
    }
    manifest_path = out_path / "nbpslice2d_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def make_nbp_slice2d_quality_subset(
    zip_path: str | Path,
    out: str | Path,
    *,
    cases_per_type: int = 1,
    converted_shape: tuple[int, int] = (256, 256),
    n_transducers: int = 128,
    reference_sound_speed_mps: float = 1500.0,
) -> dict[str, Any]:
    """Create 256x256 NBPslice2D cases for visual quality comparison."""

    return make_nbp_slice2d_smoke_subset(
        zip_path,
        out,
        cases_per_type=cases_per_type,
        converted_shape=converted_shape,
        n_transducers=n_transducers,
        reference_sound_speed_mps=reference_sound_speed_mps,
        subset_role="quality_comparison",
    )
