"""Tiny waveform-inversion proof-of-life components."""

from __future__ import annotations

from usctbench.core.registry import register_algorithm

from .algorithm import WUSTFWIAlgorithm
from .tiny import TinyFWIAlgorithm


def register_fwi_algorithms(*, replace: bool = False) -> None:
    from usctbench.core.algorithm_specs import SPECS

    register_algorithm(
        "fwi_tiny",
        TinyFWIAlgorithm,
        specification=SPECS["fwi_tiny"],
        description="Tiny synthetic waveform-inversion proof-of-life.",
        tags=("fwi", "synthetic", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "fwi_wust",
        WUSTFWIAlgorithm,
        specification=SPECS["fwi_wust"],
        description="Production sound-speed FWI through pinned WUST CUDA runtime.",
        tags=("fwi", "full_wave", "wust"),
        replace=replace,
    )


__all__ = [
    "WUSTFWIAlgorithm",
    "TinyFWIAlgorithm",
    "register_fwi_algorithms",
]
