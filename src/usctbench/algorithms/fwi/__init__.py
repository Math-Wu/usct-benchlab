"""Tiny waveform-inversion proof-of-life components."""

from __future__ import annotations

from usctbench.core.registry import register_algorithm

from .adapter import KWaveFWIAdapterAlgorithm
from .diffusion_adapter import DiffusionKWaveFWIAdapterAlgorithm
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
        "fwi_kwave_adapter",
        KWaveFWIAdapterAlgorithm,
        specification=SPECS["fwi_kwave_adapter"],
        description="Adapter for k-Wave/WaveformInversionUST MATLAB FWI results.",
        tags=("fwi", "kwave", "external"),
        replace=replace,
    )
    register_algorithm(
        "diffusion_fwi_kwave_adapter",
        DiffusionKWaveFWIAdapterAlgorithm,
        description="Adapter for external diffusion-prior k-Wave/FWI DPS results.",
        tags=("fwi", "kwave", "diffusion", "external"),
        replace=replace,
    )


__all__ = [
    "DiffusionKWaveFWIAdapterAlgorithm",
    "KWaveFWIAdapterAlgorithm",
    "TinyFWIAlgorithm",
    "register_fwi_algorithms",
]
