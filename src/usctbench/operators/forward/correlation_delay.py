"""Finite-frequency correlation-delay derivative, not a geometric ray integral.

For the positive-sign Fourier transform, delayed water has ratio exp(+i*w*tau).
Linearizing the maximum of its water-weighted correlation at tau=0 gives
dtau = sum(w |P0|^2 Im(dP/P0)) / sum(w^2 |P0|^2).
This is a small-perturbation observable and must not be relabelled first-arrival
ToF, phase unwrapping, or a general finite-amplitude delay estimator.
"""

import numpy as np


class CorrelationDelayDerivative:
    """Pair-local real-linear pressure-ratio to delay map and exact transpose.

    Frequencies and water pressure must already obey the acquisition/holdout
    contract. Uniform frequency quadrature is not assumed: optional positive
    quadrature weights multiply spectral energy. At least three usable bands
    are required per pair; invalid pairs produce NaN forward values.
    """

    def __init__(self, frequencies_hz, water, *, quadrature=None, relative_floor=1e-3):
        f = np.asarray(frequencies_hz, dtype=float)
        ref = np.asarray(water, dtype=complex)
        if (
            f.ndim != 1
            or len(f) < 3
            or not np.isfinite(f).all()
            or np.any(f <= 0)
            or np.any(np.diff(f) <= 0)
            or ref.ndim != 3
            or ref.shape[0] != len(f)
        ):
            raise ValueError(
                "require increasing positive frequencies and (f,tx,rx) water"
            )
        if not np.isfinite(relative_floor) or not 0 < relative_floor < 1:
            raise ValueError("relative_floor must be in (0,1)")
        q = (
            np.ones(len(f))
            if quadrature is None
            else np.asarray(quadrature, dtype=float)
        )
        if q.shape != f.shape or not np.isfinite(q).all() or np.any(q <= 0):
            raise ValueError("quadrature must be a positive finite frequency vector")
        self.data_shape = ref.shape
        self.frequencies_hz = f.copy()
        self.ray_shape = ref.shape[1:]
        magnitude = np.where(np.isfinite(ref), np.abs(ref), 0)
        scale = magnitude.max(axis=0)
        usable = np.isfinite(ref) & (magnitude > relative_floor * scale[None])
        self.valid = (usable.sum(axis=0) >= 3) & (scale > 0)
        self.frequency_valid = usable & self.valid[None]
        energy = (magnitude / np.maximum(scale, np.finfo(float).tiny)[None]) ** 2
        energy *= q[:, None, None]
        energy = np.where(self.frequency_valid, energy, 0)
        omega = 2 * np.pi * f[:, None, None]
        denom = np.sum(omega**2 * energy, axis=0)
        self.coefficient = np.divide(
            omega * energy,
            denom[None],
            out=np.zeros_like(energy),
            where=denom[None] > 0,
        )

    def forward(self, relative_scattered):
        value = np.asarray(relative_scattered, dtype=complex)
        if value.shape != self.data_shape:
            raise ValueError("pressure ratio shape mismatch")
        if not np.isfinite(value[self.frequency_valid]).all():
            raise ValueError("nonfinite pressure on an active frequency")
        result = np.sum(
            self.coefficient * np.where(self.frequency_valid, value.imag, 0), axis=0
        )
        return np.where(self.valid, result, np.nan)

    def adjoint(self, delay_sensitivity):
        if np.iscomplexobj(delay_sensitivity):
            raise ValueError("delay sensitivity must be real")
        value = np.asarray(delay_sensitivity, dtype=float)
        if value.shape != self.ray_shape or not np.isfinite(value[self.valid]).all():
            raise ValueError("require finite sensitivity on valid receiver pairs")
        return 1j * self.coefficient * np.where(self.valid, value, 0)[None]


class CorrelationTravelTimeJacobian:
    """Compose existing Born physics and a fixed observation derivative.

    Input is squared-slowness perturbation, not sound speed. Water normalization
    cancels a multiplicative source/receiver response, not general model error.
    No training, GT correction or alteration of the straight-ray operator occurs.
    """

    def __init__(self, born, observation):
        if not np.all(born.background == born.exterior_speed_mps):
            raise ValueError("this composition requires a homogeneous water background")
        if not np.array_equal(born.frequencies_hz, observation.frequencies_hz):
            raise ValueError("Born and observation frequencies differ")
        self.born, self.observation = born, observation
        self.grid = born.grid
        self.ray_shape = observation.ray_shape
        self.n_rays = int(np.prod(self.ray_shape))
        self.reference = born.background_data()
        if self.reference.shape != observation.data_shape:
            raise ValueError("Born and observation shape mismatch")
        if np.any(
            np.abs(self.reference[observation.frequency_valid]) <= np.finfo(float).tiny
        ):
            raise ValueError("vanishing predicted water reference")

    def forward(self, squared_slowness_perturbation):
        scattered = self.born.forward(squared_slowness_perturbation)
        relative = np.divide(
            scattered,
            self.reference,
            out=np.zeros_like(scattered),
            where=self.observation.frequency_valid,
        )
        return self.observation.forward(relative).ravel()

    def adjoint(self, values):
        sensitivity = self.observation.adjoint(
            np.asarray(values).reshape(self.ray_shape)
        )
        pressure_sensitivity = np.divide(
            sensitivity,
            np.conj(self.reference),
            out=np.zeros_like(sensitivity),
            where=self.observation.frequency_valid,
        )
        return self.born.adjoint(pressure_sensitivity)
