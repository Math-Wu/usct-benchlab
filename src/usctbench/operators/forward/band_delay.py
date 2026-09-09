"""Bounded finite-band correlation arrivals and their implicit derivatives.

The observable maximizes Re sum_f a_f Q_f exp(-i omega_f tau), where Q is
pressure divided by independent water. It is NOT a geometrical first arrival.
Derivatives are local to a unique interior maximum; switching peaks is nonsmooth.
"""

from dataclasses import dataclass

import numpy as np

from usctbench.operators.base import Linearization
from usctbench.operators.forward.correlation_delay import CorrelationDelayDerivative


@dataclass
class BandDelayLinearization:
    value: np.ndarray
    valid: np.ndarray
    coherence: np.ndarray
    peak_gap: np.ndarray
    coefficient: np.ndarray

    def forward(self, pressure_ratio_perturbation):
        value = np.asarray(pressure_ratio_perturbation, dtype=complex)
        if value.shape != self.coefficient.shape[1:]:
            raise ValueError("pressure perturbation shape mismatch")
        active = np.any(np.abs(self.coefficient) > 0, axis=0)
        if not np.isfinite(value[active]).all():
            raise ValueError("nonfinite active pressure perturbation")
        return np.einsum(
            "bftr,ftr->btr", self.coefficient, np.where(active, value, 0)
        ).imag

    def adjoint(self, delay_sensitivity):
        value = np.asarray(delay_sensitivity)
        if np.iscomplexobj(value) or value.shape != self.value.shape:
            raise ValueError("require real band-delay sensitivity")
        if not np.isfinite(value[self.valid]).all():
            raise ValueError("nonfinite active delay sensitivity")
        return np.einsum(
            "bftr,btr->ftr",
            1j * self.coefficient.conj(),
            np.where(self.valid, value, 0),
        )


class BandCorrelationDelay:
    """Identical water-weighted observation map for measurements and predictions.

    `bands` is (band, frequency), nonnegative fixed spectral power weights.
    Bounds are per-pair seconds, fixed from geometry/speed limits without GT.
    Frequency holdouts must be removed BEFORE constructing bands; overlapping
    bands cannot be called independent frequency validation data.
    """

    def __init__(
        self, frequencies_hz, water, bands, lower_s, upper_s, *, oversample=32
    ):
        base = CorrelationDelayDerivative(frequencies_hz, water)
        self.frequencies_hz = base.frequencies_hz
        self.omega = 2 * np.pi * self.frequencies_hz
        self.data_shape = base.data_shape
        bands = np.asarray(bands, dtype=float)
        if (
            bands.ndim != 2
            or bands.shape[1] != len(self.omega)
            or not np.isfinite(bands).all()
            or np.any(bands < 0)
            or np.any(np.sum(bands > 0, axis=1) < 3)
        ):
            raise ValueError("every band requires at least three positive weights")
        if (
            isinstance(oversample, bool)
            or int(oversample) != oversample
            or oversample < 8
        ):
            raise ValueError("oversample must be an integer at least 8")
        self.lower = np.broadcast_to(np.asarray(lower_s, float), base.ray_shape)
        self.upper = np.broadcast_to(np.asarray(upper_s, float), base.ray_shape)
        if (
            not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or np.any(self.lower >= self.upper)
        ):
            raise ValueError("require finite ordered per-pair delay bounds")
        mag = np.where(base.frequency_valid, np.abs(water), 0)
        scale = np.maximum(mag.max(axis=0), np.finfo(float).tiny)
        energy = (mag / scale[None]) ** 2
        power = bands[:, :, None, None] * energy[None]
        total = power.sum(axis=1, keepdims=True)
        self.power = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
        self.base_valid = np.sum(power > 0, axis=1) >= 3
        step = 1 / (oversample * self.frequencies_hz[-1])
        count = int(np.ceil((self.upper.max() - self.lower.min()) / step)) + 3
        if count > 20000:
            raise ValueError("delay range too large for bounded correlation")
        self.lags = np.linspace(self.lower.min() - step, self.upper.max() + step, count)
        self.step = self.lags[1] - self.lags[0]
        self.phasors = np.exp(-1j * self.lags[:, None] * self.omega[None])

    def linearize(self, pressure_ratio):
        ratio = np.asarray(pressure_ratio, dtype=complex)
        if ratio.shape != self.data_shape:
            raise ValueError("pressure ratio shape mismatch")
        active = np.any(self.power > 0, axis=0)
        if not np.isfinite(ratio[active]).all():
            raise FloatingPointError("nonfinite pressure in correlation band")
        ratio = np.where(active, ratio, 0)
        shape = self.base_valid.shape
        delay = np.zeros(shape)
        valid = self.base_valid.copy()
        coherence, gap = np.zeros(shape), np.zeros(shape)
        coefficients = np.zeros_like(self.power, dtype=complex)
        for b in range(shape[0]):
            weights = self.power[b].reshape(len(self.omega), -1)
            q = ratio.reshape(len(self.omega), -1)
            for start in range(0, q.shape[1], 128):
                stop = min(start + 128, q.shape[1])
                w, z = weights[:, start:stop], q[:, start:stop]
                lo, hi = self.lower.ravel()[start:stop], self.upper.ravel()[start:stop]
                correlation = (self.phasors @ (w * z)).real
                inside = (self.lags[:, None] >= lo) & (self.lags[:, None] <= hi)
                candidate = np.where(inside, correlation, -np.inf)
                peak = np.argmax(candidate, axis=0)
                tau = self.lags[peak].copy()
                left = np.maximum(lo, tau - self.step)
                right = np.minimum(hi, tau + self.step)
                for _ in range(6):
                    phase = np.exp(-1j * self.omega[:, None] * tau)
                    rotated = w * z * phase
                    slope = np.sum(self.omega[:, None] * rotated.imag, axis=0)
                    curvature = np.sum(self.omega[:, None] ** 2 * rotated.real, axis=0)
                    increment = np.divide(
                        slope, curvature, out=np.zeros_like(tau), where=curvature > 0
                    )
                    tau = np.clip(tau + increment, left, right)
                phase = np.exp(-1j * self.omega[:, None] * tau)
                rotated = w * z * phase
                score = rotated.real.sum(axis=0)
                curvature = np.sum(self.omega[:, None] ** 2 * rotated.real, axis=0)
                envelope = np.sum(w * np.abs(z), axis=0)
                curvature_scale = np.sum(
                    w * np.abs(z) * self.omega[:, None] ** 2, axis=0
                )
                okay = (curvature > 1e-4 * curvature_scale) & (envelope > 0)
                okay &= (tau > lo + self.step) & (tau < hi - self.step)
                local_max = np.zeros_like(inside)
                local_max[1:-1] = (correlation[1:-1] > correlation[:-2]) & (
                    correlation[1:-1] >= correlation[2:]
                )
                competitor = (
                    local_max
                    & inside
                    & (np.abs(self.lags[:, None] - tau) > 2 * self.step)
                )
                second = np.max(np.where(competitor, correlation, 0), axis=0)
                delay[b].ravel()[start:stop] = tau
                valid[b].ravel()[start:stop] &= okay
                coherence[b].ravel()[start:stop] = np.divide(
                    score, envelope, out=np.zeros_like(score), where=envelope > 0
                )
                gap[b].ravel()[start:stop] = np.divide(
                    score - second,
                    np.abs(score),
                    out=np.zeros_like(score),
                    where=np.abs(score) > 0,
                )
                coefficient = np.divide(
                    w * self.omega[:, None] * phase,
                    curvature[None],
                    out=np.zeros_like(phase),
                    where=okay[None],
                )
                coefficients[b].reshape(len(self.omega), -1)[
                    :, start:stop
                ] = coefficient
        coefficients = np.where(valid[:, None], coefficients, 0)
        return BandDelayLinearization(
            np.where(valid, delay, np.nan), valid, coherence, gap, coefficients
        )


class _ComposedJacobian:
    def __init__(self, pressure, observation, reference):
        self.pressure, self.observation, self.reference = (
            pressure,
            observation,
            reference,
        )
        self.grid = pressure.grid

    def forward(self, perturbation):
        return self.observation.forward(
            self.pressure.forward(perturbation) / self.reference
        )

    def adjoint(self, sensitivity):
        return self.pressure.adjoint(
            self.observation.adjoint(sensitivity) / self.reference.conj()
        )


class FiniteFrequencyTravelTimeForward:
    """Nonlinear H(F(m)) with the full discrete chain rule, m = c^-2.

    This is finite-frequency traveltime physics, NOT renamed geometric CGLS/Bent.
    Only the volume-integral backend provides the claimed discrete derivative.
    """

    def __init__(self, pressure_forward, observation, reference):
        if pressure_forward.settings.get("green_backend") != "volume_integral":
            raise ValueError("exact composition requires volume_integral Green physics")
        if not np.array_equal(
            pressure_forward.frequencies_hz, observation.frequencies_hz
        ):
            raise ValueError("pressure and observation frequencies differ")
        self.pressure, self.observation = pressure_forward, observation
        self.reference = np.asarray(reference, complex)
        if (
            self.reference.shape != observation.data_shape
            or not np.isfinite(self.reference).all()
            or np.any(np.abs(self.reference) == 0)
        ):
            raise ValueError("require finite nonzero predicted water pressure")
        self.grid = pressure_forward.grid

    def linearize(self, model):
        pressure = self.pressure.linearize(model)
        observation = self.observation.linearize(pressure.value / self.reference)
        return Linearization(
            observation.value,
            _ComposedJacobian(pressure.jacobian, observation, self.reference),
            derivative_kind="discrete_full_green_and_implicit_correlation_peak",
        )

    def forward(self, model):
        return self.linearize(model).value

    def __getattr__(self, name):
        if name in {
            "background_builds",
            "eikonal_solves",
            "green_solves",
            "green_matvecs",
        }:
            return getattr(self.pressure, name)
        raise AttributeError(name)
