"""Finite-frequency single-scattering operator with refracted Green functions.

For squared slowness m, K dm = omega^2 integral G(r,x) G(x,s) q_s dm(x) dx.
The Fourier convention is exp(-i omega t); the outgoing homogeneous 2-D Green
function is i/4 H0^(1)(k r), satisfying (Laplacian + k^2) G = -delta.
Heterogeneous phase uses fast marching; positive WKB intensity solves the
conservative transport equation div(A^2 grad(T)) = 0 on upwind cell faces.

This is a fixed-background ray-Born linearization, not a full-wave solver or a
complete port of the upstream r-Wave Hessian-free, paraxial shooting algorithm.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.special import hankel1

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.forward.eikonal import EikonalForward, fast_march


def homogeneous_green(
    distance: np.ndarray, frequency_hz: float, speed_mps: float, radius: float
):
    """Outgoing 2-D Green function with an explicit cell-scale source cutoff."""
    r = np.maximum(np.asarray(distance, dtype=float), radius)
    return 0.25j * hankel1(0, (2 * np.pi * frequency_hz / speed_mps) * r)


def transport_log_amplitude(tape, slowness: np.ndarray, spacing, source):
    """Conservative first-order transport (no caustic phase corrections).

    Do not differentiate a numerical first-arrival front twice to build its
    curvature: front/stencil changes become spurious attenuation and focusing.
    Use shared face fluxes, positive intensity and independent water calibration.
    """
    shape = slowness.shape
    coordinates = np.indices(shape).reshape(2, -1).T
    distance = np.linalg.norm((coordinates - source) * np.asarray(spacing), axis=1)
    # Frequency-independent amplitude; the common 1/sqrt(8*pi*omega) cancels.
    log_amplitude = -0.5 * np.log(
        np.maximum(distance, min(spacing) / 2) * slowness.ravel()
    )
    from usctbench.operators._marching import conservative_transport

    return conservative_transport(
        tape.order,
        tape.parents,
        tape.times,
        shape,
        np.asarray(spacing),
        log_amplitude,
    )


class RayBornForward:
    """Nonlinear Eikonal/WKB pressure and its distorted-wave Born update map.

    The background is rebuilt at every call. The Born map approximates the
    derivative of continuum wave physics; it is NOT the exact derivative of
    this discretized WKB pressure. Its separately implemented adjoint is exact
    for the frozen Born map. Nonlinear solvers must check actual WKB predictions
    in their line search instead of trusting a linearized residual decrease.
    The model variable is squared slowness in s^2/m^2, not speed.
    """

    def __init__(self, grid, geometry, frequencies_hz, **settings):
        if "background_sound_speed_mps" in settings:
            raise ValueError("the current model supplies the background")
        self.grid, self.geometry = grid, geometry
        self.frequencies_hz = np.asarray(frequencies_hz)
        self.settings = dict(settings)
        self.settings.setdefault("reference_cache", {})
        self.eikonal_solves = 0
        self.background_builds = 0
        self.green_solves = 0
        self.green_matvecs = 0

    def linearize(self, squared_slowness):
        from usctbench.operators.base import Linearization

        model = np.asarray(squared_slowness, dtype=float)
        if (
            model.shape != self.grid.shape
            or not np.all(np.isfinite(model))
            or np.any(model <= 0)
        ):
            raise ValueError("squared slowness must be positive, finite and match grid")
        operator = RayBornOperator(
            self.grid,
            self.geometry,
            self.frequencies_hz,
            background_sound_speed_mps=1 / np.sqrt(model),
            **self.settings,
        )
        try:
            value = operator.background_data()
        finally:
            self.background_builds += 1
            self.eikonal_solves += operator.eikonal_solves
            self.green_solves += operator.green_solves
            self.green_matvecs += operator.green_matvecs
        return Linearization(
            value,
            operator,
            derivative_kind=(
                "discrete_volume_integral_born_derivative"
                if operator.green_backend == "volume_integral"
                else "continuum_born_approximation_not_discrete_wkb_derivative"
            ),
        )

    def forward(self, squared_slowness):
        return self.linearize(squared_slowness).value


class RayBornOperator:
    """Matrix-free real squared-slowness to complex-pressure perturbation map.

    Data shape is (frequency, transmitter, receiver). Background and source
    spectrum are fixed during this linearization. No ground truth is inspected.
    Green fields, not the dense data-by-pixel Jacobian, are cached within a byte
    budget. Source-receiver self channels must be excluded by acquisition masks.
    """

    def __init__(
        self,
        grid: GridSpec,
        geometry: GeometrySpec,
        frequencies_hz: np.ndarray,
        *,
        background_sound_speed_mps: np.ndarray | float = 1500.0,
        exterior_speed_mps: float = 1500.0,
        source_spectrum: np.ndarray | None = None,
        max_cache_bytes: int = 128 * 1024**2,
        reference_cache=None,
        green_backend="eikonal_wkb",
        green_solver_rtol=1e-7,
        green_solver_maxiter=20,
        budget_check=None,
    ):
        if green_backend not in {"eikonal_wkb", "volume_integral"}:
            raise ValueError("green_backend must be eikonal_wkb or volume_integral")
        if not np.isfinite(green_solver_rtol) or not 0 < green_solver_rtol < 1:
            raise ValueError("green_solver_rtol must be between zero and one")
        if (
            isinstance(green_solver_maxiter, (bool, np.bool_))
            or not isinstance(green_solver_maxiter, (int, np.integer))
            or green_solver_maxiter <= 0
        ):
            raise ValueError("green_solver_maxiter must be a positive integer")
        self.green_backend = green_backend
        self.green_solver_rtol, self.green_solver_maxiter = (
            green_solver_rtol,
            green_solver_maxiter,
        )
        self.green_solves, self.green_matvecs = 0, 0
        self.budget_check = budget_check
        self.green_maximum_relative_residual = 0.0
        self.grid = grid
        self.geometry = geometry
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        if (
            self.frequencies_hz.ndim != 1
            or not self.frequencies_hz.size
            or not np.all(np.isfinite(self.frequencies_hz))
            or np.any(self.frequencies_hz <= 0)
        ):
            raise ValueError("frequencies_hz must be a finite positive vector")
        if not np.isfinite(exterior_speed_mps) or exterior_speed_mps <= 0:
            raise ValueError("exterior_speed_mps must be finite and positive")
        background = np.asarray(background_sound_speed_mps, dtype=float)
        if background.ndim == 0:
            background = np.full(grid.shape, background)
        if (
            background.shape != grid.shape
            or not np.all(np.isfinite(background))
            or np.any(background <= 0)
        ):
            raise ValueError(
                "background sound speed must be finite, positive and match the grid"
            )
        self.background = background.copy()
        self.background_squared_slowness = 1 / self.background**2
        self.exterior_speed_mps = float(exterior_speed_mps)
        self.reference_cache = reference_cache if reference_cache is not None else {}
        self.data_shape = (
            len(self.frequencies_hz),
            len(geometry.tx_pos_m),
            len(geometry.rx_pos_m),
        )
        self.n_rays = int(np.prod(self.data_shape))
        self.n_pixels = int(np.prod(grid.shape))
        self.area = float(np.prod(grid.spacing_m))
        self.source_radius_m = float(np.sqrt(self.area / np.pi))
        self.max_cache_bytes = int(max_cache_bytes)
        if self.max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be nonnegative")
        if source_spectrum is None:
            source_spectrum = np.ones(self.data_shape[:2], dtype=complex)
        source_spectrum = np.asarray(source_spectrum, dtype=complex)
        if source_spectrum.shape == (self.data_shape[0],):
            source_spectrum = np.repeat(
                source_spectrum[:, None], self.data_shape[1], axis=1
            )
        if source_spectrum.shape != self.data_shape[:2] or not np.all(
            np.isfinite(source_spectrum)
        ):
            raise ValueError(
                "source_spectrum must have shape (n_frequency, n_tx) or (n_frequency,)"
            )
        self.source_spectrum = source_spectrum.copy()
        all_positions = np.vstack((geometry.tx_pos_m, geometry.rx_pos_m))
        if not np.all(np.isfinite(all_positions)):
            raise ValueError("transducer positions must be finite")
        self.positions, inverse = np.unique(all_positions, axis=0, return_inverse=True)
        self.tx_ids, self.rx_ids = np.split(inverse, [len(geometry.tx_pos_m)])
        coordinates = np.indices(grid.shape).reshape(2, -1).T
        coordinates = np.array(grid.origin_m) + (coordinates + 0.5) * grid.spacing_m
        self.distance = np.linalg.norm(
            self.positions[:, None] - coordinates[None], axis=-1
        )
        self.direct_distance = np.linalg.norm(
            geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
        )
        self.valid_pair_mask = self.direct_distance > 0
        self.delay = np.zeros_like(self.distance)
        self.log_amplitude_ratio = np.zeros_like(self.distance)
        self.direct_delay = np.zeros(self.data_shape[1:])
        self.direct_log_amplitude_ratio = np.zeros(self.data_shape[1:])
        self.green_method = "analytic_hankel"
        self.eikonal_solves = 0
        self._cache = OrderedDict()
        if not np.all(self.background == self.exterior_speed_mps):
            if self.green_backend == "eikonal_wkb":
                self._heterogeneous_fields()
            else:
                self.green_method = "full_green_volume_integral"

    def _heterogeneous_fields(self):
        geometry = GeometrySpec(
            tx_pos_m=self.positions, rx_pos_m=self.geometry.rx_pos_m
        )
        model = EikonalForward(
            self.grid, geometry, background_speed_mps=self.exterior_speed_mps
        )
        slowness = model.extend(
            1 / self.background, background=1 / self.exterior_speed_mps
        )
        water = np.full(model.shape, 1 / self.exterior_speed_mps)
        for index, source in enumerate(model.source_coordinates):
            if self.budget_check is not None:
                self.budget_check()
            key = (
                model.shape,
                tuple(self.grid.spacing_m),
                tuple(source),
                self.exterior_speed_mps,
            )
            if key not in self.reference_cache:
                reference = fast_march(water, self.grid.spacing_m, source)
                reference_log = transport_log_amplitude(
                    reference, water, self.grid.spacing_m, source
                )
                self.reference_cache[key] = (reference.times, reference_log)
                self.eikonal_solves += 1
            reference_times, reference_log = self.reference_cache[key]
            tape = fast_march(slowness, self.grid.spacing_m, source)
            self.eikonal_solves += 1
            delay = tape.times - reference_times
            log_ratio = (
                transport_log_amplitude(tape, slowness, self.grid.spacing_m, source)
                - reference_log
            )
            if not np.all(np.isfinite(log_ratio)) or np.max(np.abs(log_ratio)) > 10:
                raise FloatingPointError(
                    "WKB transport is unstable: refine the grid or use a full-wave backend"
                )
            self.delay[index] = delay.reshape(model.shape)[model.crop].ravel()
            self.log_amplitude_ratio[index] = log_ratio.reshape(model.shape)[
                model.crop
            ].ravel()
            for tx in np.flatnonzero(self.tx_ids == index):
                self.direct_delay[tx] = model.sample_receivers(delay)
                self.direct_log_amplitude_ratio[tx] = model.sample_receivers(log_ratio)
        self.green_method = "eikonal_wkb_transport"

    def green_fields(self, frequency_index: int):
        if frequency_index in self._cache:
            self._cache.move_to_end(frequency_index)
            return self._cache[frequency_index]
        frequency = self.frequencies_hz[frequency_index]
        green = homogeneous_green(
            self.distance, frequency, self.exterior_speed_mps, self.source_radius_m
        )
        green *= np.exp(self.log_amplitude_ratio + 2j * np.pi * frequency * self.delay)
        if self.green_method == "full_green_volume_integral":
            from usctbench.operators.forward.volume_integral import VolumeIntegralGreen

            solver = VolumeIntegralGreen(
                self.grid,
                frequency,
                self.background,
                self.exterior_speed_mps,
                rtol=self.green_solver_rtol,
                maxiter=self.green_solver_maxiter,
                budget_check=self.budget_check,
            )
            try:
                green = solver.fields(green)
            finally:
                self.green_solves += solver.solves
                self.green_matvecs += solver.matvecs
            self.green_maximum_relative_residual = max(
                self.green_maximum_relative_residual, solver.maximum_relative_residual
            )
        if green.nbytes <= self.max_cache_bytes:
            while (
                self._cache
                and sum(g.nbytes for g in self._cache.values()) + green.nbytes
                > self.max_cache_bytes
            ):
                self._cache.popitem(last=False)
            self._cache[frequency_index] = green
        return green

    def background_data(self) -> np.ndarray:
        result = np.empty(self.data_shape, dtype=complex)
        for index, frequency in enumerate(self.frequencies_hz):
            result[index] = homogeneous_green(
                self.direct_distance,
                frequency,
                self.exterior_speed_mps,
                self.source_radius_m,
            ) * np.exp(
                self.direct_log_amplitude_ratio
                + 2j * np.pi * frequency * self.direct_delay
            )
            result[index] *= self.source_spectrum[index, :, None]
            if self.green_method == "full_green_volume_integral":
                fields = self.green_fields(index)
                receivers = homogeneous_green(
                    self.distance[self.rx_ids],
                    frequency,
                    self.exterior_speed_mps,
                    self.source_radius_m,
                )
                potential = (
                    (2 * np.pi * frequency) ** 2
                    * self.area
                    * (
                        self.background_squared_slowness
                        - 1 / self.exterior_speed_mps**2
                    )
                )
                result[index] += (
                    (fields[self.tx_ids] * potential.ravel()) @ receivers.T
                ) * self.source_spectrum[index, :, None]
        return result

    def forward(self, squared_slowness_perturbation: np.ndarray) -> np.ndarray:
        dm = np.asarray(squared_slowness_perturbation)
        if (
            dm.shape != self.grid.shape
            or np.iscomplexobj(dm)
            or not np.all(np.isfinite(dm))
        ):
            raise ValueError("squared-slowness perturbation must be a finite real grid")
        result = np.empty(self.data_shape, dtype=complex)
        for index, frequency in enumerate(self.frequencies_hz):
            green = self.green_fields(index)
            field = (green[self.tx_ids] * dm.ravel()) @ green[self.rx_ids].T
            result[index] = (
                field
                * (2 * np.pi * frequency) ** 2
                * self.area
                * self.source_spectrum[index, :, None]
            )
        return result

    def predict(self, sound_speed_mps: np.ndarray) -> np.ndarray:
        speed = np.asarray(sound_speed_mps, dtype=float)
        if (
            speed.shape != self.grid.shape
            or not np.all(np.isfinite(speed))
            or np.any(speed <= 0)
        ):
            raise ValueError("sound speed must be finite, positive and match the grid")
        return self.background_data() + self.forward(
            1 / speed**2 - self.background_squared_slowness
        )

    def adjoint(self, data: np.ndarray) -> np.ndarray:
        from usctbench.operators.adjoint.ray_born import ray_born_adjoint

        return ray_born_adjoint(self, data)

    def normal_diagonal(self, precision):
        """Exact diagonal of Re(J^H W J), without a dense Jacobian.

        Precision must already exclude validation channels. This makes penalty
        scaling covariant with the pressure/source units, not arbitrary units.
        """
        weights = np.broadcast_to(np.asarray(precision, dtype=float), self.data_shape)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("precision must be finite and nonnegative")
        result = np.zeros(self.n_pixels)
        for index, frequency in enumerate(self.frequencies_hz):
            green = self.green_fields(index)
            tx = np.abs(green[self.tx_ids]) ** 2
            rx = np.abs(green[self.rx_ids]) ** 2
            weighted = (
                weights[index] * np.abs(self.source_spectrum[index, :, None]) ** 2
            )
            result += (
                np.sum(tx * (weighted @ rx), axis=0)
                * (2 * np.pi * frequency) ** 4
                * self.area**2
            )
        return result.reshape(self.grid.shape)
